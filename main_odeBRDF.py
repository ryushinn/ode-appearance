import argparse
from datetime import datetime
import numpy as np
from functools import partial
from copy import deepcopy

import os.path as opath
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import jax, optax, jax.numpy as jnp, diffrax
from optax import tree_utils as otu
import nets_jax
import metrics_jax
from metrics_jax import pad2d_constant

pad2d_constant_batched = jax.jit(jax.vmap(pad2d_constant))
from utils_jax import seed_all, seed_batch, read_images, size_of_model
import equinox as eqx
from reg_lib_jax import create_regularization_fns

from renderer import render, tonemapping
from rendering_utils import (
    random_take,
    random_crop,
    select_renderer,
    to_maps,
    clip_maps,
    stitch_maps,
    height_to_normal,
)


parser = argparse.ArgumentParser("ODE BRDF")

parser.add_argument(
    "--solver", type=str, default="heun", choices=["euler", "tsit5", "heun"]
)
parser.add_argument("--tol", type=float, default=1e-2)
parser.add_argument("--step_size", type=float, default=1e-2)
parser.add_argument("--adjoint", action="store_true")

parser.add_argument("--n_aug_channels", type=int, default=9)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--dim_mults", type=str, default="1,2")
parser.add_argument("--n_attn_heads", type=int, default=4)
parser.add_argument("--attn_head_dim", type=int, default=8)


parser.add_argument("--loss_type", type=str, choices=["GRAM", "SW"], default="SW")


# Regularizations
parser.add_argument(
    "--total-derivative", type=float, default=None, help="df(z(t), t)/dt"
)
parser.add_argument(
    "--kinetic-energy", type=float, default=None, help="int_t ||f||_2^2"
)
parser.add_argument(
    "--jacobian-norm2", type=float, default=None, help="int_t ||df/dx||_F^2"
)

parser.add_argument("--renderer", type=str)

parser.add_argument("--n_iter", type=int, default=60000)
parser.add_argument("--n_init_iter", type=int, default=20000)
parser.add_argument("--disp_iter", type=int, default=500)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-4)

parser.add_argument("--exemplars_path", type=str)
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--crop_size", type=int, default=128)
parser.add_argument("--exp_path", type=str)
parser.add_argument("--comment", type=str, default="")

args = parser.parse_args()

if __name__ == "__main__":
    # reproducibility
    key = seed_all(42)
    # jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32

    # logging
    workspace = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ws_path = opath.join(args.exp_path, workspace if not args.comment else args.comment)
    writer = SummaryWriter(log_dir=ws_path)

    # load data
    camera = {"fov": 50, "distance": 1.0}
    flash_light = {"intensity": jnp.log(jnp.array(1.0)), "xy-position": (0.0, 0.0)}
    renderer_pp, clip_maps, n_BRDF_channels = select_renderer(args.renderer)
    render = partial(
        render,
        camera=camera,
        renderer_pp=renderer_pp,
    )
    exemplars = read_images(
        args.exemplars_path, n_images=100, resize=(args.size, args.size)
    )
    n_frames, c, h, w = exemplars.shape
    time_between = 0.1
    print(f"Loaded {n_frames} images of size {c}x{h}x{w}")
    writer.add_images(
        "exemplars", np.array(exemplars)[[0, n_frames // 2, n_frames - 1], ...]
    )

    # create model
    n_normal_channels = 1  # we model normal map as a single channel height map
    n_channels = n_BRDF_channels + n_normal_channels + args.n_aug_channels
    to_maps = partial(
        to_maps, n_BRDF_channels=n_BRDF_channels, n_normal_channels=n_normal_channels
    )
    vheight_to_normal = jax.jit(jax.vmap(height_to_normal, in_axes=(0)))
    vvheight_to_normal = jax.jit(jax.vmap(vheight_to_normal, in_axes=(0)))
    key, subkey = jax.random.split(key)
    odefunc = nets_jax.NDAE(
        args.dim,
        in_dim=n_channels,
        out_dim=n_channels,
        dim_mults=[int(m) for m in args.dim_mults.split(",")],
        attn_heads=args.n_attn_heads,
        attn_head_dim=args.attn_head_dim,
        use_attn=False,
        key=subkey,
    )

    reg_fns, reg_weights = create_regularization_fns(args)
    reg_weights = jnp.array(reg_weights)
    model = nets_jax.NeuralODE(odefunc, reg_fns)
    size_of_model(model)

    if args.solver == "euler":
        solver = diffrax.Euler()
        dt0 = args.step_size
        stepsize_controller = diffrax.ConstantStepSize()
    elif args.solver == "tsit5":
        solver = diffrax.Tsit5()
        dt0 = None
        stepsize_controller = diffrax.PIDController(
            rtol=args.tol,
            atol=args.tol,
        )
    elif args.solver == "heun":
        solver = diffrax.Heun()
        dt0 = None
        stepsize_controller = diffrax.PIDController(
            rtol=args.tol,
            atol=args.tol,
        )
    else:
        raise NotImplementedError(f"Solver {args.solver} not implemented")

    if args.adjoint:
        adjoint = diffrax.BacksolveAdjoint()
    else:
        adjoint = diffrax.RecursiveCheckpointAdjoint()

    diffeqsolve_args = {
        "solver": solver,
        "dt0": dt0,
        "stepsize_controller": stepsize_controller,
        "adjoint": adjoint,
    }

    # training config
    vgg = metrics_jax.load_pretrained_VGG19_from_pth(
        "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth", dtype=dtype
    )

    if args.loss_type == "GRAM":
        loss_fn = metrics_jax.gram_loss
    elif args.loss_type == "SW":
        loss_fn = metrics_jax.slice_loss
        # When training, we use a weighted loss function
        # loss_fn = partial(loss_fn, _weights=[3, 64, 128, 256, 512, 512])

    transform = optax.contrib.reduce_on_plateau(factor=0.5, patience=5)
    optimizer = optax.adam(args.lr)

    opt_state = optimizer.init(eqx.filter((model, flash_light), eqx.is_inexact_array))
    tsfm_state = transform.init(eqx.filter((model, flash_light), eqx.is_inexact_array))
    step_records = []

    ## LOAD CHECKPOINT IF REQUIRED
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path

        if opath.exists(opath.join(checkpoint_path, "model.eqx")):
            print("loading model from checkpoint")
            model = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "model.eqx"), model
            )

        if opath.exists(opath.join(checkpoint_path, "light.eqx")):
            print("loading light parameters from checkpoint")
            flash_light = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "light.eqx"), flash_light
            )

        if opath.exists(opath.join(checkpoint_path, "states.eqx")):
            print("loading states from checkpoint")
            [opt_state, tsfm_state] = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "states.eqx"), [opt_state, tsfm_state]
            )

    n_iter = args.n_iter
    disp_iter = args.disp_iter

    random_crop = partial(random_crop, crop_h=args.crop_size, crop_w=args.crop_size)
    random_take = partial(random_take, new_h=args.crop_size, new_w=args.crop_size)

    @eqx.filter_jit
    def step(key, optimizable, xs, T0, T1, opt_state, tsfm_state, target, init=False):
        # loss function
        n_crops = 32
        _random_crop = random_take if init else random_crop
        cropkey, losskey = jax.random.split(key)
        cropkeys = jax.random.split(cropkey, n_crops)

        cropped_targets = jax.vmap(_random_crop, in_axes=(0, None))(cropkeys, target)

        def lossfn(optimizable):
            model, flash_light = optimizable

            def _render(BRDF_maps, normal_map, key):
                return render(
                    BRDF_maps,
                    normal_map,
                    flash_light=flash_light,
                    region={"H": h, "W": w, "crop": _random_crop, "cropkey": key},
                )

            vvrender = jax.vmap(
                jax.vmap(_render, in_axes=(0, 0, None)), in_axes=(None, None, 0)
            )

            solutions = model(T0, T1, xs, **diffeqsolve_args)
            stats = solutions.stats
            xst1 = solutions.ys["x"][-1]

            BRDF_maps, height_map = to_maps(xst1)
            if init:
                pseudo_height_map = jnp.zeros_like(height_map)
                normal_map = vheight_to_normal(pseudo_height_map)
            else:
                normal_map = vheight_to_normal(height_map)

            renderings = vvrender(clip_maps(BRDF_maps), normal_map, cropkeys)
            normal_loss = 0.0
            if init:
                rendering_loss = 0.0
                init_loss = jnp.mean(
                    (tonemapping(renderings) - cropped_targets[:, None, ...]) ** 2
                )
                init_loss += jnp.mean((height_map - pseudo_height_map) ** 2)
            else:
                rendering_losses = jax.vmap(
                    jax.vmap(loss_fn, in_axes=(None, None, 0, 0)),
                    in_axes=(None, 0, 0, None),
                )(
                    vgg,
                    cropped_targets,
                    tonemapping(renderings),
                    jax.random.split(losskey, args.batchsize),
                )
                rendering_loss = jnp.mean(rendering_losses)
                init_loss = 0.0

            overflow_loss = jnp.mean((BRDF_maps - clip_maps(BRDF_maps)) ** 2)

            return rendering_loss + 1e2 * overflow_loss + normal_loss + init_loss, (
                rendering_loss,
                overflow_loss,
                normal_loss,
                init_loss,
                xst1,
                BRDF_maps,
                normal_map,
                renderings,
                stats,
            )

        grads, (
            rendering_loss,
            overflow_loss,
            normal_loss,
            init_loss,
            xst1,
            BRDF_maps,
            normal_map,
            renderings,
            stats,
        ) = eqx.filter_grad(lossfn, has_aux=True)(optimizable)

        grads = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        updates, opt_state = optimizer.update(grads, opt_state, optimizable)
        updates = otu.tree_scalar_mul(tsfm_state.scale, updates)
        optimizable = eqx.apply_updates(optimizable, updates)

        return (
            rendering_loss,
            overflow_loss,
            normal_loss,
            init_loss,
            optimizable,
            opt_state,
            xst1,
            BRDF_maps,
            normal_map,
            renderings,
            cropped_targets,
            stats,
        )

    ## Duration of the input
    duration = n_frames * time_between

    ## synthesis duration
    syn_t = duration * 0.2

    @eqx.filter_jit
    def inference(key, model, n, size, diffeqsolve_args):
        xs = seed_batch(key, n, n_channels, size)
        solutions = model(
            -syn_t, duration, xs, get_reg=False, key=None, **diffeqsolve_args
        )
        return solutions

    def get_idx(T):
        return jnp.minimum(
            jnp.maximum(T / time_between, 0.0).astype(jnp.int32), n_frames - 1
        )

    def stratified_uniform(key, n, minval, maxval):
        length = (maxval - minval) / n
        return (
            jax.random.uniform(key, shape=(n,)) * length
            + minval
            + length * jnp.arange(n)
        )

    refresh_rate = 2
    it_refresh = 0
    init = True
    watershed = args.n_init_iter
    for it in tqdm(range(n_iter), desc="Iter"):
        if it == watershed:
            # when the training is stable, we can increase the refresh rate
            refresh_rate = 6
            it_refresh = 0
            init = False
            # renew the optimizer with new states
            opt_state = optimizer.init(
                eqx.filter((model, flash_light), eqx.is_inexact_array)
            )

        if it_refresh % refresh_rate == 0:
            key, subkey = jax.random.split(key)
            xs = seed_batch(subkey, args.batchsize, n_channels, args.crop_size)
            T0, T = -syn_t, syn_t
        else:
            if it_refresh % refresh_rate == 1:
                key, subkey = jax.random.split(key)
                Ts = stratified_uniform(subkey, refresh_rate - 1, 0.0, duration)
                Ts = Ts.at[1:].set(Ts[1:] - Ts[:-1])
            T = Ts[it_refresh % refresh_rate - 1]

        T1 = T0 + T

        target_index = get_idx(T1)
        target = exemplars[target_index]
        step_records.append(target_index)

        # print(f"iter: {it}, T0: {T0}, T1: {T1}, target_indices: {target_indices}")
        key, subkey = jax.random.split(key)
        (
            rendering_loss,
            overflow_loss,
            normal_loss,
            init_loss,
            (model, flash_light),
            opt_state,
            xst1,
            BRDF_maps,
            normal_map,
            renderings,
            cropped_targets,
            stats,
        ) = step(
            subkey,
            (model, flash_light),
            xs,
            T0,
            T1,
            opt_state,
            tsfm_state,
            target,
            init,
        )

        T0 = T1
        xs = xst1
        it_refresh += 1

        # log at each iteration
        if not init:
            writer.add_scalar("loss/rendering_loss", np.array(rendering_loss), it)
        writer.add_scalar("loss/overflow_loss", np.array(overflow_loss), it)
        writer.add_scalar("loss/normal_loss", np.array(normal_loss), it)
        if init:
            writer.add_scalar("loss/init_loss", np.array(init_loss), it)
        writer.add_scalar("stats/num_steps", np.array(stats["num_steps"] / T), it)
        writer.add_scalar(
            "stats/num_accepted_steps", np.array(stats["num_accepted_steps"] / T), it
        )
        writer.add_scalar(
            "stats/num_rejected_steps", np.array(stats["num_rejected_steps"] / T), it
        )
        writer.add_scalar(
            "lighting/intensity", np.array(jnp.exp(flash_light["intensity"])), it
        )
        writer.add_scalar(
            "lighting/x-position",
            np.array(flash_light["xy-position"][0]),
            it,
        )
        writer.add_scalar(
            "lighting/y-position",
            np.array(flash_light["xy-position"][1]),
            it,
        )

        # log per some iterations
        # do an inference, and try to use the inference loss to adaptively update lr
        if it == 0 or (it + 1) % disp_iter == 0 or it == watershed:
            stitches = jax.vmap(
                jax.vmap(stitch_maps, in_axes=(None, 0, 0, 0)),
                in_axes=(0, 0, None, None),
            )(cropped_targets, tonemapping(renderings), BRDF_maps, normal_map)
            # only show the first 4 crops and the first one in batch
            stitches = stitches[:4, 0, ...]
            writer.add_images(
                (
                    "training/initializing"
                    if it <= watershed
                    else "training/target-rendering-BRDF"
                ),
                np.array(pad2d_constant_batched(stitches)),
                it,
            )

            if it >= watershed:
                key, inference_key, loss_key = jax.random.split(key, 3)
                # inference
                n_inference = 1
                inference_args = deepcopy(diffeqsolve_args)
                inference_args.update(
                    {"saveat": diffrax.SaveAt(ts=jnp.linspace(0.0, duration, n_frames))}
                )
                solutions = inference(
                    inference_key, model, n_inference, args.size, inference_args
                )
                BRDF_maps, height_map = to_maps(solutions.ys["x"])
                normal_map = vvheight_to_normal(height_map)
                stats = solutions.stats
                num_accepted_steps = stats["num_accepted_steps"]

                vrender_from_maps = jax.vmap(
                    jax.vmap(
                        partial(
                            render,
                            flash_light=flash_light,
                        )
                    )
                )
                renderings = vrender_from_maps(clip_maps(BRDF_maps), normal_map)

                # inference loss
                loss_keys = jax.vmap(jax.random.split, in_axes=(0, None))(
                    jax.random.split(loss_key, n_frames), n_inference
                )
                inference_losses = jax.jit(
                    jax.vmap(
                        jax.vmap(loss_fn, in_axes=(None, None, 0, 0)),
                        in_axes=(None, 0, 0, 0),
                    )
                )(vgg, exemplars, tonemapping(renderings), loss_keys)
                inference_loss = jnp.mean(inference_losses)

                # adaptively update learning rate
                _, tsfm_state = transform.update(
                    updates=eqx.filter(model, eqx.is_inexact_array),
                    state=tsfm_state,
                    value=inference_loss,
                )
                # minimal lr will be 1e-4
                min_lr = 1e-4
                if tsfm_state.scale < (min_lr / args.lr):
                    tsfm_state = eqx.tree_at(
                        lambda x: x.scale, tsfm_state, min_lr / args.lr
                    )
                writer.add_scalar("loss/inference_loss", np.array(inference_loss), it)
                writer.add_scalar(
                    "stats/inference_num_accepted_steps",
                    np.array(num_accepted_steps),
                    it,
                )

                writer.add_histogram("step_records", np.array(step_records), it)
                writer.add_scalar("lr_scale", np.array(tsfm_state.scale), it)

                eqx.tree_serialise_leaves(opath.join(ws_path, "model.eqx"), model)
                eqx.tree_serialise_leaves(opath.join(ws_path, "light.eqx"), flash_light)
                eqx.tree_serialise_leaves(
                    opath.join(ws_path, "states.eqx"), [opt_state, tsfm_state]
                )

    ts_synthesis = jnp.logspace(0.0, jnp.log10(1.0 + syn_t), 50) - 1.0 - syn_t
    ts_transition = jnp.linspace(0.0, duration, n_frames)
    ts_samples = jnp.concatenate([ts_synthesis, ts_transition])

    sampling_args = deepcopy(diffeqsolve_args)
    sampling_args.update({"saveat": diffrax.SaveAt(ts=ts_samples)})

    solutions = inference(key, model, 1, args.size, sampling_args)

    BRDF_maps_synthesis, height_map_synthesis = to_maps(
        solutions.ys["x"][: len(ts_synthesis)]
    )
    BRDF_maps_transition, height_map_transition = to_maps(
        solutions.ys["x"][len(ts_synthesis) :]
    )
    normal_map_synthesis = vvheight_to_normal(height_map_synthesis)
    normal_map_transition = vvheight_to_normal(height_map_transition)

    vrender = jax.jit(jax.vmap(jax.vmap(partial(render, flash_light=flash_light))))

    images_synthesis = tonemapping(
        vrender(clip_maps(BRDF_maps_synthesis), normal_map_synthesis)
    )
    images_transition = tonemapping(
        vrender(clip_maps(BRDF_maps_transition), normal_map_transition)
    )
    stitches = jax.vmap(jax.vmap(stitch_maps, in_axes=(None, 0, 0, 0)))(
        exemplars, images_transition, BRDF_maps_transition, normal_map_transition
    )

    writer.add_video(
        "inference/dynamic_BRDF", np.array(stitches).transpose(1, 0, 2, 3, 4)
    )
    writer.add_video(
        "other/synthesis",
        np.array(images_synthesis).transpose(1, 0, 2, 3, 4),
    )
    writer.close()
