import argparse
from datetime import datetime
import numpy as np
from copy import deepcopy
import os.path as opath

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import jax, optax, jax.numpy as jnp, diffrax, equinox as eqx
from optax import tree_utils as otu
import nets_jax
import metrics_jax
from metrics_jax import pad2d_constant

pad2d_constant_batched = jax.jit(jax.vmap(pad2d_constant))
from utils_jax import (
    seed_all,
    preprocess_exemplar,
    l2i,
    seed_batch,
    load_gif,
    size_of_model,
)
from reg_lib_jax import create_regularization_fns


parser = argparse.ArgumentParser("ODE texture")

parser.add_argument(
    "--solver", type=str, default="heun", choices=["euler", "tsit5", "heun"]
)
parser.add_argument("--tol", type=float, default=1e-2)
parser.add_argument("--step_size", type=float, default=1e-2)
parser.add_argument("--adjoint", action="store_true")

parser.add_argument("--n_aug_channels", type=int, default=9)
parser.add_argument("--dim", type=int, default=32)
parser.add_argument("--dim_mults", type=str, default="1,2,4")
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

parser.add_argument("--n_iter", type=int, default=50000)
parser.add_argument("--test_freq", type=int, default=100)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-4)

parser.add_argument("--exemplars_path", type=str)
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--size", type=int, default=128)
parser.add_argument("--exp_path", type=str)
parser.add_argument("--comment", type=str, default="")

args = parser.parse_args()

if __name__ == "__main__":
    # reproducibility
    key = seed_all(42)

    # jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32

    # for recording
    workspace = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ws_path = opath.join(args.exp_path, workspace if not args.comment else args.comment)
    writer = SummaryWriter(log_dir=ws_path)

    # load data
    exemplars, time_between = load_gif(args.exemplars_path)
    n_frames = len(exemplars)

    assert (
        time_between == 0.05
    ), f"time between frames is {time_between}, but we expect 0.05"

    exemplars = [preprocess_exemplar(e, (args.size, args.size)) for e in exemplars]

    ## to numpy
    exemplars_np = np.stack(
        [np.array(e, dtype=dtype).transpose(2, 0, 1) / 255.0 for e in exemplars]
    )
    writer.add_images("exemplars", exemplars_np)

    # create model
    key, subkey = jax.random.split(key)
    n_channels = 3 + args.n_aug_channels
    odefunc = nets_jax.NDAE(
        args.dim,
        in_dim=n_channels,
        out_dim=n_channels,
        dim_mults=[int(m) for m in args.dim_mults.split(",")],
        attn_heads=args.n_attn_heads,
        attn_head_dim=args.attn_head_dim,
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
            pcoeff=0.0,
            icoeff=1.0,
            dcoeff=0,
        )
    elif args.solver == "heun":
        solver = diffrax.Heun()
        dt0 = None
        stepsize_controller = diffrax.PIDController(
            rtol=args.tol,
            atol=args.tol,
            pcoeff=0.0,
            icoeff=1.0,
            dcoeff=0,
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
    overflow_lossfn = metrics_jax.create_overflow_loss(0.0, 1.0)

    transform = optax.contrib.reduce_on_plateau(factor=0.5, patience=5)
    optimizer = optax.adam(args.lr)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    tsfm_state = transform.init(eqx.filter(model, eqx.is_inexact_array))
    step_records = []

    # loading checkpoints if required
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path

        if opath.exists(opath.join(checkpoint_path, "model.eqx")):
            print("loading model from checkpoint")
            model = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "model.eqx"), model
            )

        if opath.exists(opath.join(checkpoint_path, "states.eqx")):
            print("loading states from checkpoint")
            [opt_state, tsfm_state] = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "states.eqx"), [opt_state, tsfm_state]
            )

        if opath.exists(opath.join(checkpoint_path, "diffeqsolve_args.eqx")):
            print("loading solver arguments from checkpoint")
            diffeqsolve_args = eqx.tree_deserialise_leaves(
                opath.join(checkpoint_path, "diffeqsolve_args.eqx"), diffeqsolve_args
            )

    n_iter = args.n_iter
    disp_iter = n_iter // args.test_freq

    to_rgb = lambda x: l2i(x[..., :3, :, :])

    @eqx.filter_jit
    def step(model, opt_state, tsfm_state, xs, T0, T1, key, target, diffeqsolve_args):
        reg_key, loss_key = jax.random.split(key)

        def calc_loss(model, xs, T0, T1):
            solutions = model(T0, T1, xs, get_reg=True, key=reg_key, **diffeqsolve_args)
            stats = solutions.stats

            xst1 = solutions.ys["x"][-1]  # solutions.ys["x"] is (T, B, C, H, W)
            images = to_rgb(xst1)
            losses = jax.vmap(loss_fn, in_axes=(None, None, 0, 0))(
                vgg,
                target,
                images.clip(0.0, 1.0),
                jax.random.split(loss_key, args.batchsize),
            )
            style_loss = jnp.mean(losses, axis=0)

            overflow_loss = overflow_lossfn(images)

            reg_loss = 0.0
            if reg_fns:
                reg = solutions.ys["reg"][-1]
                reg_loss += jnp.dot(reg_weights, reg)

            return style_loss + 1e2 * overflow_loss + reg_loss, (
                style_loss,
                overflow_loss,
                reg_loss,
                xst1,
                images,
                stats,
            )

        (
            _,
            (style_loss, overflow_loss, reg_loss, xst1, images, stats),
        ), grads = eqx.filter_value_and_grad(calc_loss, has_aux=True)(model, xs, T0, T1)
        grads = jax.tree.map(lambda g: g / (jnp.linalg.norm(g) + 1e-8), grads)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        updates = otu.tree_scalar_mul(tsfm_state.scale, updates)
        model = eqx.apply_updates(model, updates)

        return (
            style_loss,
            overflow_loss,
            reg_loss,
            model,
            opt_state,
            xst1,
            images,
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

    refresh_rate = 6
    it_refresh = 0
    for it in tqdm(range(n_iter), desc="Iter"):

        if it_refresh % refresh_rate == 0:
            key, subkey = jax.random.split(key)
            xs = seed_batch(subkey, args.batchsize, n_channels, args.size)
            T0, T = -syn_t, syn_t
        else:
            if it_refresh % refresh_rate == 1:
                key, subkey = jax.random.split(key)
                Ts = stratified_uniform(subkey, refresh_rate - 1, 0.0, duration)
                Ts = Ts.at[1:].set(Ts[1:] - Ts[:-1])
            T = Ts[it_refresh % refresh_rate - 1]

        T1 = T0 + T

        target_index = get_idx(T1)
        target = exemplars_np[target_index]
        step_records.append(target_index)

        # print(f"iter: {it}, T0: {T0}, T1: {T1}, target_indices: {target_indices}")
        key, subkey = jax.random.split(key)
        (
            style_loss,
            overflow_loss,
            reg_loss,
            model,
            opt_state,
            xst1,
            images,
            stats,
        ) = step(
            model, opt_state, tsfm_state, xs, T0, T1, subkey, target, diffeqsolve_args
        )

        T0 = T1
        xs = xst1
        it_refresh += 1

        # log at each iteration
        writer.add_scalar("loss/style_loss", np.array(style_loss), it)
        writer.add_scalar("loss/overflow_loss", np.array(overflow_loss), it)
        writer.add_scalar("loss/reg_loss", np.array(reg_loss), it)
        writer.add_scalar("stats/num_steps", np.array(stats["num_steps"] / T), it)
        writer.add_scalar(
            "stats/num_accepted_steps", np.array(stats["num_accepted_steps"] / T), it
        )
        writer.add_scalar(
            "stats/num_rejected_steps", np.array(stats["num_rejected_steps"] / T), it
        )
        writer.add_images("test_tex", np.array(pad2d_constant_batched(images)), it)

        # log per some iterations
        # do an inference, and try to use the inference loss to adaptively update lr
        if it == 0 or (it + 1) % disp_iter == 0:
            key, inference_key, loss_key = jax.random.split(key, 3)
            # inference
            n_inference = 3
            inference_args = deepcopy(diffeqsolve_args)
            inference_args.update(
                {"saveat": diffrax.SaveAt(ts=jnp.linspace(0.0, duration, n_frames))}
            )
            solutions = inference(
                inference_key, model, n_inference, args.size, inference_args
            )
            images = to_rgb(solutions.ys["x"])
            stats = solutions.stats
            num_accepted_steps = stats["num_accepted_steps"]

            # inference loss
            loss_keys = jax.vmap(jax.random.split, in_axes=(0, None))(
                jax.random.split(loss_key, n_frames), n_inference
            )
            inference_losses = jax.jit(
                jax.vmap(
                    jax.vmap(loss_fn, in_axes=(None, None, 0, 0)),
                    in_axes=(None, 0, 0, 0),
                )
            )(vgg, exemplars_np, images.clip(0.0, 1.0), loss_keys)
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
                tsfm_state = eqx.tree_at(lambda x: x.scale, tsfm_state, min_lr / args.lr)

            writer.add_histogram("step_records", np.array(step_records), it)
            writer.add_scalar(
                "stats/inference_num_accepted_steps", np.array(num_accepted_steps), it
            )
            writer.add_scalar("loss/inference_loss", np.array(inference_loss), it)
            writer.add_scalar("lr_scale", np.array(tsfm_state.scale), it)
            writer.add_images(
                "inference_images",
                np.array(pad2d_constant_batched(images[:, 0, ...])),
                it,
            )

            eqx.tree_serialise_leaves(opath.join(ws_path, "model.eqx"), model)
            eqx.tree_serialise_leaves(
                opath.join(ws_path, "states.eqx"), [opt_state, tsfm_state]
            )
            eqx.tree_serialise_leaves(
                opath.join(ws_path, "diffeqsolve_args.eqx"), diffeqsolve_args
            )

    ts_synthesis = jnp.logspace(0.0, jnp.log10(1.0 + syn_t), 50) - 1.0 - syn_t
    ts_transition = jnp.linspace(0.0, duration, n_frames)
    ts_samples = jnp.concatenate([ts_synthesis, ts_transition])

    sampling_args = deepcopy(diffeqsolve_args)
    sampling_args.update({"saveat": diffrax.SaveAt(ts=ts_samples)})

    solutions = inference(key, model, 1, args.size * 2, sampling_args)

    images = to_rgb(solutions.ys["x"])
    images_synthesis = np.array(
        images[: len(ts_synthesis), ...].transpose(1, 0, 2, 3, 4)
    )
    images_transition = np.array(
        images[len(ts_synthesis) :, ...].transpose(1, 0, 2, 3, 4)
    )
    writer.add_video(
        "anime_synthesis",
        images_synthesis,
    )

    padded_exemplars = np.zeros(images_transition.shape)
    padded_exemplars[..., : args.size, : args.size] = exemplars_np
    writer.add_video(
        "anime",
        np.concatenate([images_transition, padded_exemplars]),
    )

    writer.close()
