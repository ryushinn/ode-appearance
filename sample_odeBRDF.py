import argparse
from datetime import datetime
import numpy as np
import os
from PIL import Image
import subprocess
from functools import partial
from copy import deepcopy
from rendering_utils import height_to_normal

import os.path as opath

import jax, jax.numpy as jnp, diffrax
import nets_jax
from metrics_jax import pad2d_constant

pad2d_constant_batched = jax.jit(jax.vmap(pad2d_constant))
from utils_jax import seed_all, seed_batch, read_images
import equinox as eqx

from copy import deepcopy
from einops import rearrange
from renderer import render, tonemapping
from rendering_utils import (
    to_maps,
    clip_maps,
    stitch_maps,
    height_to_normal,
    select_renderer,
)


parser = argparse.ArgumentParser("ODE BRDF")

parser.add_argument(
    "--solver", type=str, default="heun", choices=["euler", "tsit5", "heun"]
)
parser.add_argument("--tol", type=float, default=1e-2)
parser.add_argument("--step_size", type=float, default=1e-2)

parser.add_argument("--n_aug_channels", type=int, default=9)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--dim_mults", type=str, default="1,2")
parser.add_argument("--n_attn_heads", type=int, default=4)
parser.add_argument("--attn_head_dim", type=int, default=8)

parser.add_argument("--renderer", type=str)
parser.add_argument("--exemplars_path", type=str)
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--size", type=int, default=256)
parser.add_argument("--sample_size", type=int, default=256)
parser.add_argument("--exp_path", type=str)
parser.add_argument("--comment", type=str, default="")

args = parser.parse_args()


def save_images(path, images):
    # images in size of (T, B, C, H, W)
    # make it (T, C, H, W)
    images = images[:, 0, ...]
    os.makedirs(path, exist_ok=True)

    frames = []
    for t in range(images.shape[0]):  # Loop through time steps
        # Convert to (H, W, C) format for RGB images
        img = images[t].transpose(1, 2, 0)

        # Ensure the pixel values are appropriate for Pillow (e.g., 0-255 for uint8)
        img = (img * 255).astype(np.uint8)

        # Convert to Pillow Image and save
        if img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)
            pil_img = Image.fromarray(img, mode="L")
        else:
            pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(path, f"frames_{t:04d}.png"))
        frames.append(pil_img)
    frames[0].save(
        opath.join(path, f"animations.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        f"{opath.join(path, f'animations.gif')}",
        "-c:v",
        "rawvideo",
        "-r",
        f"{1000 / 100}",
        f"{opath.join(path, f'animations.avi')}",
    ]
    subprocess.run(
        ffmpeg_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        f"{opath.join(path, f'animations.gif')}",
        # frame rate
        "-r",
        f"{1000 / 100}",
        "-c:v",
        "libx264",
        # Good enough quality
        "-crf",
        "23",
        "-preset",
        "slow",
        # Compatibility with most browsers and players
        "-pix_fmt",
        "yuv420p",
        f"{opath.join(path, f'animations.mp4')}",
    ]
    subprocess.run(
        ffmpeg_command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    # reproducibility
    key = seed_all(42)

    # jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32

    # for recording
    workspace = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ws_path = opath.join(args.exp_path, workspace if not args.comment else args.comment)
    if not opath.exists(ws_path):
        os.makedirs(ws_path, exist_ok=True)

    # load data
    camera = {"fov": 50, "distance": 1.0}
    flash_light = {"intensity": jnp.log(jnp.array(1.0)), "xy-position": (0.0, 0.0)}
    renderer_pp, clip_maps, n_BRDF_channels = select_renderer(args.renderer)
    render = partial(
        render,
        renderer_pp=renderer_pp,
    )
    exemplars = read_images(
        args.exemplars_path, n_images=100, resize=(args.size, args.size)
    )
    n_frames, c, h, w = exemplars.shape

    # save exemplars only once when sampling iso-cook-torrance
    if args.renderer == "diffuse_iso_cook_torrance":
        save_images(opath.join(ws_path, "exemplars"), np.array(exemplars[:, None, ...]))
        if "synthetic" in args.exemplars_path:
            print("this is a synthetic data, also sampling the relit exemplar images")
            exemplars_relit_up = read_images(
                args.exemplars_path + "_relit_up",
                n_images=100,
                resize=(args.size, args.size),
            )
            exemplars_relit_left = read_images(
                args.exemplars_path + "_relit_left",
                n_images=100,
                resize=(args.size, args.size),
            )
            exemplars_relit_rightdown = read_images(
                args.exemplars_path + "_relit_rightdown",
                n_images=100,
                resize=(args.size, args.size),
            )
            save_images(
                opath.join(ws_path, "exemplars_relit_up"),
                np.array(exemplars_relit_up[:, None, ...]),
            )
            save_images(
                opath.join(ws_path, "exemplars_relit_left"),
                np.array(exemplars_relit_left[:, None, ...]),
            )
            save_images(
                opath.join(ws_path, "exemplars_relit_rightdown"),
                np.array(exemplars_relit_rightdown[:, None, ...]),
            )

    time_between = 0.1
    print(f"Loaded {n_frames} images of size {c}x{h}x{w}")

    # create model
    n_normal_channels = 1  # we model normal map as a single channel height map
    n_channels = n_BRDF_channels + n_normal_channels + args.n_aug_channels
    to_maps = partial(
        to_maps, n_BRDF_channels=n_BRDF_channels, n_normal_channels=n_normal_channels
    )
    height_to_normal = jax.vmap(jax.vmap(height_to_normal, in_axes=(0)), in_axes=(0))
    key, subkey = jax.random.split(key)
    odefunc = nets_jax.NDAE(
        args.dim,
        in_dim=n_channels,
        out_dim=n_channels,
        dim_mults=[int(m) for m in args.dim_mults.split(",")],
        use_attn=False,
        attn_heads=args.n_attn_heads,
        attn_head_dim=args.attn_head_dim,
        key=subkey,
    )
    reg_fns = ()
    model = nets_jax.NeuralODE(odefunc, reg_fns)

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

    diffeqsolve_args = {
        "solver": solver,
        "dt0": dt0,
        "stepsize_controller": stepsize_controller,
    }

    checkpoint_path = args.checkpoint_path

    print("loading model from checkpoint")
    model = eqx.tree_deserialise_leaves(
        opath.join(checkpoint_path, "model.eqx"), model
    )

    print("loading light parameters from checkpoint")
    flash_light = eqx.tree_deserialise_leaves(
        opath.join(checkpoint_path, "light.eqx"), flash_light
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

    ts_synthesis = jnp.logspace(0.0, jnp.log10(1.0 + syn_t), 50) - 1.0 - syn_t
    ts_transition = jnp.linspace(0.0, duration, n_frames)
    ts_samples = jnp.concatenate([ts_synthesis, ts_transition])

    sampling_args = deepcopy(diffeqsolve_args)
    sampling_args.update({"saveat": diffrax.SaveAt(ts=ts_samples)})

    solutions = inference(key, model, 1, args.sample_size, sampling_args)

    BRDF_maps_synthesis, height_map_synthesis = to_maps(
        solutions.ys["x"][: len(ts_synthesis)]
    )
    BRDF_maps_transition, height_map_transition = to_maps(
        solutions.ys["x"][len(ts_synthesis) :]
    )
    clipped_BRDF_maps_transition = clip_maps(BRDF_maps_transition)
    normal_map_transition = height_to_normal(height_map_transition)

    vrender = jax.jit(
        jax.vmap(jax.vmap(partial(render, camera=camera, flash_light=flash_light)))
    )
    images_transition = tonemapping(
        vrender(clipped_BRDF_maps_transition, normal_map_transition)
    )

    vrender_relit_rotating = jax.jit(
        jax.vmap(
            jax.vmap(render, in_axes=(0, 0, None, None)),
            in_axes=(0, 0, None, {"intensity": None, "xy-position": 0}),
        )
    )
    speed = 0.5
    xy_position_rotating = (
        0.8
        * camera["distance"]
        * jnp.tan(jnp.deg2rad(camera["fov"] / 2))
        * jnp.stack(
            [
                jnp.sin(speed * ts_transition * 2 * jnp.pi),
                jnp.cos(speed * ts_transition * 2 * jnp.pi),
            ],
            axis=-1,
        )
    )
    relit_flash_light_rotating = {
        "intensity": flash_light["intensity"],
        "xy-position": xy_position_rotating,
    }
    images_transition_relit_rotating = tonemapping(
        vrender_relit_rotating(
            clipped_BRDF_maps_transition,
            normal_map_transition,
            camera,
            relit_flash_light_rotating,
        )
    )

    relit_up_flash_light = {
        "intensity": flash_light["intensity"],
        "xy-position": jnp.array((0.0, 0.6)),
    }
    relit_left_flash_light = {
        "intensity": flash_light["intensity"],
        "xy-position": jnp.array((-0.5, 0.0)),
    }
    relit_rightdown_flash_light = {
        "intensity": flash_light["intensity"],
        "xy-position": jnp.array((0.5, -0.5)),
    }
    vrender_relit_up = jax.jit(
        jax.vmap(
            jax.vmap(partial(render, camera=camera, flash_light=relit_up_flash_light))
        )
    )
    images_transition_relit_up = tonemapping(
        vrender_relit_up(clipped_BRDF_maps_transition, normal_map_transition)
    )
    vrender_relit_left = jax.jit(
        jax.vmap(
            jax.vmap(partial(render, camera=camera, flash_light=relit_left_flash_light))
        )
    )
    images_transition_relit_left = tonemapping(
        vrender_relit_left(clipped_BRDF_maps_transition, normal_map_transition)
    )
    vrender_relit_rightdown = jax.jit(
        jax.vmap(
            jax.vmap(
                partial(render, camera=camera, flash_light=relit_rightdown_flash_light)
            )
        )
    )
    images_transition_relit_rightdown = tonemapping(
        vrender_relit_rightdown(clipped_BRDF_maps_transition, normal_map_transition)
    )

    stitches = jax.vmap(jax.vmap(stitch_maps, in_axes=(None, 0, 0, 0)))(
        exemplars,
        images_transition,
        clipped_BRDF_maps_transition,
        normal_map_transition,
    )
    save_images(opath.join(ws_path, "renderings"), np.array(images_transition))
    save_images(
        opath.join(ws_path, "renderings_relit_up"), np.array(images_transition_relit_up)
    )
    save_images(
        opath.join(ws_path, "renderings_relit_left"),
        np.array(images_transition_relit_left),
    )
    save_images(
        opath.join(ws_path, "renderings_relit_rightdown"),
        np.array(images_transition_relit_rightdown),
    )
    save_images(
        opath.join(ws_path, "renderings_relit_rotating"),
        np.array(images_transition_relit_rotating),
    )
    save_images(
        opath.join(ws_path, "diffuse"),
        np.array(clipped_BRDF_maps_transition[..., :3, :, :]),
    )
    save_images(
        opath.join(ws_path, "specular"),
        np.array(clipped_BRDF_maps_transition[..., 3:6, :, :]),
    )
    if n_BRDF_channels == 7:
        assert args.renderer == "diffuse_iso_cook_torrance"
        save_images(
            opath.join(ws_path, "roughness"),
            np.array(clipped_BRDF_maps_transition[..., 6:7, :, :]),
        )
    if n_BRDF_channels == 8:
        assert args.renderer == "diffuse_cook_torrance"
        save_images(
            opath.join(ws_path, "roughness_u"),
            np.array(clipped_BRDF_maps_transition[..., 6:7, :, :]),
        )
        save_images(
            opath.join(ws_path, "roughness_v"),
            np.array(clipped_BRDF_maps_transition[..., 7:8, :, :]),
        )

    save_images(
        opath.join(ws_path, "normal"),
        np.array(normal_map_transition) * 0.5 + 0.5,
    )

    def scale_to_01(x):
        return (x - x.min()) / (x.max() - x.min())

    save_images(
        opath.join(ws_path, "height"),
        np.array(jax.vmap(scale_to_01)(height_map_transition)),
    )
    save_images(opath.join(ws_path, "stitches"), np.array(stitches))
