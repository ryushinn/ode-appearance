import argparse
from datetime import datetime
import numpy as np
from PIL import Image
from copy import deepcopy

import os.path as opath


import jax, jax.numpy as jnp, diffrax
import nets_jax
from metrics_jax import pad2d_constant

pad2d_constant_batched = jax.jit(jax.vmap(pad2d_constant))
from utils_jax import seed_all, preprocess_exemplar, l2i, seed_batch, load_gif
import equinox as eqx


import os
import time
import subprocess

parser = argparse.ArgumentParser("ODE texture")

# sampling solver
parser.add_argument(
    "--solver", type=str, default="heun", choices=["euler", "tsit5", "heun"]
)
parser.add_argument("--tol", type=float, default=1e-2)
parser.add_argument("--step_size", type=float, default=1e-2)

parser.add_argument("--n_aug_channels", type=int, default=9)
parser.add_argument("--dim", type=int, default=32)
parser.add_argument("--dim_mults", type=str, default="1,2,4")
parser.add_argument("--n_attn_heads", type=int, default=4)
parser.add_argument("--attn_head_dim", type=int, default=8)

parser.add_argument("--size", type=int, default=128)
parser.add_argument("--sample_size", type=int, default=256)

# sampling config
parser.add_argument("--exemplars_path", type=str)
parser.add_argument("--checkpoint_path", type=str)
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
    if not opath.exists(ws_path):
        os.makedirs(ws_path, exist_ok=True)

    # load GIF
    exemplars, time_between = load_gif(args.exemplars_path)

    assert (
        time_between == 0.05
    ), f"time between frames is {time_between}, but we expect 0.05"

    n_frames = len(exemplars)
    exemplars = [preprocess_exemplar(e, (args.size, args.size)) for e in exemplars]

    ## to numpy
    exemplars_np = np.stack(
        [np.array(e, dtype=dtype).transpose(2, 0, 1) / 255.0 for e in exemplars]
    )

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

    reg_fns = ()
    model = nets_jax.NeuralODE(odefunc, reg_fns)

    checkpoint_path = args.checkpoint_path
    model = eqx.tree_deserialise_leaves(opath.join(checkpoint_path, "model.eqx"), model)

    to_rgb = lambda x: l2i(x[..., :3, :, :])

    @eqx.filter_jit
    def inference(key, model, n, size, diffeqsolve_args):
        xs = seed_batch(key, n, n_channels, size)
        solutions = model(
            -syn_t, duration, xs, get_reg=False, key=None, **diffeqsolve_args
        )
        return solutions

    ## Duration of the input
    duration = n_frames * time_between

    ## synthesis duration
    syn_t = duration * 0.2

    ts_synthesis = jnp.logspace(0.0, jnp.log10(1.0 + syn_t), 50) - 1.0 - syn_t
    ts_transition = jnp.linspace(0.0, duration, n_frames)
    ts_samples = jnp.concatenate([ts_synthesis, ts_transition])

    sampling_args = deepcopy(diffeqsolve_args)
    sampling_args.update({"saveat": diffrax.SaveAt(ts=ts_samples)})

    def make_directory(dir):
        dir_path = opath.join(ws_path, dir)
        if not opath.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def save_images(images_np, dir, duration=100):
        frames = []
        save_path = make_directory(dir)
        for i, image in enumerate(images_np):
            image = Image.fromarray(
                (np.clip(image, a_min=0.0, a_max=1.0) * 255)
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
            image.save(opath.join(save_path, f"frame_{i:04d}.png"))
            frames.append(image)
        frames[0].save(
            opath.join(save_path, f"{dir}.gif"),
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            f"{opath.join(save_path, dir)}.gif",
            "-c:v",
            "rawvideo",
            "-r",
            f"{1000 / duration}",
            f"{opath.join(save_path, dir)}.avi",
        ]
        subprocess.run(
            ffmpeg_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    _ = inference(key, model, 1, args.sample_size, sampling_args)
    print("----------------------------------------------------")
    print(
        f"Start sampling with {args.solver} solver [atol={args.tol}, rtol={args.tol}, stepsize={args.step_size}]..."
    )
    start_time = time.time()
    solutions = inference(key, model, 1, args.sample_size, sampling_args)
    _ = solutions.ys["x"].block_until_ready()

    cur_time = time.time()
    print(f"Sampling finished in {cur_time - start_time:.4f} seconds.")
    start_time = cur_time

    images = to_rgb(solutions.ys["x"])
    images_synthesis = np.array(
        images[: len(ts_synthesis), ...].transpose(1, 0, 2, 3, 4)
    )[0]
    images_transition = np.array(
        images[len(ts_synthesis) :, ...].transpose(1, 0, 2, 3, 4)
    )[0]
    padded_exemplars = np.zeros(images_transition.shape)
    padded_exemplars[..., : args.size, : args.size] = exemplars_np

    save_images(images_synthesis, "synthesis")
    save_images(images_transition, "transition", time_between * 1000)
    save_images(exemplars_np, "exemplars", time_between * 1000)
    save_images(
        np.concatenate([images_transition, padded_exemplars], axis=3),
        "comparisons",
        time_between * 1000,
    )

    cur_time = time.time()
    print(f"Saving images finished in {cur_time - start_time:.4f} seconds.")
    print("----------------------------------------------------")
