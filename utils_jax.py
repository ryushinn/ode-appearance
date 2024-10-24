import random
import numpy as np
import os
from PIL import Image
import json
import jax
import jax.numpy as jnp
import equinox as eqx


def seed_all(seed):
    """
    provide the seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    return jax.random.PRNGKey(seed)


def preprocess_exemplar(image: Image, new_size: tuple[int, int] = (128, 128)):
    new_dimension = min(image.width, image.height)

    # Calculate the coordinates for the central crop
    left = (image.width - new_dimension) / 2
    top = (image.height - new_dimension) / 2
    right = (image.width + new_dimension) / 2
    bottom = (image.height + new_dimension) / 2

    # Crop the image to the central square
    cropped_image = image.crop((left, top, right, bottom))

    resized_image = cropped_image.resize(new_size)

    return resized_image


i2l = lambda x: 2 * x - 1.0
l2i = lambda x: x * 0.5 + 0.5


def seed_batch(key, n, n_channels, size):
    return jax.vmap(seed, in_axes=(0, None, None))(
        jax.random.split(key, n), n_channels, size
    )


def seed(key, n_channels, size):
    return jax.random.normal(key, (n_channels, size, size))


def seed_uniform(key, n, n_channels, size, minval=-0.5, maxval=0.5):
    if type(size) == int:
        return jax.random.uniform(
            key, (n, n_channels, size, size), minval=minval, maxval=maxval
        )
    elif type(size) == tuple:
        assert len(size) == 2
        return jax.random.uniform(
            key, (n, n_channels, *size), minval=minval, maxval=maxval
        )
    else:
        raise NotImplementedError("size must be int or tuple of length 2")


def load_gif(path):
    gif = Image.open(path)

    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = gif.convert("RGB")
        frames.append(frame)

    time_between = gif.info.get("duration", 100) / 1000

    return frames, time_between


def read_images(path, n_images=100, resize=(128, 128)):
    images = []

    filenames = os.listdir(path)
    # sort filenames to make sure the order is consistent
    filenames = [
        f
        for f in filenames
        if f.endswith("JPG")
        or f.endswith("jpg")
        or f.endswith("png")
        or f.endswith("PNG")
    ]
    filenames.sort()
    print(f"Found {len(filenames)} images in {path}, sample {n_images} images.")
    # sample n_image iamges (definitely include the first and last image)
    interval = (len(filenames) - 1) / (n_images - 1)
    selected_indices = [round(i * interval) for i in range(1, n_images - 1)]
    selected_indices = [0] + selected_indices + [len(filenames) - 1]
    filenames = [filenames[i] for i in selected_indices]
    image_filenames = [f for f in filenames if os.path.isfile(os.path.join(path, f))]

    for image_filename in image_filenames:
        image = Image.open(os.path.join(path, image_filename)).convert("RGB")
        image = preprocess_exemplar(image, new_size=resize)
        images.append(jnp.array(image))

    return jnp.array(images).transpose((0, 3, 1, 2)) / 255.0


def size_of_model(model):
    model_params, _ = eqx.partition(model, eqx.is_inexact_array)
    n_parameters = 0
    leaves = jax.tree_util.tree_leaves(model_params)
    for leaf in leaves:
        n_parameters += leaf.size
    print(f"number of parameters: {n_parameters} ({n_parameters / 1e6:.2f}M)")
