import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Any, Callable
from equinox import field
import torch

pad2d_constant = jax.vmap(partial(jnp.pad, mode="constant", pad_width=(1, 1)))
pad2d_reflect = jax.vmap(partial(jnp.pad, mode="reflect", pad_width=(1, 1)))
pad2d_circular = jax.vmap(partial(jnp.pad, mode="wrap", pad_width=(1, 1)))


class VGG19(eqx.Module):
    block1: list
    block2: list
    block3: list
    block4: list
    block5: list
    padding: Callable = field(static=True)
    activation: Callable = field(static=True)
    downsampling: Callable = field(static=True)

    def __init__(self, key):
        keys = jax.random.split(key, 16)
        self.block1 = [
            eqx.nn.Conv2d(3, 64, 3, key=keys[0]),
            eqx.nn.Conv2d(64, 64, 3, key=keys[1]),
        ]

        self.block2 = [
            eqx.nn.Conv2d(64, 128, 3, key=keys[2]),
            eqx.nn.Conv2d(128, 128, 3, key=keys[3]),
        ]

        self.block3 = [
            eqx.nn.Conv2d(128, 256, 3, key=keys[4]),
            eqx.nn.Conv2d(256, 256, 3, key=keys[5]),
            eqx.nn.Conv2d(256, 256, 3, key=keys[6]),
            eqx.nn.Conv2d(256, 256, 3, key=keys[7]),
        ]

        self.block4 = [
            eqx.nn.Conv2d(256, 512, 3, key=keys[8]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[9]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[10]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[11]),
        ]

        self.block5 = [
            eqx.nn.Conv2d(512, 512, 3, key=keys[12]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[13]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[14]),
            eqx.nn.Conv2d(512, 512, 3, key=keys[15]),
        ]

        self.padding = pad2d_reflect
        self.activation = jax.nn.relu
        self.downsampling = eqx.nn.AvgPool2d((2, 2), stride=2)

    def __call__(self, x):
        features = []
        features.append(x)

        mean = jnp.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = jnp.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        x = (x - mean) / std

        # block1
        for conv in self.block1:
            x = self.activation(conv(self.padding(x)))
        features.append(x)
        x = self.downsampling(x)

        # block2
        for conv in self.block2:
            x = self.activation(conv(self.padding(x)))
        features.append(x)
        x = self.downsampling(x)

        # block3
        for conv in self.block3:
            x = self.activation(conv(self.padding(x)))
        features.append(x)
        x = self.downsampling(x)

        # block4
        for conv in self.block4:
            x = self.activation(conv(self.padding(x)))
        features.append(x)
        x = self.downsampling(x)

        # block5
        for conv in self.block5:
            x = self.activation(conv(self.padding(x)))
        features.append(x)
        x = self.downsampling(x)

        return features


def load_pretrained_VGG19_from_pth(pth_path, dtype=jnp.float32):
    # get treedef from a dummy VGG
    VGG_dummy = VGG19(jax.random.key(0))
    _, treedef = jax.tree_util.tree_flatten(VGG_dummy)

    # formulate pretrained weights as corresponding leaves
    vgg_pth = torch.hub.load_state_dict_from_url(pth_path, map_location="cpu")
    vgg_jnp = jax.tree.map(lambda x: jnp.array(x.numpy(), dtype=dtype), vgg_pth)
    leaves, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map_with_path(
            lambda kp, x: x[..., None, None] if "bias" in str(kp) else x, vgg_jnp
        )
    )

    leaves = leaves[:32]

    # unflatten back to model
    return jax.tree_util.tree_unflatten(treedef, leaves)


def create_overflow_loss(min_v, max_v):
    assert min_v <= max_v, f"min_v: {min_v} > max_v: {max_v}"

    return lambda x: jnp.mean(jnp.abs(x - x.clip(min_v, max_v)))


def gram_loss(features, exemplar, sample, key=None):
    features_exemplar = features(exemplar)
    gmatrices_exemplar = jax.tree.map(gram_matrix, features_exemplar)

    features_sample = features(sample)
    gmatrices_sample = jax.tree.map(gram_matrix, features_sample)

    mse = lambda x, y: jnp.mean((x - y) ** 2)
    loss = sum(jax.tree.map(mse, gmatrices_exemplar, gmatrices_sample))
    return loss


def slice_loss(features, exemplar, sample, key, _weights=[1, 1, 1, 1, 1, 1]):
    weights = [w / sum(_weights) * len(_weights) for w in _weights]
    features_exemplar = features(exemplar)
    features_sample = features(sample)

    keys = list(jax.random.split(key, num=len(features_sample)))
    return sum(
        jax.tree.map(
            lambda w, l: w * l,
            weights,
            jax.tree.map(
                sliced_wasserstein_loss, features_exemplar, features_sample, keys
            ),
        )
    )


def gram_matrix(f):
    f = f.reshape(f.shape[0], -1)
    gram_matrix = f @ f.transpose()

    gram_matrix = gram_matrix / f.shape[-1]
    return gram_matrix


def sliced_wasserstein_loss(fe, fs, key):
    fe = fe.reshape(fe.shape[0], -1)
    fs = fs.reshape(fs.shape[0], -1)

    # get c random directions
    c, n = fs.shape
    Vs = jax.random.normal(key, (c, c))
    Vs = Vs / jnp.sqrt(jnp.sum(Vs**2, axis=1, keepdims=True))

    # project
    pfe = jnp.einsum("cn,mc->mn", fe, Vs)
    pfs = jnp.einsum("cn,mc->mn", fs, Vs)

    # sort
    spfe = jnp.sort(pfe, axis=1)
    spfs = jnp.sort(pfs, axis=1)
    ## apply interpolation like an image to match the dimension
    spfe = jax.image.resize(spfe, spfs.shape, method="nearest")

    # MSE
    loss = jnp.mean((spfe - spfs) ** 2)

    return loss
