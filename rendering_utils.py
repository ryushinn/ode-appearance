from PIL import Image
import jax.numpy as jnp
import jax
from jax import lax
from os import path as opath


EPSILON = 1e-6


def safe_dot(a, b):
    """
    dot product and clip the result within [0, 1]
    """
    return jnp.clip(jnp.dot(a, b), 0.0, 1.0)


@jax.jit
def normalize(v):
    """Normalize a vector"""
    return v / jnp.linalg.norm(v)


"""channelwise normalize a map of vectors (c, H, W)"""
channelwise_normalize = jax.jit(
    jax.vmap(jax.vmap(normalize, in_axes=(1), out_axes=(1)), in_axes=(2), out_axes=(2))
)


def localize(vec, normal):
    """
    Project a vector from world space to the local tangent space perturbed by a surface normal in world space
    The world space is right-handed
    """
    tangent = jnp.array([1.0, 0.0, 0.0])  # tangent in world space
    perturbed_tangent = normalize(tangent - jnp.dot(tangent, normal) * normal)
    bitangent = normalize(jnp.cross(normal, perturbed_tangent))

    matrix = jnp.stack([perturbed_tangent, bitangent, normal], axis=0)

    return matrix @ vec


def localize_wiwo(wi, wo, normal):
    local_wi, local_wo = jax.vmap(localize, in_axes=(0, None))(
        jnp.stack([wi, wo], axis=0), normal
    )
    return local_wi, local_wo


def height_to_normal(height, scale=1.0):
    """Convert a height map to a normal map

    Args:
        height: array of height values in (1, h, w)
        scale: scale of the height map, default is 1.0, equivalent to the reciprocal of the distance between two pixels

    Returns:
        (3, h, w) array of normal vectors.
    """
    _, h, w = height.shape

    height = scale * height.reshape(h, w)
    height = jnp.pad(height, pad_width=((1, 1), (1, 1)), mode="edge")

    h01 = height[:-2, 1:-1]
    h21 = height[2:, 1:-1]
    h10 = height[1:-1, :-2]
    h12 = height[1:-1, 2:]

    # gradient in x and y directions, note that here x and y are in world space
    gx = (h12 - h10) / 2.0
    gy = (h01 - h21) / 2.0

    # So normal is already in world-space
    normal = jnp.stack([-gx, -gy, jnp.ones((h, w))], axis=0)
    normal = channelwise_normalize(normal)

    return normal


def jax_unstack(x, axis=0):
    return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


def clip_maps(maps):
    return jnp.clip(maps, EPSILON, 1.0)


def random_crop(key, image, crop_h, crop_w):
    key1, key2 = jax.random.split(key, 2)
    c, h, w = image.shape

    # Generate two random numbers for the top-left corner of the crop
    top = jax.random.randint(key1, (), 0, h - crop_h + 1)
    left = jax.random.randint(key2, (), 0, w - crop_w + 1)

    return lax.dynamic_slice(image, (0, top, left), (c, crop_h, crop_w))


def random_take(key, image, new_h, new_w):
    c, h, w = image.shape
    n = new_h * new_w
    indices = jax.random.choice(key, h * w, (n,), replace=False)

    return image.reshape(c, -1)[:, indices].reshape(c, new_h, new_w)


l2i = lambda x: x * 0.5 + 0.5
i2l = lambda x: (x - 0.5) * 2.0


def to_maps(x, n_BRDF_channels, n_normal_channels):
    """
    x: (..., N, H, W) array, N >= n_BRDF_channels + n_height_channels
    n_BRDF_channels: number of channels for BRDF maps
    n_normal_channels: number of channels for height map

    Returns:
        BRDF_maps (..., n_BRDF_channels, H, W) array
        normal_map (..., n_normal_channels, H, W) array
    """
    return (
        l2i(x[..., :n_BRDF_channels, :, :]),
        x[..., n_BRDF_channels : n_BRDF_channels + n_normal_channels, :, :],
    )


def stitch_maps(target, rendering, BRDF_maps, normal_map):
    _, h, w = target.shape
    normal = normal_map * 0.5 + 0.5

    if BRDF_maps.shape[0] == 5:
        albedo = BRDF_maps[:3]
        metallic = BRDF_maps[3:4]
        roughness = BRDF_maps[4:5]

        stitched = jnp.zeros((3, h * 2, w * 3))
        stitched = stitched.at[:, :h, :w].set(rendering)  # (0, 0)
        stitched = stitched.at[:, h:, :w].set(target)  # (1, 0)
        stitched = stitched.at[:, :h, w : 2 * w].set(albedo)  # (0, 1)
        stitched = stitched.at[:, h:, w : 2 * w].set(metallic)  # (1, 1)
        stitched = stitched.at[:, :h, 2 * w :].set(roughness)  # (0, 2)
        stitched = stitched.at[:, h:, 2 * w :].set(normal)  # (1, 2)
    elif BRDF_maps.shape[0] == 6:
        albedo = BRDF_maps[:3]
        metallic = BRDF_maps[3:4]
        roughness_u = BRDF_maps[4:5]
        roughness_v = BRDF_maps[5:6]

        stitched = jnp.zeros((3, h * 2, w * 4))
        stitched = stitched.at[:, :h, :w].set(rendering)  # (0, 0)
        stitched = stitched.at[:, h:, :w].set(target)  # (1, 0)
        stitched = stitched.at[:, :h, w : 2 * w].set(albedo)  # (0, 1)
        stitched = stitched.at[:, h:, w : 2 * w].set(metallic)  # (1, 1)
        stitched = stitched.at[:, :h, 2 * w : 3 * w].set(roughness_u)  # (0, 2)
        stitched = stitched.at[:, h:, 2 * w : 3 * w].set(roughness_v)  # (1, 2)
        stitched = stitched.at[:, :h, 3 * w :].set(normal)  # (0, 3)
    elif BRDF_maps.shape[0] == 7:
        diffuse = BRDF_maps[:3]
        specular = BRDF_maps[3:6]
        roughness = BRDF_maps[6:7]

        stitched = jnp.zeros((3, h * 2, w * 3))
        stitched = stitched.at[:, :h, :w].set(rendering)  # (0, 0)
        stitched = stitched.at[:, h:, :w].set(target)  # (1, 0)
        stitched = stitched.at[:, :h, w : 2 * w].set(diffuse)  # (0, 1)
        stitched = stitched.at[:, h:, w : 2 * w].set(specular)  # (1, 1)
        stitched = stitched.at[:, :h, 2 * w :].set(roughness)  # (0, 2)
        stitched = stitched.at[:, h:, 2 * w :].set(normal)  # (1, 2)

    elif BRDF_maps.shape[0] == 8:
        diffuse = BRDF_maps[:3]
        specular = BRDF_maps[3:6]
        roughness_u = BRDF_maps[6:7]
        roughness_v = BRDF_maps[7:8]

        stitched = jnp.zeros((3, h * 2, w * 4))
        stitched = stitched.at[:, :h, :w].set(rendering)  # (0, 0)
        stitched = stitched.at[:, h:, :w].set(target)  # (1, 0)
        stitched = stitched.at[:, :h, w : 2 * w].set(diffuse)  # (0, 1)
        stitched = stitched.at[:, h:, w : 2 * w].set(specular)  # (1, 1)
        stitched = stitched.at[:, :h, 2 * w : 3 * w].set(roughness_u)  # (0, 2)
        stitched = stitched.at[:, h:, 2 * w : 3 * w].set(roughness_v)  # (1, 2)
        stitched = stitched.at[:, :h, 3 * w :].set(normal)  # (0, 3)

    else:
        raise NotImplementedError("Unsupported BRDF maps")

    return stitched


def select_renderer(renderer):
    """
    Select
    """
    from renderer import (
        compl_iso_cook_torrance_renderer_pp,
        compl_cook_torrance_renderer_pp,
        cook_torrance_renderer_pp,
        iso_cook_torrance_renderer_pp,
        diffuse_iso_cook_torrance_renderer_pp,
        diffuse_cook_torrance_renderer_pp,
    )

    if renderer == "diffuse_iso_cook_torrance":
        print("Use Diffuse lobe + isotropic Cook-Torrance renderer")
        renderer_pp = diffuse_iso_cook_torrance_renderer_pp
        n_BRDF_channels = 7
    elif renderer == "diffuse_cook_torrance":
        print("Use Diffuse lobe + anisotropic Cook-Torrance renderer")
        renderer_pp = diffuse_cook_torrance_renderer_pp
        n_BRDF_channels = 8
    elif renderer == "compl_iso_cook_torrance":
        print("Use isotropic Cook-Torrance renderer + Diffuse complemented by metallic")
        renderer_pp = compl_iso_cook_torrance_renderer_pp
        n_BRDF_channels = 5
    elif renderer == "compl_cook_torrance":
        print("Use aniso. Cook-Torrance renderer + Diffuse complemented by metallic")
        renderer_pp = compl_cook_torrance_renderer_pp
        n_BRDF_channels = 6
    elif renderer == "cook_torrance":
        print("Use anisotropic Cook-Torrance renderer")
        renderer_pp = cook_torrance_renderer_pp
        n_BRDF_channels = 5
    elif renderer == "iso_cook_torrance":
        print("Use isotropic Cook-Torrance renderer")
        renderer_pp = iso_cook_torrance_renderer_pp
        n_BRDF_channels = 4
    else:
        raise NotImplementedError(f"Renderer {renderer} not implemented")

    return (
        renderer_pp,
        clip_maps,
        n_BRDF_channels,
    )
