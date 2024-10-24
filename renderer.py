import jax.numpy as jnp
import jax
from rendering_utils import (
    EPSILON,
    channelwise_normalize,
    normalize,
    safe_dot,
    localize_wiwo,
)


# The anisotropic GGX distribution (if alpha_u == alpha_v, it becomes isotropic)
# also known as Trowbridge-Reitz
# implementation is borrowed from Mitsuba 3
def distribution_ggx(H, alpha_u, alpha_v):
    alpha_uv = alpha_u * alpha_v
    sin_theta_cos_phi = H[0]
    sin_theta_sin_phi = H[1]
    cos_theta = jnp.maximum(H[2], 0.0)

    denom = (
        jnp.pi
        * alpha_uv
        * (
            (sin_theta_cos_phi / (alpha_u)) ** 2
            + (sin_theta_sin_phi / (alpha_v)) ** 2
            + cos_theta**2
        )
        ** 2
    )

    return 1.0 / (denom)


def iso_distribution_ggx(H, alpha):
    alpha2 = alpha**2
    cos_theta_2 = H[2] ** 2

    num = alpha2
    denom = jnp.pi * (cos_theta_2 * (alpha2 - 1) + 1) ** 2 + EPSILON

    return num / denom


def fresnel_schlick(NdotL, F0):
    return F0 + (1.0 - F0) * (1.0 - NdotL) ** 5


# The Smith shadowing-masking function for a single direction, for GGX distribution
def smith_g1_ggx(v, alpha_u, alpha_v):
    xy_alpha_2 = (alpha_u * v[0]) ** 2 + (alpha_v * v[1]) ** 2
    tan_theta_alpha_2 = xy_alpha_2 / (jnp.maximum(v[2], 0.0) ** 2 + EPSILON)

    return 2.0 / (1.0 + jnp.sqrt(1.0 + tan_theta_alpha_2))


# The Smith separable shadowing-masking approximation
def geometry_smith(V, L, alpha_u, alpha_v):
    ggx1 = smith_g1_ggx(V, alpha_u, alpha_v)
    ggx2 = smith_g1_ggx(L, alpha_u, alpha_v)

    return ggx1 * ggx2


def iso_smith_g1_ggx(v, alpha):
    alpha_2 = alpha**2
    cos_theta = v[2]
    cos_theta_2 = cos_theta**2
    sin_theta_2 = 1.0 - cos_theta_2
    tan_theta_2 = sin_theta_2 / (cos_theta_2 + EPSILON)

    return 2.0 / (1.0 + jnp.sqrt(1.0 + alpha_2 * tan_theta_2))


def iso_geometry_smith(V, L, alpha):
    ggx1 = iso_smith_g1_ggx(V, alpha)
    ggx2 = iso_smith_g1_ggx(L, alpha)

    return ggx1 * ggx2


# Anisotropic Cook-Torrance BRDF
# with GGX distribution, Smith shadowing-masking, and Schlick's approximation for Fresnel
def cook_torrance(wi, wo, albedo, alpha_u, alpha_v):
    H, L, V = normalize(wi + wo), wi, wo
    NdotL, NdotV = jnp.maximum(L[2], 0.0), jnp.maximum(V[2], 0.0)
    HdotL = HdotV = safe_dot(H, L)

    F0 = albedo
    D = distribution_ggx(H, alpha_u, alpha_v)
    G = geometry_smith(V, L, alpha_u, alpha_v)
    F = fresnel_schlick(HdotL, F0)

    reflectance = D * G * F / (4 * NdotV + EPSILON)
    return reflectance


# Isotropic Cook-Torrance BRDF
# with GGX distribution, Smith shadowing-masking, and Schlick's approximation for Fresnel
def iso_cook_torrance(wi, wo, albedo, alpha):
    H, L, V = normalize(wi + wo), wi, wo
    NdotL, NdotV = jnp.maximum(L[2], 0.0), jnp.maximum(V[2], 0.0)
    HdotL = HdotV = safe_dot(H, L)

    F0 = albedo
    D = iso_distribution_ggx(H, alpha)
    G = iso_geometry_smith(V, L, alpha)
    F = fresnel_schlick(HdotL, F0)

    reflectance = D * G * F / (4 * NdotV + EPSILON)
    return reflectance


# Diffuse BRDF, also known as Lambertian BRDF
def lambertian(wi, wo, albedo):
    NdotL = jnp.maximum(wi[2], 0.0)
    return albedo / jnp.pi * NdotL


# Anisotropic Cook-Torrance BRDF
# Complemented with a diffuse BRDF via the metallic parameter (an empirical way to compensate for the inter-reflection omitted in the model)
def compl_cook_torrance(wi, wo, albedo, metallic, alpha_u, alpha_v):
    HdotL = safe_dot(normalize(wi + wo), wi)

    F0 = jnp.array([0.04, 0.04, 0.04]) * (1.0 - metallic) + albedo * metallic

    # empirically avoid energy leakage, and blended by the metallic parameter
    F = fresnel_schlick(HdotL, F0)
    kd = (1.0 - F) * (1.0 - metallic)

    reflectance = kd * lambertian(wi, wo, albedo) + cook_torrance(
        wi, wo, F0, alpha_u, alpha_v
    )
    return reflectance


# Isotropic Cook-Torrance BRDF
# Complemented with a diffuse BRDF via the metallic parameter (an empirical way to compensate for the inter-reflection omitted in the model)
def compl_iso_cook_torrance(wi, wo, albedo, metallic, alpha):
    HdotL = safe_dot(normalize(wi + wo), wi)

    F0 = jnp.array([0.04, 0.04, 0.04]) * (1.0 - metallic) + albedo * metallic

    # empirically avoid energy leakage, and blended by the metallic parameter
    F = fresnel_schlick(HdotL, F0)
    kd = (1.0 - F) * (1.0 - metallic)

    reflectance = kd * lambertian(wi, wo, albedo) + iso_cook_torrance(wi, wo, F0, alpha)
    return reflectance


def diffuse_iso_cook_torrance(wi, wo, diffuse, specular, alpha):
    diffuse_term = lambertian(wi, wo, diffuse)

    specular_term = iso_cook_torrance(wi, wo, specular, alpha)

    return diffuse_term + specular_term


def diffuse_iso_cook_torrance_renderer_pp(wi, wo, BRDF_params):
    diffuse = BRDF_params[:3]
    specular = BRDF_params[3:6]
    roughness = BRDF_params[6:7]

    return diffuse_iso_cook_torrance(wi, wo, diffuse, specular, roughness)


def diffuse_cook_torrance(wi, wo, diffuse, specular, alpha_u, alpha_v):
    diffuse_term = lambertian(wi, wo, diffuse)

    specular_term = cook_torrance(wi, wo, specular, alpha_u, alpha_v)

    return diffuse_term + specular_term


def diffuse_cook_torrance_renderer_pp(wi, wo, BRDF_params):
    diffuse = BRDF_params[:3]
    specular = BRDF_params[3:6]
    roughness_u = BRDF_params[6:7]
    roughness_v = BRDF_params[7:8]

    return diffuse_cook_torrance(wi, wo, diffuse, specular, roughness_u, roughness_v)


def compl_iso_cook_torrance_renderer_pp(wi, wo, BRDF_params):
    albedo = BRDF_params[:3]
    metallic = BRDF_params[3:4]
    roughness = BRDF_params[4:5]

    return compl_iso_cook_torrance(wi, wo, albedo, metallic, roughness)


def compl_cook_torrance_renderer_pp(wi, wo, BRDF_params):
    albedo = BRDF_params[:3]
    metallic = BRDF_params[3:4]
    roughness_u = BRDF_params[4:5]
    roughness_v = BRDF_params[5:6]

    return compl_cook_torrance(wi, wo, albedo, metallic, roughness_u, roughness_v)


def cook_torrance_renderer_pp(wi, wo, BRDF_params):
    albedo = BRDF_params[:3]
    roughness_u = BRDF_params[3:4]
    roughness_v = BRDF_params[4:5]

    return cook_torrance(wi, wo, albedo, roughness_u, roughness_v)


def iso_cook_torrance_renderer_pp(wi, wo, BRDF_params):
    albedo = BRDF_params[:3]
    roughness = BRDF_params[3:4]

    return iso_cook_torrance(wi, wo, albedo, roughness)


def reinhard(img):
    return img / (1.0 + img)


def gamma_correct(img, gamma=2.2):
    return jnp.power(img, 1.0 / gamma)


def light_decay(distance):
    return 1.0 / (distance**2)


def create_meshgrid(height, width):
    x = jnp.linspace(-1, 1, width)
    y = jnp.linspace(-1, 1, height)
    x, y = jnp.meshgrid(x, y)
    aspect_ratio = width / height
    y /= aspect_ratio
    # NOTE: We need to flip the y-axis because the image origin is at the top-left (as we use right-handed coordinates)
    y *= -1

    return x, y


def render(
    BRDF_maps,
    normal_map,
    camera,
    flash_light,
    renderer_pp,
    region=None,
    normal_loss=False,
):
    """Render an image from a set of BRDF maps (albedo, metallic, roughness, height/normal)

    Args:
        renderer_pp: the per-pixel and direct lighting renderer.
            it takes wi, wo, C dimensional BRDF parameters, and normal map as inputs, and returns the reflectance value at this pixel.
            Callable of (wi, wo, BRDF_params) -> cosine-weighted reflectance
            ...
        BRDF_maps: C channel maps of BRDF parameters.
            1. For Cook-Torrance BRDF: (albedo(3), metallic(1), roughness(1))
            2. For Neural BRDF: (latents(32), )
        normal_map: the normal map of (3, H, W)
        camera: Dictionary of camera parameters.
            fov: Field of view in degrees.
            distance: Camera position at z axis.
        flash_light: Dictionary of light parameters.
            intensity: Lighting intensity.
            xy-position: Position of the light in xy plane.
            NOTE: The light is assumed to be in the same z plane as the camera.
        region: Dictionary of the region of the image to render.
            if None, render the entire image of (h, w).
            otherwise, input maps are actually a small region of the entire image.
                H, W: Height and width of the real image.
                crop: callable (key, image -> cropped_image) to crop the position grid,
                    such as cropping grid in (3, H, W) to (3, h, w).
                cropkey: the key to crop the position grid.

    Returns:
        (3, h, w) array of rendered images in LINEAR space.
    """

    fov = camera["fov"]
    distance = camera["distance"]
    light_intensity = jnp.exp(flash_light["intensity"])
    light_xy_position = flash_light["xy-position"]

    _, h, w = BRDF_maps.shape
    # we assume that the image is centered at z = 0
    # so the the image spans from -width to width in x axis
    width = distance * jnp.tan(jnp.deg2rad(fov / 2.0))

    if region is not None:
        H, W = region["H"], region["W"]
        crop, cropkey = region["crop"], region["cropkey"]
        x, y = create_meshgrid(H, W)
        x, y = x * width, y * width
        img_pos = jnp.stack([x, y, jnp.zeros_like(x)], axis=0)
        img_pos = crop(cropkey, img_pos)
    else:
        x, y = create_meshgrid(h, w)
        x, y = x * width, y * width
        img_pos = jnp.stack([x, y, jnp.zeros_like(x)], axis=0)

    light_pos = jnp.concatenate(
        [jnp.array(light_xy_position), jnp.array([distance])]
    ).reshape(3, 1, 1)
    view_pos = jnp.array([0.0, 0.0, distance]).reshape(3, 1, 1)

    wi = channelwise_normalize(light_pos - img_pos)
    wo = channelwise_normalize(view_pos - img_pos)

    local_wi, local_wo = jax.vmap(
        jax.vmap(localize_wiwo, in_axes=(1), out_axes=(1)), in_axes=(2), out_axes=(2)
    )(wi, wo, normal_map)

    shaded = jax.vmap(
        jax.vmap(renderer_pp, in_axes=(1), out_axes=(1)), in_axes=(2), out_axes=(2)
    )(local_wi, local_wo, BRDF_maps)

    light_distances = jnp.linalg.norm(light_pos - img_pos, axis=0, keepdims=True)

    rendered = shaded * light_intensity * light_decay(light_distances)

    invalid_indices = jnp.logical_or(local_wi[2] < 0.0, local_wo[2] < 0.0)
    rendered = jnp.where(invalid_indices, 0.0, rendered)

    if normal_loss:
        return rendered, (
            jnp.sum(jax.nn.relu(EPSILON - local_wi[2]) ** 2)
            + jnp.sum(jax.nn.relu(EPSILON - local_wo[2]) ** 2)
        )
    else:
        return rendered


def tonemapping(img):
    """
    HDR to LDR (0 ~ 1)
    """
    return gamma_correct(jnp.clip(img, EPSILON, 1.0))
