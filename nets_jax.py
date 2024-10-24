import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from typing import Any, Callable, List, Optional
from inspect import isfunction
from equinox import field
from metrics_jax import pad2d_circular, pad2d_reflect
import diffrax
from reg_lib_jax import RegularizedODEfunc
from einops import rearrange
import math


def zero_init(model):
    leaves, treedef = jax.tree_util.tree_flatten(model, eqx.is_array)
    zero_leaves = jax.tree.map(lambda x: jnp.zeros(x.shape), leaves)
    return jax.tree_util.tree_unflatten(treedef, zero_leaves)


class DefaultConv2d(eqx.nn.Conv2d):
    """
    A default 2D convolution module with 3x3 kernel, same padding, and circular padding mode
    This is implemented as a subclass of `eqx.nn.Conv2d`
    because equinox v0.11.3 doesn't support `padding_mode` argument
    """

    def __init__(self, dim, dim_out, *, key):
        super().__init__(dim, dim_out, 3, padding=0, key=key)

    def __call__(self, x: jax.Array, *, key: Any | None = None) -> jax.Array:
        x = pad2d_circular(x)
        return super().__call__(x, key=key)


class SpatialLinear(eqx.nn.Conv2d):
    """
    A spatial linear module, which is a 1x1 convolution without padding
    """

    def __init__(self, dim, dim_out, *, key):
        super().__init__(dim, dim_out, 1, padding=0, key=key)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(eqx.Module):
    fn: eqx.Module

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def image_resize(x, factor):
    h, w = x.shape[-2:]
    if factor > 1:
        x = jax.image.resize(
            x, (*x.shape[:-2], h * factor, w * factor), method="bilinear"
        )
    elif factor < 1:
        x = jax.image.resize(
            x, (*x.shape[:-2], int(h * factor), int(w * factor)), method="bilinear"
        )
    return x


class Resample(eqx.Module):
    factor: int = eqx.field(static=True)
    conv: DefaultConv2d

    def __init__(self, dim_in, dim_out, factor, *, key):
        self.factor = factor
        self.conv = DefaultConv2d(dim_in, dim_out, key=key)

    def __call__(self, x):
        x = image_resize(x, self.factor)
        return self.conv(x)


class SinusoidalPosEmb(eqx.Module):
    emb: jax.Array

    def __init__(self, dim):
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x):
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class ConvBlock(eqx.Module):
    proj: DefaultConv2d
    norm: eqx.nn.GroupNorm
    mlp: Optional[eqx.nn.Sequential]
    act: Callable

    def __init__(
        self, dim, dim_out, kernel_size=3, emb_dim=None, act=jax.nn.silu, *, key
    ):
        super().__init__()
        keys = jax.random.split(key, 2)
        assert kernel_size == 1 or kernel_size == 3, "kernel size must be 1 or 3"
        if kernel_size == 3:
            self.proj = DefaultConv2d(dim, dim_out, key=keys[0])
        else:
            self.proj = SpatialLinear(dim, dim_out, key=keys[0])
        self.norm = eqx.nn.GroupNorm(
            min(dim_out // 4, 32), dim_out, channelwise_affine=not exists(emb_dim)
        )
        self.mlp = (
            eqx.nn.Sequential(
                [
                    eqx.nn.Lambda(jax.nn.silu),
                    zero_init(eqx.nn.Linear(emb_dim, dim_out * 2, key=keys[1])),
                ]
            )
            if exists(emb_dim)
            else None
        )
        self.act = act

    def __call__(self, x, emb=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(self.mlp) and exists(emb):
            scale_shift = self.mlp(emb)
            scale_shift = rearrange(scale_shift, "c -> c 1 1")
            scale, shift = jnp.split(scale_shift, 2, axis=0)
            # scale + 1 to avoid random scale at initialization
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class LinearTimeSelfAttention(eqx.Module):
    group_norm: eqx.nn.GroupNorm
    heads: int
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(self, dim, heads=4, dim_head=32, *, key):
        keys = jax.random.split(key, 2)
        self.group_norm = eqx.nn.GroupNorm(min(dim // 4, 32), dim)
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = eqx.nn.Conv2d(dim, hidden_dim * 3, 1, key=keys[0])
        self.to_out = eqx.nn.Conv2d(hidden_dim, dim, 1, key=keys[1])

        # model surgery: zero init for better training
        self.to_out = zero_init(self.to_out)

    def __call__(self, x):
        c, h, w = x.shape
        x = self.group_norm(x)
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "(qkv heads c) h w -> qkv heads c (h w)", heads=self.heads, qkv=3
        )
        k = jax.nn.softmax(k, axis=-1)
        context = jnp.einsum("hdn,hen->hde", k, v)
        out = jnp.einsum("hde,hdn->hen", context, q)
        out = rearrange(
            out, "heads c (h w) -> (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


# Neural Differential Appearance Equations
class NDAE(eqx.Module):
    init_conv: ConvBlock
    sinusoidal_pos_emb: SinusoidalPosEmb
    time_mlp: eqx.nn.MLP
    downs: List[List[eqx.Module]]
    mid: List[eqx.Module]
    ups: List[List[eqx.Module]]
    final_conv: List[eqx.Module]

    def __init__(
        self,
        dim,
        in_dim=None,
        out_dim=None,
        dim_mults=(1, 2),
        use_attn=True,
        attn_heads=4,
        attn_head_dim=8,
        *,
        key
    ):
        super().__init__()

        assert dim_mults[0] == 1, "first dim_mult must be 1"

        keys = jax.random.split(key, 6)
        in_dim = default(in_dim, dim)
        out_dim = default(out_dim, dim)

        self.init_conv = ConvBlock(in_dim, dim, kernel_size=1, key=keys[0])

        # time embeddings
        time_dim = dim * 2
        self.sinusoidal_pos_emb = SinusoidalPosEmb(dim)
        self.time_mlp = eqx.nn.MLP(
            dim, time_dim, time_dim, 1, activation=jax.nn.silu, key=keys[1]
        )

        # down, mid, and up layers
        dims = [dim * mult for mult in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = []
        self.ups = []
        convblock = partial(ConvBlock, emb_dim=time_dim)
        attn = partial(
            LinearTimeSelfAttention, heads=attn_heads, dim_head=attn_head_dim
        )

        down_keys = jax.random.split(keys[2], len(in_out))
        for ind, (dim_in, dim_out) in enumerate(in_out):
            _keys = jax.random.split(down_keys[ind], 3)
            down = [
                convblock(dim_in, dim_in, key=_keys[0]),
                Residual(attn(dim_in, key=_keys[1])) if use_attn else eqx.nn.Identity(),
                Resample(dim_in, dim_out, 0.5, key=_keys[2]),
            ]
            self.downs.append(down)

        mid_dim = dims[-1]
        mid_keys = jax.random.split(keys[3], 2)
        self.mid = [
            convblock(mid_dim, mid_dim, key=mid_keys[0]),
            Residual(attn(mid_dim, key=mid_keys[1])) if use_attn else eqx.nn.Identity(),
        ]

        up_keys = jax.random.split(keys[4], len(in_out))
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            _keys = jax.random.split(up_keys[ind], 3)
            up = [
                Resample(dim_out, dim_in, 2, key=_keys[0]),
                convblock(dim_in * 2, dim_in, key=_keys[1]),
                Residual(attn(dim_in, key=_keys[2])) if use_attn else eqx.nn.Identity(),
            ]
            self.ups.append(up)

        final_keys = jax.random.split(keys[5], 2)
        self.final_conv = [
            # use sigmoid activation to avoid unbounded ODE
            ConvBlock(
                dim * 2, dim, kernel_size=1, act=jax.nn.sigmoid, key=final_keys[0]
            ),
            SpatialLinear(dim, out_dim, key=final_keys[1]),
        ]

        # model surgery: zero init for better training
        self.final_conv[-1] = zero_init(self.final_conv[-1])

    @partial(eqx.filter_vmap, in_axes=(None, None, 0, None))
    def __call__(self, time, x, args=None):
        t = self.time_mlp(self.sinusoidal_pos_emb(time))

        x = self.init_conv(x)

        h = []
        h.append(x)

        for block, attn, downsample in self.downs:
            x = block(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        block, attn = self.mid
        x = block(x, t)
        x = attn(x)

        for upsample, block, attn in self.ups:
            x = upsample(x)
            x = jnp.concatenate((x, h.pop()), axis=0)
            x = block(x, t)
            x = attn(x)

        x = jnp.concatenate((x, h.pop()), axis=0)

        for layers in self.final_conv:
            x = layers(x)

        assert len(h) == 0, "all hidden states should be used"

        return x


class NeuralODE(eqx.Module):
    odefunc: eqx.Module
    n_reg: int = field(static=True)

    def __init__(self, odefunc, reg_fns=()):
        self.n_reg = len(reg_fns)
        self.odefunc = RegularizedODEfunc(odefunc, reg_fns)

    def __call__(self, t0, t1, y0, get_reg=False, key=None, **kwargs):
        states = {
            "x": y0,
        }
        args = {"get_reg": get_reg}
        if get_reg and self.n_reg:
            states["reg"] = jnp.zeros(self.n_reg)
            args["_e"] = jax.random.normal(key, y0.shape)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.odefunc),
            t0=t0,
            t1=t1,
            y0=states,
            args=args,
            # max_steps=100_000, # no limit
            **kwargs,
        )

        return solution
