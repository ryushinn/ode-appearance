import six
import jax.numpy as jnp
import equinox as eqx
from jax import vjp
from typing import Callable


def quadratic_cost(t, x, dx, vdfdx_fun, args, context):
    dx = dx.reshape(-1)
    return 0.5 * jnp.mean(dx**2)


def jacobian_frobenius_regularization_fn(t, x, dx, vdfdx_fun, args, context):
    _e = args["_e"]
    (e_dfdx,) = vdfdx_fun(_e)
    sqjacnorm = jnp.mean(e_dfdx.reshape(-1) ** 2)
    return sqjacnorm


REGULARIZATION_FNS = {
    "kinetic_energy": quadratic_cost,
    "jacobian_norm2": jacobian_frobenius_regularization_fn,
}


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if getattr(args, arg_key) is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(eval("args." + arg_key))

    regularization_fns = tuple(regularization_fns)
    regularization_coeffs = tuple(regularization_coeffs)
    return regularization_fns, regularization_coeffs


class RegularizedODEfunc(eqx.Module):
    odefunc: eqx.Module
    reg_fns: tuple[Callable] = eqx.field(static=True)

    def __init__(self, odefunc, regularization_fns):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.reg_fns = regularization_fns

    def __call__(self, t, states, args):
        x = states["x"]
        dx, vjpfunc = vjp(lambda x: self.odefunc(t, x, args), x)

        dstates = {"x": dx}

        if args["get_reg"] and self.reg_fns:
            dreg = jnp.stack(
                tuple(reg_fn(t, x, dx, vjpfunc, args, self) for reg_fn in self.reg_fns)
            )
            dstates["reg"] = dreg

        return dstates
