"""
This module contains functions to compute and visualize the curvature of a homotopy
between two functions, f and g, using their derivatives.

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root


def f(x):
    return x**4 - 15 * x**3 + 3 * x - 20


def f_prime(x):
    return 4 * x**3 - 45 * x**2 + 3


def f_double_prime(x):
    return 12 * x**2 - 90 * x


def g(x):
    return x**2 - 20


def g_prime(x):
    return 2 * x


def g_double_prime(x):
    return 2


def h(x, beta):
    return beta * g(x) + (1 - beta) * f(x)


def h_prime(x, beta):
    return beta * g_prime(x) + (1 - beta) * f_prime(x)


def h_double_prime(x, beta):
    return beta * g_double_prime(x) + (1 - beta) * f_double_prime(x)


def hessian_h(x, beta):
    def hessian(v1, v2):
        assert v1.shape[-1] == 2, "Input vector v1 must have ``shape=(..., 2)``."
        assert v2.shape[-1] == 2, "Input vector v2 must have ``shape=(..., 2)``."
        h_xx = h_double_prime(x, beta)
        h_xbeta = g_prime(x) - f_prime(x)
        h_betabeta = np.zeros_like(h_xx)
        hessian_mat = np.stack(
            [
                np.concat([h_xx, h_xbeta], axis=-1),
                np.concat([h_xbeta, h_betabeta], axis=-1),
            ],
            axis=-1,
        )
        assert hessian_mat.shape[-2:] == (2, 2), "Hessian must have shape (..., 2, 2)."
        return np.sum(np.matmul(hessian_mat, v1[..., None]).squeeze() * v2, axis=-1)[
            ..., None
        ]

    return hessian


def z(x, beta):
    return (g(x) - f(x)) / h_prime(x, beta)


def beta_dot(x, beta):
    return -1 / (z(x, beta) ** 2 + 1) ** (1 / 2)


def x_dot(x, beta):
    return -1 * beta_dot(x, beta) * z(x, beta)


def x_dot_dot(x, beta):
    # Calculate :math:`w_2=\nabla^2 H(c(s))[\dot{c}(s), \dot{c}(s)]`.
    assert x.shape[-1] == 1, "Input x must have shape (..., 1)."
    assert beta.shape[-1] == 1, "Input beta must have shape (..., 1)."
    c_dot = np.concat([x_dot(x, beta), beta_dot(x, beta)], axis=-1)  # ``shape=(n, 2)``
    hessian = hessian_h(x, beta)
    w_2 = hessian(c_dot, c_dot)
    assert w_2.shape[-1] == 1, "w_2 must have shape (..., 1)."

    # Solve :math:`\nabla_x H(c(s)) z_2 = -w_2`.
    z_2 = -w_2 / h_prime(x, beta)

    # Calculate :math:`\ddot{\beta} = \frac{z_2 \cdot z}{z \cdot z + 1}`.
    z_value = z(x, beta)
    beta_dot_dot = z_2 * z_value / (z_value**2 + 1)

    # Sanity check: ensure that beta_dot_dot is close to its finite difference
    # approximation.
    ss = map_to_arclength(x, beta)[..., None]
    hh = ss[1:] - ss[:-1]
    beta_dot_approx = (beta[1:] - beta[:-1]) / hh
    beta_dot_dot_approx = (beta_dot_approx[1:] - beta_dot_approx[:-1]) * (
        2 / (hh[1:] + hh[:-1])
    )

    # assert np.isclose(beta_dot_dot[1:-1], beta_dot_dot_approx, rtol=0.1).all(), (
    #     "beta_dot_dot does not match finite difference approximation."
    # )

    # Finally, calculate :math:`\ddot{x} = z_2 - \ddot{beta} z`.
    return z_2 - beta_dot_dot * z_value


def homotopy_curvature(x, beta):
    return np.abs(x_dot_dot(x, beta))


def homotopy_curvature_approx(xx, ss):
    # Second derivative approximation using finite differences. Note that h is not
    # constant, but varies with ss.
    hh = ss[1:] - ss[:-1]
    tangent = (xx[1:] - xx[:-1]) / hh
    curvature = (tangent[1:] - tangent[:-1]) * (2 / (hh[1:] + hh[:-1]))
    return curvature


def map_to_arclength(x, betas):
    hh = np.abs(beta_dot(x, betas))
    # Loop through betas and beta primes and use the IFT and finite differences to
    # approximate the arclength. Note that beta decreases, i.e., beta_i > beta_{i+1}.
    dss = (betas[:-1] - betas[1:]) / hh[:-1]
    # Compute cumulative sum to get the arclength and insert 0 at the start.
    return np.insert(np.cumsum(dss), 0, 0)[..., None]


def solve_h_for_beta(beta, x0=0):
    if np.isscalar(beta):

        def h_for_fixed_beta(x):
            return h(x, beta)

        sol = root(h_for_fixed_beta, x0)
        return sol.x[0]
    else:
        # For array of betas, solve for each beta
        result = np.zeros_like(beta)
        for i, b in enumerate(beta):
            result[i] = solve_h_for_beta(b, x0)
        return result


def homotopy_curvature_plot():
    betas = np.linspace(0, 1, 100)[::-1][..., None]
    xx = solve_h_for_beta(betas, x0=-3)

    ss = map_to_arclength(xx, betas)
    tangent = x_dot(xx, betas)

    plt.figure(figsize=(10, 6))
    plt.quiver(
        ss[:-10:10],
        xx[:-10:10],
        ss[10::10] - ss[:-10:10],
        tangent[:-10:10].squeeze() * (ss[10::10] - ss[:-10:10]).squeeze(),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        alpha=0.7,
        width=0.005,
        label=r"$x$ tangents",
    )
    plt.quiver(
        ss[:-10:10],
        betas[:-10:10],
        ss[10::10] - ss[:-10:10],
        beta_dot(xx, betas)[:-10:10].squeeze() * (ss[10::10] - ss[:-10:10]).squeeze(),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="blue",
        alpha=0.7,
        width=0.005,
        label=r"$\lambda$ tangents",
    )
    plt.plot(ss, betas, "k-", lw=1.5, label=r"$\lambda$")
    plt.plot(ss, xx, "r-", lw=1.5, label=r"$x$")
    plt.plot(ss, tangent, "r-.", lw=1.5, label=r"$\dot{x}$")
    plt.plot(
        ss,
        x_dot_dot(xx, betas),
        "r--",
        lw=1.5,
        label=r"$\ddot{x}$",
    )
    plt.plot(
        ss[1:-1],
        homotopy_curvature_approx(xx, ss),
        "r:",
        lw=1.5,
        label=r"$\ddot{x}_{approx}$",
    )
    plt.plot(ss, g(xx) - f(xx), "g-.", lw=1.5, label=r"$g(x) - f(x)$")
    plt.plot(ss, h_prime(xx, betas) ** 2, "b-.", lw=1.5, label=r"$\dot{h}^2$")
    plt.plot(ss, h_double_prime(xx, betas), "b--", lw=1.5, label=r"$\ddot{h}$")

    plt.title("Homotopy Curvature")
    plt.xlabel(r"$s$")
    plt.ylim(-10, 10)
    plt.legend()
    plt.show()


homotopy_curvature_plot()
