"""
This module contains functions to compute and visualize the curvature of a homotopy
between two functions, f and g, using their derivatives.

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

"""

from typing import Callable, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

NumericInput = float | int | np.ndarray


class HomotopyCurvature:
    """Class to compute the curvature of a homotopy between two functions."""

    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        g: Callable[[np.ndarray], np.ndarray],
        f_prime: Callable[[np.ndarray], np.ndarray],
        g_prime: Callable[[np.ndarray], np.ndarray],
        f_double_prime: Callable[[np.ndarray], np.ndarray],
        g_double_prime: Callable[[np.ndarray], np.ndarray],
        x0: float = 0.0,
    ):
        self.f = f
        self.g = g
        self.f_prime = f_prime
        self.g_prime = g_prime
        self.f_double_prime = f_double_prime
        self.g_double_prime = g_double_prime

        self.x = x0

    def h(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return beta * self.g(x) + (1 - beta) * self.f(x)

    def h_prime(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return beta * self.g_prime(x) + (1 - beta) * self.f_prime(x)

    def h_double_prime(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return beta * self.g_double_prime(x) + (1 - beta) * self.f_double_prime(x)

    def hessian_h(
        self, x: np.ndarray, beta: NumericInput
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def hessian(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            assert v1.shape[-1] == 2, "Input vector v1 must have shape (..., 2)."
            assert v2.shape[-1] == 2, "Input vector v2 must have shape (..., 2)."
            h_xx = self.h_double_prime(x, beta)
            h_xbeta = self.g_prime(x) - self.f_prime(x)
            h_betabeta = np.zeros_like(h_xx)
            hessian_mat = np.stack(
                [
                    np.concatenate([h_xx, h_xbeta], axis=-1),
                    np.concatenate([h_xbeta, h_betabeta], axis=-1),
                ],
                axis=-1,
            )
            assert hessian_mat.shape[-2:] == (2, 2), (
                "Hessian must have shape (..., 2, 2)."
            )
            return np.sum(
                np.matmul(hessian_mat, v1[..., None]).squeeze() * v2, axis=-1
            )[..., None]

        return hessian

    def z(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return (self.g(x) - self.f(x)) / self.h_prime(x, beta)

    def beta_dot(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return -1 / (self.z(x, beta) ** 2 + 1) ** 0.5

    def x_dot(self, x: np.ndarray, beta: NumericInput) -> np.ndarray:
        return -1 * self.beta_dot(x, beta) * self.z(x, beta)

    def x_dot_dot(self, xs: np.ndarray, betas: np.ndarray):
        # Calculate :math:`w_2=\nabla^2 H(c(s))[\dot{c}(s), \dot{c}(s)]`.
        assert xs.shape[-1] == 1, "Input x must have shape (..., 1)."
        assert betas.shape[-1] == 1, "Input beta must have shape (..., 1)."
        c_dot = np.concatenate(
            [self.x_dot(xs, betas), self.beta_dot(xs, betas)], axis=-1
        )  # ``shape=(n, 2)``
        hessian = self.hessian_h(xs, betas)
        w_2 = hessian(c_dot, c_dot)
        assert w_2.shape[-1] == 1, "w_2 must have shape (..., 1)."

        # Solve :math:`\nabla_x H(c(s)) z_2 = -w_2`.
        z_2 = -w_2 / self.h_prime(xs, betas)

        # Calculate :math:`\ddot{\beta} = \frac{z_2 \cdot z}{z \cdot z + 1}`.
        z_value = self.z(xs, betas)
        beta_dot_dot = z_2 * z_value / (z_value**2 + 1)

        # Sanity check: ensure that beta_dot_dot is close to its finite difference
        # approximation.
        # ss = self.map_to_arclength(x: NumericInput, beta: float)[..., None]
        # hh = ss[1:] - ss[:-1]
        # beta_dot_approx = (beta[1:] - beta[:-1]) / hh
        # beta_dot_dot_approx = (beta_dot_approx[1:] - beta_dot_approx[:-1]) * (
        #     2 / (hh[1:] + hh[:-1])
        # )
        # assert np.isclose(beta_dot_dot[1:-1], beta_dot_dot_approx: NumericInput, rtol=0.1).all(), (
        #     "beta_dot_dot does not match finite difference approximation."
        # )

        # Finally, calculate :math:`\ddot{x} = z_2 - \ddot{beta} z`.
        return z_2 - beta_dot_dot * z_value

    def homotopy_curvature(self, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return np.abs(self.x_dot_dot(x, beta))

    @staticmethod
    def homotopy_curvature_approx(xs: np.ndarray, ss: np.ndarray) -> np.ndarray:
        # Second derivative approximation using finite differences. Note that h is not
        # constant, but varies with ss.
        hh = ss[1:] - ss[:-1]
        tangent = (xs[1:] - xs[:-1]) / hh
        curvature = (tangent[1:] - tangent[:-1]) * (2 / (hh[1:] + hh[:-1]))
        return curvature

    def map_to_arclength(self, x: np.ndarray, betas: np.ndarray) -> np.ndarray:
        hh = np.abs(self.beta_dot(x, betas))
        # Loop through betas and beta primes and use the IFT and finite differences to
        # approximate the arclength. Note that beta decreases, i.e., beta_i > beta_{i+1}.
        dss = (betas[:-1] - betas[1:]) / hh[:-1]
        # Compute cumulative sum to get the arclength and insert 0 at the start.
        return np.insert(np.cumsum(dss), 0, 0)[..., None]

    def solve_h_for_beta(self, beta: NumericInput, x0: float = 0.0) -> NumericInput:
        if np.isscalar(beta):
            # For scalar, solve only for the given beta.

            def h_for_fixed_beta(x: float) -> float:
                return self.h(x, beta)  # type: ignore

            sol = root(h_for_fixed_beta, x0)
            return sol.x[0]
        else:
            # For array of betas, solve for each beta.
            result = np.zeros_like(beta)  #
            beta = cast(np.ndarray, beta)
            for i, b in enumerate(beta):
                result[i] = self.solve_h_for_beta(b, x0)
            return result

    def plot_curvature(self) -> None:
        betas = np.linspace(0, 1, 100)[::-1][..., None]
        xs: np.ndarray = self.solve_h_for_beta(betas, x0=-3.0)  # type: ignore
        ss: np.ndarray = self.map_to_arclength(xs, betas)  # type: ignore
        tangent = self.x_dot(xs, betas)  # type: ignore
        plt.figure(figsize=(10, 6))
        plt.quiver(
            ss[:-10:10],
            xs[:-10:10],
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
            self.beta_dot(xs, betas)[:-10:10].squeeze()
            * (ss[10::10] - ss[:-10:10]).squeeze(),
            angles="xy",
            scale_units="xy",
            scale=1,
            color="blue",
            alpha=0.7,
            width=0.005,
            label=r"$\lambda$ tangents",
        )
        plt.plot(ss, betas, "k-", lw=1.5, label=r"$\lambda$")
        plt.plot(ss, xs, "r-", lw=1.5, label=r"$x$")
        plt.plot(ss, tangent, "r-.", lw=1.5, label=r"$\dot{x}$")
        plt.plot(ss, self.x_dot_dot(xs, betas), "r--", lw=1.5, label=r"$\ddot{x}$")
        plt.plot(
            ss[1:-1],
            self.homotopy_curvature_approx(xs, ss),
            "r:",
            lw=1.5,
            label=r"$\ddot{x}_{approx}$",
        )
        plt.plot(ss, self.g(xs) - self.f(xs), "g-.", lw=1.5, label=r"$g(x) - f(x)$")
        plt.plot(ss, self.h_prime(xs, betas) ** 2, "b-.", lw=1.5, label=r"$\dot{h}^2$")
        plt.plot(ss, self.h_double_prime(xs, betas), "b--", lw=1.5, label=r"$\ddot{h}$")
        plt.title("Homotopy Curvature")
        plt.xlabel(r"$s$")
        plt.ylim(-10, 10)
        plt.legend()
        plt.show()
