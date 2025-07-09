"""
This module contains functions to compute and visualize the curvature of a homotopy
between two functions, f and g, using their derivatives.

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.optimize import root

INTERESTING_POLYNOMIALS = [
    # (a4, a3, a2, a1, a0)
    (0.7, -6.33, 6.04, -5.88, 28.81),
]


def interactive_homotopy_curvature_plot():
    # Initial coefficient values
    a4_init = 0.0  # x^4 coefficient
    a3_init = 0.0  # x^3 coefficient
    a2_init = 1.0  # x^2 coefficient
    a1_init = 0.0  # x^1 coefficient
    a0_init = -20.0  # x^0 coefficient

    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12, 8))
    plt.subplots_adjust(bottom=0.35)  # Make room for sliders

    class HomotopyFunctions:
        def __init__(self, a4, a3, a2, a1, a0):
            # Coefficients for the polynomial f(x).
            self.a4 = a4
            self.a3 = a3
            self.a2 = a2
            self.a1 = a1
            self.a0 = a0
            # Coefficients for the polynomial g(x).
            self.b2 = 1.0
            self.b1 = 0.0
            self.b0 = -20.0

        def f(self, x):
            return (
                self.a4 * x**4 + self.a3 * x**3 + self.a2 * x**2 + self.a1 * x + self.a0
            )

        def f_prime(self, x):
            return 4 * self.a4 * x**3 + 3 * self.a3 * x**2 + 2 * self.a2 * x + self.a1

        def f_double_prime(self, x):
            return 12 * self.a4 * x**2 + 6 * self.a3 * x + 2 * self.a2

        def g(self, x):
            return self.b2 * x**2 + self.b1 * x + self.b0

        def g_prime(self, x):
            return 2 * self.b2 * x + self.b1

        def g_double_prime(self, x):
            return 2 * self.b2

        def h(self, x, beta):
            return beta * self.g(x) + (1 - beta) * self.f(x)

        def disc(self, beta):
            # Calculate discriminant of the homotopy polynomial. Only valid for a4 = 0
            # and a3 = 0.
            c2 = beta * self.b2 + (1 - beta) * self.a2
            c1 = beta * self.b1 + (1 - beta) * self.a1
            c0 = beta * self.b0 + (1 - beta) * self.a0
            return c1**2 - 4 * c2 * c0

        def h_prime(self, x, beta):
            return beta * self.g_prime(x) + (1 - beta) * self.f_prime(x)

        def h_double_prime(self, x, beta):
            return beta * self.g_double_prime(x) + (1 - beta) * self.f_double_prime(x)

        def hessian_h(self, x, beta):
            def hessian(v1, v2):
                assert v1.shape[-1] == 2, (
                    "Input vector v1 must have ``shape=(..., 2)``."
                )
                assert v2.shape[-1] == 2, (
                    "Input vector v2 must have ``shape=(..., 2)``."
                )
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

        def z(self, x, beta):
            return (self.g(x) - self.f(x)) / self.h_prime(x, beta)

        def beta_dot(self, x, beta):
            return -1 / (self.z(x, beta) ** 2 + 1) ** (1 / 2)

        def x_dot(self, x, beta):
            return -1 * self.beta_dot(x, beta) * self.z(x, beta)

        def c_dot_dot(self, x, beta):
            assert x.shape[-1] == 1, "Input x must have shape (..., 1)."
            assert beta.shape[-1] == 1, "Input beta must have shape (..., 1)."
            c_dot = np.concatenate(
                [self.x_dot(x, beta), self.beta_dot(x, beta)], axis=-1
            )
            hessian = self.hessian_h(x, beta)
            w_2 = hessian(c_dot, c_dot)

            z_2 = -w_2 / self.h_prime(x, beta)

            z_value = self.z(x, beta)
            beta_dot_dot = z_2 * z_value / (z_value**2 + 1)

            x_dot_dot = z_2 - beta_dot_dot * z_value

            return x_dot_dot, beta_dot_dot

        def homotopy_tangent_approx(self, xx, hh):
            tangent = (xx[1:] - xx[:-1]) / hh
            return tangent

        def homotopy_curvature_approx(self, xx, hh):
            tangent = (xx[1:] - xx[:-1]) / hh
            curvature = (tangent[1:] - tangent[:-1]) * (2 / (hh[1:] + hh[:-1]))
            return curvature

        def map_to_arclength(self, x, betas):
            hh = np.abs(self.beta_dot(x, betas))
            dss = (betas[:-1] - betas[1:]) / hh[:-1]
            return np.insert(np.cumsum(dss), 0, 0)[..., None]

        def solve_h_for_beta(self, beta, x0=0):
            if np.isscalar(beta):

                def h_for_fixed_beta(x):
                    return self.h(x, beta)

                sol = root(h_for_fixed_beta, x0)
                return sol.x[0]
            else:
                result = np.zeros_like(beta)
                for i, b in enumerate(beta):
                    result[i] = self.solve_h_for_beta(b, x0)
                return result

        def update_coefficients(self, a4, a3, a2, a1, a0):
            self.a4 = a4
            self.a3 = a3
            self.a2 = a2
            self.a1 = a1
            self.a0 = a0

    # Create the homotopy function object with initial coefficients
    hfunc = HomotopyFunctions(a4_init, a3_init, a2_init, a1_init, a0_init)
    betas = np.linspace(0, 1, 200)[::-1][..., None]

    def update(val):
        # Get current slider values
        a4 = a4_slider.val
        a3 = a3_slider.val
        a2 = a2_slider.val
        a1 = a1_slider.val
        a0 = a0_slider.val

        # Update coefficients in the homotopy functions
        hfunc.update_coefficients(a4, a3, a2, a1, a0)

        # Clear the axes
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Recalculate with new coefficients
        try:
            # Solve for x values using the homotopy function. Initial guess is previous
            # solution to ensure a continuous curve in the case of multiple real
            # solutions.
            # This could be sped up at the cost of lower accuracy by using the previous
            # solution for the next 10 iterations.
            x0 = -3
            xx = np.array([x0 := hfunc.solve_h_for_beta(b, x0=x0) for b in betas])

            ss = hfunc.map_to_arclength(xx, betas)
            tangent = hfunc.x_dot(xx, betas)

            # Plotting code
            # ax1.quiver(
            #     ss[:-10:10],
            #     xx[:-10:10],
            #     ss[10::10] - ss[:-10:10],
            #     tangent[:-10:10].squeeze() * (ss[10::10] - ss[:-10:10]).squeeze(),
            #     angles="xy",
            #     scale_units="xy",
            #     scale=1,
            #     color="k",
            #     alpha=0.7,
            #     width=0.005,
            #     label=r"$x$ tangents",
            # )
            # ax1.quiver(
            #     ss[:-10:10],
            #     betas[:-10:10],
            #     ss[10::10] - ss[:-10:10],
            #     hfunc.beta_dot(xx, betas)[:-10:10].squeeze()
            #     * (ss[10::10] - ss[:-10:10]).squeeze(),
            #     angles="xy",
            #     scale_units="xy",
            #     scale=1,
            #     color="y",
            #     alpha=0.7,
            #     width=0.005,
            #     label=r"$\lambda$ tangents",
            # )
            ax1.plot(ss, betas, "y-", lw=1.5, label=r"$\lambda$")
            # ax1.plot(
            #     ss, np.sign(hfunc.disc(betas)), "y:", lw=1.5, label=r"$sgn(\Delta)$"
            # )
            ax1.plot(ss, xx, "k-", lw=1.5, label=r"$x$")
            ax1.plot(ss, tangent, "r-.", lw=1.5, label=r"$\dot{x}$")
            ax1.plot(
                ss,
                hfunc.c_dot_dot(xx, betas)[0],
                "r--",
                lw=1.5,
                label=r"$\ddot{x}$",
            )
            ax1.plot(
                ss[1:-1],
                hfunc.homotopy_curvature_approx(xx, ss[1:] - ss[:-1]),
                "r:",
                lw=1.5,
                label=r"$\ddot{x}_{approx}$",
            )

            # Display polynomial equation in the title
            eq = f"$f(x) = {a4:.2f}x^4"
            if a3 >= 0:
                eq += f" + {a3:.2f}x^3"
            else:
                eq += f" - {abs(a3):.2f}x^3"
            if a2 >= 0:
                eq += f" + {a2:.2f}x^2"
            else:
                eq += f" - {abs(a2):.2f}x^2"
            if a1 >= 0:
                eq += f" + {a1:.2f}x"
            else:
                eq += f" - {abs(a1):.2f}x"
            if a0 >= 0:
                eq += f" + {a0:.2f}$"
            else:
                eq += f" - {abs(a0):.2f}$"

            ax1.set_title(eq + "\nHomotopy Curvature")
            ax1.set_xlabel(r"$s$")
            ax1.legend()

            ax2.plot(
                ss, hfunc.g(xx) - hfunc.f(xx), "g-.", lw=1.5, label=r"$g(x) - f(x)$"
            )
            ax2.plot(
                ss,
                hfunc.h_prime(xx, betas),
                "b-.",
                lw=1.5,
                label=r"$\partial_{x} h$",
            )
            ax2.plot(
                ss,
                hfunc.h_double_prime(xx, betas),
                "b--",
                lw=1.5,
                label=r"$\partial_{x}^2 h$",
            )
            ax2.set_xlabel(r"$s$")
            ax2.legend()

            ax3.plot(betas, xx, "k-", lw=1.5, label=r"$x$")
            ax3.plot(
                betas,
                -1 * hfunc.z(xx, betas),
                "r-.",
                lw=1.5,
                label=r"$\dot{x} / \dot{\lambda}$",
            )
            ax3.plot(
                betas[1:],
                hfunc.homotopy_tangent_approx(xx, betas[1:] - betas[:-1]),
                "r.",
                lw=1.5,
                label=r"$\partial_{lambda, approx} x$",
            )
            x_dot = hfunc.x_dot(xx, betas)
            beta_dot = hfunc.beta_dot(xx, betas)
            x_dot_dot, beta_dot_dot = hfunc.c_dot_dot(xx, betas)
            x_prime_prime = (x_dot_dot * beta_dot - x_dot * beta_dot_dot) / (
                beta_dot**3
            )
            ax3.plot(
                betas,
                x_prime_prime,
                "r--",
                lw=1.5,
                label=r"$\partial_{\lambda}^2 x$",
            )
            ax3.plot(
                betas[1:-1],
                hfunc.homotopy_curvature_approx(xx, betas[1:] - betas[:-1]),
                "r:",
                lw=1.5,
                label=r"$\partial_{lambda, approx}^2 x$",
            )

            ax3.set_xlabel(r"$\lambda$")
            ax3.set_xlim(1, 0)
            ax3.set_ylim(-10, 10)
            ax3.legend()

        except Exception as e:
            ax1.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        fig.canvas.draw_idle()

    # Create sliders
    ax_a4 = plt.axes((0.1, 0.15, 0.35, 0.03))
    ax_a3 = plt.axes((0.55, 0.15, 0.35, 0.03))
    ax_a2 = plt.axes((0.1, 0.1, 0.35, 0.03))
    ax_a1 = plt.axes((0.55, 0.1, 0.35, 0.03))
    ax_a0 = plt.axes((0.1, 0.05, 0.35, 0.03))

    a4_slider = Slider(ax_a4, r"$x^4$", -5.0, 5.0, valinit=a4_init)
    a3_slider = Slider(ax_a3, r"$x^3$", -20.0, 20.0, valinit=a3_init)
    a2_slider = Slider(ax_a2, r"$x^2$", -20.0, 20.0, valinit=a2_init)
    a1_slider = Slider(ax_a1, r"$x^1$", -20.0, 20.0, valinit=a1_init)
    a0_slider = Slider(ax_a0, r"$x^0$", -30.0, 30.0, valinit=a0_init)

    # Connect update function to sliders
    a4_slider.on_changed(update)
    a3_slider.on_changed(update)
    a2_slider.on_changed(update)
    a1_slider.on_changed(update)
    a0_slider.on_changed(update)

    # Add a reset button to restore initial coefficients
    def reset(event):
        a4_slider.set_val(a4_init)
        a3_slider.set_val(a3_init)
        a2_slider.set_val(a2_init)
        a1_slider.set_val(a1_init)
        a0_slider.set_val(a0_init)

    reset_button_ax = plt.axes((0.65, 0.05, 0.08, 0.04))
    reset_button = plt.Button(reset_button_ax, "Reset")
    reset_button.on_clicked(reset)

    # Add a button to cycle through interesting polynomials
    current_index = 0

    def cycle_polynomials(event):
        nonlocal current_index
        next_index = (current_index + 1) % len(INTERESTING_POLYNOMIALS)
        next_poly = INTERESTING_POLYNOMIALS[next_index]
        a4_slider.set_val(next_poly[0])
        a3_slider.set_val(next_poly[1])
        a2_slider.set_val(next_poly[2])
        a1_slider.set_val(next_poly[3])
        a0_slider.set_val(next_poly[4])
        current_index = next_index

    cycle_button_ax = plt.axes((0.55, 0.05, 0.08, 0.04))
    cycle_button = plt.Button(cycle_button_ax, "Cycle Poly")
    cycle_button.on_clicked(cycle_polynomials)

    # Initial update to show the plot
    update(None)

    plt.show()


interactive_homotopy_curvature_plot()
