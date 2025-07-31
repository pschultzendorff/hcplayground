"""This module contains functions to compute and visualize the curvature of a homotopy
between two functions, f and g, using their derivatives.

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

TODO
- Log scale toggle resets automatically when the f polynomial is changed, but this is
not displayed in the UI.
- Can the same error occur for the y-limits toggle?

"""

import matplotlib.pyplot as plt
import numpy as np
from curvature import HomotopyCurvature
from matplotlib.widgets import CheckButtons, Slider

INTERESTING_POLYNOMIALS = [
    # (a4, a3, a2, a1, a0)
    (0.7, -6.33, 6.04, -5.88, 28.81),
]


def interactive_homotopy_curvature_plot():
    a4_init = 0.0  # x^4 coefficient
    a3_init = 0.0  # x^3 coefficient
    a2_init = 1.0  # x^2 coefficient
    a1_init = 0.0  # x^1 coefficient
    a0_init = -20.0  # x^0 coefficient

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, figsize=(13, 10), gridspec_kw={"hspace": 0.4}
    )
    plt.subplots_adjust(bottom=0.25, right=0.8)  # Make room for sliders and checkboxes.

    class HomotopyCurvaturePolynomial(HomotopyCurvature):
        def __init__(self, a4, a3, a2, a1, a0):
            # Coefficients for the polynomial f(x).
            self.a4: float = a4
            self.a3: float = a3
            self.a2: float = a2
            self.a1: float = a1
            self.a0: float = a0
            # Coefficients for the polynomial g(x).
            self.b2: float = 0.0
            self.b1: float = 0.0
            self.b0: float = -20.0
            # Initial guess for the root-finding algorithm.
            self.x0: float = -3.0

        def f(self, x):
            return (
                self.a4 * x**4 + self.a3 * x**3 + self.a2 * x**2 + self.a1 * x + self.a0
            )

        def f_prime(self, x):
            return 4 * self.a4 * x**3 + 3 * self.a3 * x**2 + 2 * self.a2 * x + self.a1

        def f_dprime(self, x):
            return 12 * self.a4 * x**2 + 6 * self.a3 * x + 2 * self.a2

        def g(self, x):
            return self.b2 * x**2 + self.b1 * x + self.b0

        def g_prime(self, x):
            return 2 * self.b2 * x + self.b1

        def g_dprime(self, x):
            return 2 * self.b2

        def disc(self, beta):
            # Calculate discriminant of the homotopy polynomial. Only valid for a4 = 0
            # and a3 = 0.
            c2 = beta * self.b2 + (1 - beta) * self.a2
            c1 = beta * self.b1 + (1 - beta) * self.a1
            c0 = beta * self.b0 + (1 - beta) * self.a0
            return c1**2 - 4 * c2 * c0

        def h_dprime(self, x, beta):
            return beta * self.g_dprime(x) + (1 - beta) * self.f_dprime(x)

        def c_ddot(self, x, beta):
            assert x.shape[-1] == 1, "Input x must have shape (..., 1)."
            assert beta.shape[-1] == 1, "Input beta must have shape (..., 1)."
            c_dot = np.concatenate(
                [self.x_dot(x, beta), self.beta_dot(x, beta)], axis=-1
            )
            hessian = self.hessian_h(x, beta)
            w_2 = hessian(c_dot, c_dot)

            z_2 = -w_2 / self.h_prime(x, beta)

            z_value = self.z(x, beta)
            beta_ddot = z_2 * z_value / (z_value**2 + 1)

            x_ddot = z_2 - beta_ddot * z_value

            return x_ddot, beta_ddot

        def homotopy_tangent_approx(self, xx, hh):
            tangent = (xx[1:] - xx[:-1]) / hh
            return tangent

        def update_coefficients(self, a4, a3, a2, a1, a0):
            self.a4 = a4
            self.a3 = a3
            self.a2 = a2
            self.a1 = a1
            self.a0 = a0

    # Create the homotopy function object with initial coefficients.
    hfunc = HomotopyCurvaturePolynomial(a4_init, a3_init, a2_init, a1_init, a0_init)
    betas: np.ndarray = np.linspace(0, 1, 200)[::-1][..., None]

    # Objects to store plot objects and visibility state.
    plot_objects = {
        "ax1": {},
        "ax2": {},
        "ax3": {},
        "ax4": {},
    }
    visibility = {
        "ax1": {},
        "ax2": {},
        "ax3": {},
        "ax4": {},
    }

    # Keys for all plots
    keys_ax1 = [
        r"$x$ tangents",
        r"$\lambda$ tangents",
        r"$\lambda$",
        r"$sgn(\Delta)$",
        r"$x$",
        r"$\dot{x}$",
        r"$\ddot{x}$",
        r"$\ddot{x}_{approx}$",
    ]
    keys_ax2 = [r"$g(x)-f(x)$", r"$\partial_x h$", r"$\partial_x^2 h$"]
    keys_ax3 = [r"$g(x)-f(x)$", r"$\partial_x h$", r"$\partial_x^2 h$"]
    keys_ax4 = [
        r"$x$",
        r"$\dot{x}/\dot{\lambda}$",
        r"$\partial_{\lambda,appr} x$",
        r"$\partial_{\lambda}^2 x$",
        r"$\partial_{\lambda,appr}^2 x$",
    ]

    # Change coefficients of f(x) with sliders.
    def recalc_and_plot(val):
        nonlocal plot_objects, visibility
        a4 = a4_slider.val
        a3 = a3_slider.val
        a2 = a2_slider.val
        a1 = a1_slider.val
        a0 = a0_slider.val
        hfunc.update_coefficients(a4, a3, a2, a1, a0)

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Recalculate with new coefficients and plot.
        try:
            # Solve for x values using the homotopy function. Initial guess is previous
            # solution to ensure a continuous curve in the case of multiple real
            # solutions.
            # This could be sped up at the cost of lower accuracy by using the previous
            # solution for the next 10 iterations or so.
            x0 = hfunc.x0
            xx = np.array([x0 := hfunc.solve_h_for_beta(b, x0=x0) for b in betas])

            ss = hfunc.map_to_arclength(xx, betas)
            tangent = hfunc.x_dot(xx, betas)

            # Calculate and plot all plots and store them in plot_objects.

            # First subplot. Homotopy curvature and tangent with respect to
            # s-parameterization.
            plot_objects["ax1"][keys_ax1[0]] = ax1.quiver(
                ss[:-10:10],
                xx[:-10:10],
                ss[10::10] - ss[:-10:10],
                tangent[:-10:10].squeeze() * (ss[10::10] - ss[:-10:10]).squeeze(),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="k",
                alpha=0.7,
                width=0.005,
                label=keys_ax1[0],
                visible=visibility["ax1"][keys_ax1[0]],
            )
            plot_objects["ax1"][keys_ax1[1]] = ax1.quiver(
                ss[:-10:10],
                betas[:-10:10],
                ss[10::10] - ss[:-10:10],
                hfunc.beta_dot(xx, betas)[:-10:10].squeeze()
                * (ss[10::10] - ss[:-10:10]).squeeze(),
                angles="xy",
                scale_units="xy",
                scale=1,
                color="y",
                alpha=0.7,
                width=0.005,
                label=keys_ax1[1],
                visible=visibility["ax1"][keys_ax1[1]],
            )
            plot_objects["ax1"][keys_ax1[2]] = ax1.plot(
                ss,
                betas,
                "y-",
                lw=1.5,
                label=keys_ax1[2],
                visible=visibility["ax1"][keys_ax1[2]],
            )[0]
            plot_objects["ax1"][keys_ax1[3]] = ax1.plot(
                ss,
                np.sign(hfunc.disc(betas)),
                "y:",
                lw=1.5,
                label=keys_ax1[3],
                visible=visibility["ax1"][keys_ax1[3]],
            )[0]
            plot_objects["ax1"][keys_ax1[4]] = ax1.plot(
                ss,
                xx,
                "k-",
                lw=1.5,
                label=keys_ax1[4],
                visible=visibility["ax1"][keys_ax1[4]],
            )[0]
            plot_objects["ax1"][keys_ax1[5]] = ax1.plot(
                ss,
                tangent,
                "r-.",
                lw=1.5,
                label=keys_ax1[5],
                visible=visibility["ax1"][keys_ax1[5]],
            )[0]
            plot_objects["ax1"][keys_ax1[6]] = ax1.plot(
                ss,
                hfunc.c_ddot(xx, betas)[0],
                "r--",
                lw=1.5,
                label=keys_ax1[6],
                visible=visibility["ax1"][keys_ax1[6]],
            )[0]
            plot_objects["ax1"][keys_ax1[7]] = ax1.plot(
                ss[1:-1],
                hfunc.homotopy_curvature_approx(xx, ss[1:] - ss[:-1]),
                "r:",
                lw=1.5,
                label=keys_ax1[7],
                visible=visibility["ax1"][keys_ax1[7]],
            )[0]

            # Second and third subplot. Homotopy function and its derivatives. Once with
            # respect to s-parameterization and once with respect to λ-parameterization.
            for ax, ax_key, x_values in zip([ax2, ax3], ["ax2", "ax3"], [ss, betas]):
                plot_objects[ax_key][keys_ax2[0]] = ax.plot(
                    x_values,
                    hfunc.g(xx) - hfunc.f(xx),
                    "g-.",
                    lw=1.5,
                    label=keys_ax2[0],
                    visible=visibility[ax_key][keys_ax2[0]],
                )[0]
                plot_objects[ax_key][keys_ax2[1]] = ax.plot(
                    x_values,
                    hfunc.h_prime(xx, betas),
                    "b-.",
                    lw=1.5,
                    label=keys_ax2[1],
                    visible=visibility[ax_key][keys_ax2[1]],
                )[0]
                plot_objects[ax_key][keys_ax2[2]] = ax.plot(
                    x_values,
                    hfunc.h_dprime(xx, betas),
                    "b--",
                    lw=1.5,
                    label=keys_ax2[2],
                    visible=visibility[ax_key][keys_ax2[2]],
                )[0]

            # Fourth subplot. Tangent and curvature with respect to λ-parametrization.
            plot_objects["ax4"][keys_ax4[0]] = ax4.plot(
                betas,
                xx,
                "k-",
                lw=1.5,
                label=keys_ax4[0],
                visible=visibility["ax4"][keys_ax4[0]],
            )[0]
            plot_objects["ax4"][keys_ax4[1]] = ax4.plot(
                betas,
                -1 * hfunc.z(xx, betas),
                "r-.",
                lw=1.5,
                label=keys_ax4[1],
                visible=visibility["ax4"][keys_ax4[1]],
            )[0]
            plot_objects["ax4"][keys_ax4[2]] = ax4.plot(
                betas[1:],
                hfunc.homotopy_tangent_approx(xx, betas[1:] - betas[:-1]),
                "r.",
                lw=1.5,
                label=keys_ax4[2],
                visible=visibility["ax4"][keys_ax4[2]],
            )[0]
            x_dot = hfunc.x_dot(xx, betas)
            beta_dot = hfunc.beta_dot(xx, betas)
            x_ddot, beta_ddot = hfunc.c_ddot(xx, betas)
            x_prime_prime = (x_ddot * beta_dot - x_dot * beta_ddot) / (beta_dot**3)
            plot_objects["ax4"][keys_ax4[3]] = ax4.plot(
                betas,
                x_prime_prime,
                "r--",
                lw=1.5,
                label=keys_ax4[3],
                visible=visibility["ax4"][keys_ax4[3]],
            )[0]
            plot_objects["ax4"][keys_ax4[4]] = ax4.plot(
                betas[1:-1],
                hfunc.homotopy_curvature_approx(xx, betas[1:] - betas[:-1]),
                "r:",
                lw=1.5,
                label=keys_ax4[4],
                visible=visibility["ax4"][keys_ax4[4]],
            )[0]

        except Exception as e:
            # If an error occurs, toggle off all plots and display the error in the
            # first plot.
            for ax in plot_objects.keys():
                for label in plot_objects[ax].keys():
                    visibility[ax][label] = False
            ax1.text(
                0.5,
                0.5,
                f"Error: {str(e)}",
                ha="center",
                va="center",
            )

        # Display polynomial equations in the title.
        eq_f = "$f(x) = "
        if a4 >= 0:
            eq_f += f"{a4:.2f}x^4"
        else:
            eq_f += f"-{abs(a4):.2f}x^4"
        if a3 >= 0:
            eq_f += f" + {a3:.2f}x^3"
        else:
            eq_f += f" - {abs(a3):.2f}x^3"
        if a2 >= 0:
            eq_f += f" + {a2:.2f}x^2"
        else:
            eq_f += f" - {abs(a2):.2f}x^2"
        if a1 >= 0:
            eq_f += f" + {a1:.2f}x"
        else:
            eq_f += f" - {abs(a1):.2f}x"
        if a0 >= 0:
            eq_f += f" + {a0:.2f}$"
        else:
            eq_f += f" - {abs(a0):.2f}$"

        eq_g = "\n$g(x) = "
        if hfunc.b2 >= 0:
            eq_g += f"{hfunc.b2:.2f}x^2"
        else:
            eq_g += f"-{abs(hfunc.b2):.2f}x^2"
        if hfunc.b1 >= 0:
            eq_g += f" + {hfunc.b1:.2f}x"
        else:
            eq_g += f" - {abs(hfunc.b1):.2f}x"
        if hfunc.b0 >= 0:
            eq_g += f" + {hfunc.b0:.2f}$"
        else:
            eq_g += f" - {abs(hfunc.b0):.2f}$"

        ax1.set_title(eq_f + eq_g + "\nHomotopy Curvature")

        set_legends()

        fig.canvas.draw_idle()

    def set_legends():
        ax1.set_xlabel(r"$s$")
        ax1.legend()

        ax2.set_xlabel(r"$s$")
        ax2.legend()

        ax3.set_xlabel(r"$\lambda$")
        ax3.set_xlim(1, 0)
        ax3.legend()

        ax4.set_xlabel(r"$\lambda$")
        ax4.set_xlim(1, 0)
        ax4.legend()

        fig.canvas.draw_idle()

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

    a4_slider.on_changed(recalc_and_plot)
    a3_slider.on_changed(recalc_and_plot)
    a2_slider.on_changed(recalc_and_plot)
    a1_slider.on_changed(recalc_and_plot)
    a0_slider.on_changed(recalc_and_plot)

    # Toggle visibility of each plot.
    def toggle_visibility(label, ax_num):
        ax_name = f"ax{ax_num}"
        visibility[ax_name][label] = not visibility[ax_name][label]
        if label in plot_objects[ax_name]:
            obj = plot_objects[ax_name][label]
            if isinstance(obj, list):
                for o in obj:
                    o.set_visible(visibility[ax_name][label])
            else:
                obj.set_visible(visibility[ax_name][label])

        set_legends()

        fig.canvas.draw_idle()

    checkbox_ax1 = plt.axes((0.82, 0.70, 0.07, 0.20))
    checkbox_ax2 = plt.axes((0.82, 0.55, 0.07, 0.10))
    checkbox_ax3 = plt.axes((0.82, 0.40, 0.07, 0.10))
    checkbox_ax4 = plt.axes((0.82, 0.20, 0.07, 0.15))

    for label in keys_ax1:
        visibility["ax1"][label] = True
    for label in keys_ax2:
        visibility["ax2"][label] = True
    for label in keys_ax3:
        visibility["ax3"][label] = True
    for label in keys_ax4:
        visibility["ax4"][label] = True

    check1 = CheckButtons(checkbox_ax1, keys_ax1, [True] * len(keys_ax1))
    check2 = CheckButtons(checkbox_ax2, keys_ax2, [True] * len(keys_ax2))
    check3 = CheckButtons(checkbox_ax3, keys_ax3, [True] * len(keys_ax3))
    check4 = CheckButtons(checkbox_ax4, keys_ax4, [True] * len(keys_ax4))

    check1.on_clicked(lambda label: toggle_visibility(label, 1))
    check2.on_clicked(lambda label: toggle_visibility(label, 2))
    check3.on_clicked(lambda label: toggle_visibility(label, 3))
    check4.on_clicked(lambda label: toggle_visibility(label, 4))

    # Toggle y-limit and y-scale of each subfigure.
    def toggle_ylim_scale(label, ax_num):
        axes = [ax1, ax2, ax3, ax4]
        ax = axes[ax_num - 1]

        if label == "Auto y-limits":
            # Toggle between auto and fixed y-limits
            if ax.get_autoscaley_on():
                ax.set_ylim(default_ylims[f"ax{ax_num}"])
                ax.set_autoscaley_on(False)
            else:
                ax.set_autoscaley_on(True)
                ax.relim()
                ax.autoscale_view(scaley=True, scalex=False)

        elif label == "Log scale":
            # Toggle between linear and log scale
            current_scale = ax.get_yscale()
            if current_scale == "linear":
                try:
                    ax.set_yscale("symlog")
                except ValueError:
                    # If log scale fails, keep it unchecked.
                    ylim_check = [ylim_check1, ylim_check2, ylim_check3, ylim_check4][
                        ax_num - 1
                    ]
                    ylim_check.set_active(1)  # Toggle log scale checkbox.
            else:
                ax.set_yscale("linear")

        fig.canvas.draw_idle()

    ylim_ax1 = plt.axes((0.92, 0.70, 0.07, 0.10))
    ylim_ax2 = plt.axes((0.92, 0.55, 0.07, 0.05))
    ylim_ax3 = plt.axes((0.92, 0.40, 0.07, 0.05))
    ylim_ax4 = plt.axes((0.92, 0.20, 0.07, 0.10))

    # Create checkbox widgets for y-limits and scale
    ylim_options = ["Auto y-limits", "Log scale"]
    ylim_check1 = CheckButtons(ylim_ax1, ylim_options, [True, False])
    ylim_check2 = CheckButtons(ylim_ax2, ylim_options, [True, False])
    ylim_check3 = CheckButtons(ylim_ax3, ylim_options, [True, False])
    ylim_check4 = CheckButtons(ylim_ax4, ylim_options, [True, False])

    default_ylims = {
        "ax1": (-10, 10),
        "ax2": (-10, 10),
        "ax3": (-10, 10),
        "ax4": (-10, 10),
    }

    ylim_check1.on_clicked(lambda label: toggle_ylim_scale(label, 1))
    ylim_check2.on_clicked(lambda label: toggle_ylim_scale(label, 2))
    ylim_check3.on_clicked(lambda label: toggle_ylim_scale(label, 3))
    ylim_check4.on_clicked(lambda label: toggle_ylim_scale(label, 4))

    # Reset f(x) coefficients to initial values.
    def reset(event):
        a4_slider.set_val(a4_init)
        a3_slider.set_val(a3_init)
        a2_slider.set_val(a2_init)
        a1_slider.set_val(a1_init)
        a0_slider.set_val(a0_init)

    reset_button_ax = plt.axes((0.65, 0.05, 0.08, 0.04))
    reset_button = plt.Button(reset_button_ax, "Reset")
    reset_button.on_clicked(reset)

    # Cycle through interesting polynomials f(x).
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
    recalc_and_plot(None)
    plt.show()


if __name__ == "__main__":
    interactive_homotopy_curvature_plot()
