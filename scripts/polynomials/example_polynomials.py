import pathlib
import sys

import jax.numpy as jnp

src_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "src" / "polynomials"
sys.path.append(str(src_dir))

from solvers import hc_solver
from viz import plot_hc, plot_newton

results_dir = pathlib.Path(__file__).parent / "results" / "polynomials"
results_dir.mkdir(parents=True, exist_ok=True)


def example_1() -> None:
    def f(x):
        return x**3 - 5 * x**2 + 20

    def g(x):
        return (x + 3) ** 2

    converged, betas, xss, yss = hc_solver(f, g, x0=0.5, decay=0.7)
    if converged:
        plot_hc(f, g, betas, xss, yss, filename=results_dir / "example1_hc.png")


def example_2() -> None:
    def f(x):
        return x**5 / 5 - 29 * x**3 / 3 + 100 * x + 200

    def g(x):
        return x**5 / 5

    converged, betas, xss, yss = hc_solver(f, g, x0=0.5, decay=0.8)
    xs_fine = jnp.linspace(-7, 3, 1000)
    if converged:
        for i, (beta, xs, ys) in enumerate(zip(betas, xss, yss)):

            def h(x):
                return beta * g(x) + (1 - beta) * f(x)

            plot_newton(
                h,
                xs,
                ys,
                filename=results_dir / f"example2_newton_{beta:.2f}.png",
                xs_fine=xs_fine,
                y_bottom=-50,
                y_top=100,
                f_x_label="$h_{" + f"{beta:.2f}" + "}(x)$",
                title=rf"Newton steps at $\lambda = {beta:.2f}$",
            )
            if i >= 10:
                break
    plot_newton(
        f,
        [0.5],
        [f(0.5)],
        filename=results_dir / "example2_f.png",
        xs_fine=xs_fine,
        y_bottom=-50,
        y_top=300,
        f_x_label="$f(x)$",
        newton_steps_label="initial guess",
        title=r"$f(x) = \frac{x^5}{5} - \frac{29 x^3}{3} + 100 x + 200$",
    )
    plot_newton(
        g,
        [0.5],
        [g(0.5)],
        filename=results_dir / "example2_g.png",
        xs_fine=xs_fine,
        y_bottom=-10,
        y_top=10,
        f_x_label="$g(x)$",
        newton_steps_label="initial guess",
        title=r"$g(x)  = \frac{x^5}{5}$",
    )


example_1()
example_2()
