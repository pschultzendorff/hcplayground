import jax.numpy as jnp
from hc import HC, Newton
from viz import plot_hc, plot_newton


def example_1() -> None:
    f = lambda x: x**3 - 5 * x**2 + 20
    g = lambda x: (x + 3) ** 2
    hc = HC(f, g, x=0.5, decay=0.7)
    converged: bool = hc.solve()
    if converged:
        plot_hc(f, g, hc.lambdas, hc.xx, hc.hxx, filename="example1_hc.png")


def example_2() -> None:
    f = lambda x: x**5 / 5 - 29 * x**3 / 3 + 100 * x + 200
    g = lambda x: x**5 / 5
    hc = HC(f, g, x=0.5, decay=0.8)
    converged: bool = hc.solve()
    xx_fine: jnp.ndarray = jnp.linspace(-7, 3, 1000)
    if converged:
        for i, (lambda_, xx, hxx) in enumerate(zip(hc.lambdas, hc.xx, hc.hxx)):
            h = lambda x: lambda_ * g(x) + (1 - lambda_) * f(x)
            plot_newton(
                h,
                xx,
                hxx,
                filename=f"example2_newton_{lambda_}.png",
                xx_fine=xx_fine,
                y_bottom=-50,
                y_top=100,
                f_x_label="$h_{" + f"{lambda_:.2f}" + "}(x)$",
                title=rf"Newton steps at $\lambda = {lambda_:.2f}$",
            )
            if i >= 10:
                break
    plot_newton(
        f,
        [0.5],
        [f(0.5)],
        filename="example2_f.png",
        xx_fine=xx_fine,
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
        filename="example2_g.png",
        xx_fine=xx_fine,
        y_bottom=-10,
        y_top=10,
        f_x_label="$g(x)$",
        newton_steps_label="initial guess",
        title=r"$g(x)  = \frac{x^5}{5}$",
    )


if __name__ == "__main__":
    example_2()
