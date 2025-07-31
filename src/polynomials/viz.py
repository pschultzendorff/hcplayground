import pathlib
from typing import Callable, Optional, TypeAlias

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

NumericInput: TypeAlias = int | float | np.ndarray | jnp.ndarray


def plot_newton(
    f: Callable[[NumericInput], NumericInput],
    xs: list[float] | jnp.ndarray,
    ys: list[float] | jnp.ndarray,
    filename: pathlib.Path | str = "Newton.png",
    xs_fine: Optional[jnp.ndarray] = None,
    **kwargs,
) -> None:
    fig, ax = plt.subplots()
    xs = jnp.array(xs)
    ys = jnp.array(ys)

    if xs_fine is None:
        xs_fine = jnp.arange(xs.min(), xs.max(), 0.01)
    ys_fine: jnp.ndarray = f(xs_fine)  # type: ignore

    ax.plot(
        xs,
        ys,
        label=kwargs.get("newton_steps_label", "Newton steps"),
        marker="s",
        color="red",
    )
    ax.plot(
        xs_fine, ys_fine, label=kwargs.get("f_x_label", "$f(x)$"), color="royalblue"
    )
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    if "y_bottom" in kwargs:
        ax.set_ylim(bottom=kwargs["y_bottom"])
    if "y_top" in kwargs:
        ax.set_ylim(top=kwargs["y_top"])
    ax.legend()
    ax.set_title(kwargs.get("title", "Newton's method"))
    fig.savefig(filename)


def plot_hc(
    f: Callable[[NumericInput], NumericInput],
    g: Callable[[NumericInput], NumericInput],
    betas: list[float] | jnp.ndarray,
    xss: list | jnp.ndarray,
    yss: list | jnp.ndarray,
    filename: pathlib.Path | str = "hc.png",
    **kwargs,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Calculate exact h(x) for fine grid for each beta.
    betas_fine: jnp.ndarray = jnp.arange(1, 0, -0.05)
    xs_fine: jnp.ndarray = jnp.linspace(
        min([min(xx) for xx in xss]), max([max(xx) for xx in xss]), 50
    )
    yss_list: list[jnp.ndarray] = []

    # Plot surface of h..
    for beta in betas_fine:

        def h(x: NumericInput) -> NumericInput:
            return beta * g(x) + (1 - beta) * f(x)

        yss_list.append(h(xs_fine))  # type: ignore
    yss_fine: jnp.ndarray = jnp.array(yss_list)

    xs_fine, betas_fine = jnp.meshgrid(xs_fine, betas_fine)

    ax.plot_surface(  # type: ignore
        betas_fine,
        xs_fine,
        yss_fine,
        edgecolor="royalblue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )

    # Plot solution curve for each Newton step.
    for beta, xs, ys in zip(betas, xss, yss):
        ax.plot(
            [beta] * len(xs),
            xs,
            ys,
        )

    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("$x$")
    if "y_bottom" in kwargs:
        ax.set_ylim(bottom=kwargs["y_bottom"])
    if "y_top" in kwargs:
        ax.set_ylim(top=kwargs["y_top"])
    ax.legend()
    ax.set_title(kwargs.get("title", "Homotopy continuation method"))
    fig.savefig(filename)
