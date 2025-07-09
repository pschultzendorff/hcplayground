import pathlib
from typing import Callable, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

sns.set_style("whitegrid")
# plt.rcParams['text.usetex'] = True


def plot_newton(
    f: Callable,
    xx: list[float] | jnp.ndarray,
    fxx: list[float] | jnp.ndarray,
    filename: pathlib.Path | str = "Newton.png",
    xx_fine: Optional[jnp.ndarray] = None,
    **kwargs
) -> None:
    fig, ax = plt.subplots()
    xx = jnp.array(xx)
    fxx = jnp.array(fxx)

    if xx_fine is None:
        xx_fine = jnp.arange(xx.min(), xx.max(), 0.01)
    fxx_fine: jnp.ndarray = f(xx_fine)

    ax.plot(
        xx,
        fxx,
        label=kwargs.get("newton_steps_label", "Newton steps"),
        marker="s",
        color="red",
    )
    ax.plot(
        xx_fine, fxx_fine, label=kwargs.get("f_x_label", "$f(x)$"), color="royalblue"
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
    f,
    g,
    lambdas: list[float],
    xxx: list[list[float]],
    hxxx: list[list[float]],
    filename: pathlib.Path | str = "hc.png",
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Calculate exact h(x) for fine grid for each lambda.
    lambdas_fine: jnp.ndarray = jnp.arange(1, 0, -0.05)
    xx_fine: jnp.ndarray = jnp.linspace(
        min([min(xx) for xx in xxx]), max([max(xx) for xx in xxx]), 50
    )
    hxx_fine: list[jnp.ndarray] | jnp.ndarray = []

    # Plot surface of h..
    for lambda_ in lambdas_fine:
        h = lambda x: lambda_ * g(x) + (1 - lambda_) * f(x)
        hxx_fine.append(h(xx_fine))
    hxx_fine = jnp.array(hxx_fine)

    xx_fine, lambdas_fine = jnp.meshgrid(xx_fine, lambdas_fine)

    ax.plot_surface(
        lambdas_fine,
        xx_fine,
        hxx_fine,
        edgecolor="royalblue",
        lw=0.5,
        rstride=8,
        cstride=8,
        alpha=0.3,
    )

    # Plot solution curve for each Newton step.
    for lambda_, xx, hxx in zip(lambdas, xxx, hxxx):
        ax.plot(
            [lambda_] * len(xx),
            xx,
            hxx,
        )

    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("lambda")
    ax.set_ylabel("x")
    ax.legend()
    fig.savefig(filename)
