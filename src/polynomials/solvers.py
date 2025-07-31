import logging
from typing import Callable

import jax
import jax.numpy as jnp
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def newton(
    f: Callable[[float], float],
    x0: float = 0.0,
    tol: float = 1e-3,
    max_iter: int = 30,
) -> tuple[bool, jnp.ndarray, jnp.ndarray]:
    """Run Newton's method to find a root of a 1D function.

    Parameters:
        f: The function for which to find the roots.
        x0: The initial guess. Defaults to 0.0.
        tol: Residual tolerance to determine convergence. Defaults to 1e-3.
        max_iter: Maximum number of iterations. Defaults to 30.

    """
    x: float = x0
    y: float = f(x)
    xs: list[float] = [x]
    ys: list[float] = [y]
    f_grad = jax.grad(f)
    iter: int = 0

    def step() -> None:
        nonlocal x, y, iter
        x = x - y / f_grad(x)
        y = f(x)
        xs.append(x)
        ys.append(y)
        iter += 1

    with logging_redirect_tqdm([logging.root]):
        with tqdm(
            total=max_iter, desc="Newton method", unit="step", position=1, leave=False
        ) as pbar:
            while jnp.abs(y) > tol and iter < max_iter:
                step()
                pbar.set_postfix(
                    {
                        "x": x,
                        "f(x)": y,
                        "Delta x": jnp.abs(x - xs[-2]),
                    }
                )
                pbar.update(1)
    if jnp.abs(y) <= tol:
        logger.info(f"Newton converged to solution x={x} in {iter} steps.")
        return True, jnp.array(xs), jnp.array(ys)
    else:
        logger.info(
            f"Newton did not converge to solution after {max_iter} steps."
            + f" Last residual is {y}."
        )
        return False, jnp.array(xs), jnp.array(ys)


def hc_solver(
    f: Callable[[float], float],
    g: Callable[[float], float],
    x0: float = 0.0,
    decay: float = 0.9,
    max_iter: int = 30,
    newton_tol: float = 1e-3,
    newton_max_iter: int = 30,
) -> tuple[bool, jnp.ndarray, list, list]:
    x: float = x0
    beta: float = 1.0

    betas: list[float] = [beta]
    xss: list[jnp.ndarray] = []
    yss: list[jnp.ndarray] = []

    iter: int = 0
    converged: bool = False
    diverged: bool = False

    def h(x: float) -> float:
        nonlocal beta
        return beta * g(x) + (1 - beta) * f(x)

    def step() -> None:
        nonlocal beta, x, iter, diverged
        newton_converged, xs, ys = newton(
            h, x, tol=newton_tol, max_iter=newton_max_iter
        )
        x = xs[-1].item()
        beta *= decay
        betas.append(beta)
        xss.append(xs)
        yss.append(ys)

        diverged = not newton_converged
        iter += 1

    with logging_redirect_tqdm([logging.root]):
        with tqdm(
            total=beta,
            desc="Homotopy continuation",
            unit="step",
            position=0,
        ) as pbar:
            while iter < max_iter:
                step()
                if diverged:
                    logger.info(
                        "Stopping homotopy continuation due to non-convergence."
                    )
                    break

                pbar.set_postfix({"x": x, "beta": beta})
                pbar.update(betas[-2] - beta)
    if not diverged:
        converged = True
        logger.info(f"Homotopy continuation converged to solution x={x}")
    return converged, jnp.array(betas), xss, yss
