"""Newton and homotopy continuation solver for two-phase flow problems."""

import logging

import jax.numpy as jnp
import tqdm
from hc import DiffusionHC, HCModel
from tqdm.contrib.logging import logging_redirect_tqdm

from hcplayground.src.two_phase_flow.model_tpf import TPFModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def newton(
    model: TPFModel,
    x_init: jnp.ndarray,
    x_prev: jnp.ndarray,
    dt: float,
    max_iter: int = 100,
    tol: float = 1e-10,
    appleyard: bool = False,
) -> tuple[jnp.ndarray, bool]:
    """Solve the system using Newton"s method."""
    x = x_init.copy()
    converged = False

    newton_progressbar = tqdm.trange(
        max_iter, desc="Newton iterations", position=0, leave=False
    )

    with logging_redirect_tqdm([logger]):
        for i in newton_progressbar:
            r = model.residual(x, dt=dt, x_prev=x_prev)
            newton_progressbar.set_postfix(
                {"residual_norm": jnp.linalg.norm(r) / jnp.sqrt(r.size)}
            )

            if jnp.linalg.norm(r) / jnp.sqrt(r.size) < tol:
                converged = True
                break

            J = model.jacobian(x, dt=dt, x_prev=x_prev)
            model.linear_system = (J, r)

            dx = jnp.linalg.solve(J, -r)
            if jnp.isnan(dx).any() or jnp.isinf(dx).any():
                # print(
                #     jax.make_jaxpr(jax.jacrev(model.residual, argnums=0))(
                #         x, dt, x_prev=x_prev
                #     )
                # )
                raise ValueError("Newton step resulted in NaN or Inf values.")

            # Limit cellwise saturation updates to [-0.2, 0.2].
            if appleyard:
                dp = dx[::2]
                ds = jnp.clip(dx[1::2], -0.2, 0.2)
                dx = jnp.stack([dp, ds], axis=1).flatten()

            x += dx

            # Ensure physical saturation values plus small epsilon to avoid numerical issues.
            p = x[::2]
            s = jnp.clip(x[1::2], 1e-15, 1 - 1e-15)
            x = jnp.stack([p, s], axis=1).flatten()

    return x, converged


def hc(
    model: HCModel,
    x_prev: jnp.ndarray,
    x_init: jnp.ndarray,
    dt: float,
    hc_decay: float = 1 / 100,
    hc_max_iter: int = 101,
    **kwargs,
):
    """Solve the system using homotopy continuation with Newton"s method.

    Parameters:
        model: The model to solve.
        x_prev: Previous solution, used as the initial guess for Newton's method.
        x_init: Initial guess for the homotopy continuation.
        dt: Time step size.
        hc_decay: Linear decay for the homotopy parameter :math:`beta`.
        hc_max_iter: Maximum number of homotopy continuation iterations.
        **kwargs: Additional keyword arguments for the Newton solver.

    Raises:
        ValueError: If the Newton step results in NaN or Inf values.

    """
    assert hasattr(model, "beta"), (
        "Model must have a beta attribute for homotopy continuation."
    )

    x = x_init.copy()
    model.beta = 1.0

    hc_progressbar = tqdm.trange(
        hc_max_iter, desc="Homotopy continuation", position=1, leave=False
    )

    with logging_redirect_tqdm([logger]):
        for i in hc_progressbar:
            hc_progressbar.set_postfix({r"$\lambda$": model.beta})

            # Previous solution for the predictor step. Newton's method for the
            # corrector step.
            try:
                x, converged = newton(
                    model,
                    x,
                    x_prev,
                    dt,
                    **kwargs,
                )
            except ValueError as _:
                converged = False

            if converged:
                # Store data for the homotopy curve BEFORE updating beta.
                model.store_curve_data(x, dt, x_prev=x_prev)
                model.store_intermediate_solutions(x)

                # Update the homotopy parameter beta only now.
                model.beta -= hc_decay
                # For convenience, ensure beta is non-negative and equal to zero at the
                # end of the loop.
                if abs(model.beta) < 1e-3 or model.beta < 0:
                    model.beta = 0.0
            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at continuation"
                    + f" step {i + 1}, lambda={model.beta}."
                )
                break

    return x, converged


def solve(model: TPFModel, final_time: float, n_time_steps: int, **kwargs):
    """Solve the two-phase flow problem over a given time period."""
    dt = final_time / n_time_steps
    solutions: list[jnp.ndarray] = [model.initial_conditions]

    time_progressbar_position = 2 if hasattr(model, "beta") else 1
    time_progressbar = tqdm.trange(
        n_time_steps, desc="Time steps", position=time_progressbar_position, leave=False
    )

    solver = hc if isinstance(model, HCModel) else newton

    with logging_redirect_tqdm([logger]):
        for i in time_progressbar:
            time_progressbar.set_postfix({"time_step": i + 1})

            # Previous solution is the initial guess for the solver.
            x_prev = solutions[-1]

            # For diffusion based HC, update the diffusion coefficient.
            if isinstance(model, DiffusionHC):
                model.update_adaptive_diffusion_coeff(dt, x_prev)
                logger.info(
                    f"Model {model.__class__.__name__} updated adaptive diffusion"
                    + f" cofficient.\n New values: {model.adaptive_diffusion_coeff}."
                )

            try:
                x_next, converged = solver(model, x_prev, x_prev, dt=dt, **kwargs)  # type: ignore
            except ValueError as _:
                converged = False

            if converged:
                solutions.append(x_next)
            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at time step {i + 1}."
                )
                break

    return solutions, converged
