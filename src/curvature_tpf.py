"""Curvature for a 1D two-cell two-phase flow problem solved with HC.


The example geometry looks like this:

          |-------------------|-------------------|
          |                   |                   |
Neu bc    F0                  F1                  F2      Dir bc
       ------>    p0,s0    ------>    p1,s1    ------>    p2,s2
          |                   |                   |
          |-------------------|-------------------|
          e0        c0        e1        c1        e2

The two-phase flow problem, formulated in the fractional flow formulation, is solved for
nonwetting pressure and wetting saturation.

The problem is discretized with implicit Euler in time and cell-centered finite volumes
in space. The face fluxes are evaluated with two-point flux approximation (where the
transmissibilities are fixed beforehand) and phase-potential upwinding. The resulting
system of nonlinear equations at each time step is solved with a homotopy continuation
method, where the corrector step uses Newton"s method.


"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy as sp
import tqdm
from matplotlib.widgets import Slider

NUM_CELLS = 2

TRANSMISSIBILITIES = jnp.array(
    [0.0, 1.0, 2.0]
)  # Transmissibilities for faces e0, e1, e2.
# Transmissibility at the Dirichlet boundary has to be doubled.
# Transmissibility at the Neumann boundary hast to be set to 0.

POROSITIES = jnp.array([0.5, 0.5])  # Porosities for cells c0, c1.
SOURCE_TERMS = jnp.array(
    [[0.0, 0.0], [0.0, 0.0]]
)  # Source terms for cells c0, c1. Total and wetting source.

NEU_BC = jnp.array(
    [
        [1.0, 0.9],
        [0.0, 0.0],
    ]
)  # Neumann boundary conditions for faces e0, e2. Total flux and wetting flux.
DIR_BC = jnp.array(
    [
        [0.0, 0.0],
        [3.0, 0.5],
    ]
)  # Dirichlet boundary conditions for faces e0, e2. Pressure and saturation.

INITIAL_CONDITIONS = jnp.array(
    [0.0, 0.5, 0.0, 0.5]
)  # Initial conditions for [p0, s0, p1, s1].


MU_W = 1.0  # Viscosity of the wetting phase.
MU_N = 1.0  # Viscosity of the nonwetting phase.

# Homotopy parameter, 0 for simpler system, 1 for actual system. For simplicity, we use
# a global variable.
beta = 0.0


def pc(s):
    """Capillary pressure function."""
    # return jnp.nan_to_num(0.1 * (1 - s) / s, nan=0.0)
    return s * 0.0


def mobility_w(s):
    """Mobility of the wetting phase. Corey model."""
    k_w = jnp.where(s <= 1, jnp.where(s >= 0, s**2, 0.0), 1.0)
    return k_w / MU_W  # type: ignore


def mobility_n(s):
    """Mobility of the nonwetting phase. Corey model."""
    k_n = jnp.where(s >= 0, jnp.where(s <= 1, (1 - s) ** 2, 0.0), 1.0)
    return k_n / MU_N  # type: ignore


# def mobility_w(s):
#     """Mobility of the wetting phase."""
#     k_w = beta * s + (1 - beta) * jnp.where(s >= 0, s**2, 0.0)
#     return k_w / MU_W


# def mobility_n(s):
#     """Mobility of the nonwetting phase."""
#     k_n = beta * (1 - s) + (1 - beta) * jnp.where(s <= 1, (1 - s) ** 2, 0.0)
#     return k_n / MU_N


def compute_face_fluxes(p, s):
    """Compute total and wetting phase fluxes at cell faces.

    Note: The code is written general to handle different boundary conditions and source
    terms.

    Args:
        p: Nonwetting pressures at cell centers [p0, p1].
        s: Wetting saturations at cell centers [s0, s1].

    Returns:
        F_t: Total fluxes at faces [F0, F1, F2].
        F_w: Wetting phase fluxes at faces [F0, F1, F2].

    """
    # Add Dirichlet boundary conditions.
    p = jnp.concatenate([DIR_BC[0, :1], p, DIR_BC[1, :1]])
    s = jnp.concatenate([DIR_BC[0, 1:], s, DIR_BC[1, 1:]])

    p_c = pc(s)

    # Calculate fluxes at each face. After adding Dirichlet bc, the indices work as
    # follows. Face i has cell i on the left and cell i + 1 on the right. The negative
    # pressure gradient across cell i is therefore given by p[i] - p[i + 1].

    # Phase potential upwinding of the mobilities. The mobilities for Neumann
    # boundaries are upwinded as well. This is not a problem, because the
    # transmissibilities are set to 0 at the Neumann boundary.
    m_w = jnp.zeros(NUM_CELLS + 1)
    m_n = jnp.zeros(NUM_CELLS + 1)

    for i in range(NUM_CELLS + 1):
        m_w = m_w.at[i].set(
            mobility_w(s[i]) if p[i] >= p[i + 1] else mobility_w(s[i + 1])
        )
        m_n = m_n.at[i].set(
            (
                mobility_n(s[i])
                if p[i] - p_c[i] >= p[i + 1] - p_c[i + 1]
                else mobility_n(s[i + 1])
            )
        )

    # Handle NaN values in mobilities.
    m_n = jnp.nan_to_num(m_n, nan=0.0)
    m_w = jnp.nan_to_num(m_w, nan=0.0)

    m_t = m_n + m_w

    # TPFA fluxes at the faces.
    p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
    p_c_gradient = (
        p_c[:-1] - p_c[:-1]
    )  # Negative capillary pressure gradient across cells.

    F_t = TRANSMISSIBILITIES * (m_t * p_n_gradient - m_w * p_c_gradient)
    F_w = F_t * (m_w / m_t) - TRANSMISSIBILITIES[i] * m_w * m_n / m_t * p_c_gradient

    # Add Neumann boundary conditions.
    F_t = F_t.at[0].add(NEU_BC[0, 0])
    F_w = F_w.at[0].add(NEU_BC[0, 1])
    F_t = F_t.at[-1].add(NEU_BC[1, 0])
    F_w = F_w.at[-1].add(NEU_BC[1, 1])

    return F_t, F_w


def residual(x, dt, x_prev=None):
    """Compute the residual of the system at (x, beta)."""
    p = x[::2]
    s = x[1::2]

    if x_prev is None:
        s_prev = INITIAL_CONDITIONS[1::2]
    else:
        s_prev = x_prev[1::2]

    # Compute fluxes.
    F_t, F_w = compute_face_fluxes(p, s)

    # Residuals for flow and transport equations.
    r_f = F_t[1:] - F_t[:-1] - SOURCE_TERMS[:, 0]
    r_t = POROSITIES * (s - s_prev) / dt + F_w[1:] - F_w[:-1] - SOURCE_TERMS[:, 1]

    r = jnp.concatenate([r_f, r_t])
    return r


def jacobian(x, dt):
    """Compute the Jacobian of the system at (x, beta)."""
    J = jax.jacrev(residual, argnums=0)(x, dt)
    return J.reshape(-1, len(x))


def newton(x_init, dt, max_iter=50, tol=1e-6):
    """Solve the system using Newton"s method."""
    x = x_init.copy()
    converged = False

    newton_progressbar = tqdm.trange(
        max_iter, desc="Newton iterations", position=0, leave=False
    )

    for i in newton_progressbar:
        r = residual(x, dt=dt, x_prev=x_init)
        newton_progressbar.set_postfix({"residual_norm": jnp.linalg.norm(r)})

        if jnp.linalg.norm(r) < tol:
            converged = True
            break

        J = jacobian(x, dt=dt)
        dx = jnp.linalg.solve(J, -r)
        x += dx

    return x, converged


def hc(x_init, dt, decay=0.9, max_iter=10, max_newton_iter=20, newton_tol=1e-6):
    """Solve the system using homotopy continuation with Newton"s method."""
    x = x_init.copy()

    hc_progressbar = tqdm.trange(
        max_iter, desc="Homotopy continuation", position=1, leave=False
    )

    for i in hc_progressbar:
        global beta
        hc_progressbar.set_postfix({r"$\lambda$": beta})

        # Previous solution for the predictor step. Newton's method for the corrector
        # step.
        x, converged = newton(x, max_iter=max_newton_iter, tol=newton_tol, dt=dt)

        if not converged:
            break

        beta *= decay

    return x, converged


def solve(initial_conditions, final_time, n_time_steps):
    """Solve the two-phase flow problem over a given time period."""
    dt = final_time / n_time_steps
    solutions = [initial_conditions]

    time_progressbar = tqdm.trange(
        n_time_steps, desc="Time steps", position=2, leave=True
    )
    for i in time_progressbar:
        time_progressbar.set_postfix({"time_step": i + 1})

        # Solve the nonlinear problem at the current time step.
        x_prev = solutions[-1]
        x_next, converged = newton(x_prev, dt=dt)

        if converged:
            solutions.append(x_next)
        else:
            break

    return solutions


def plot_solution(solutions):
    """Plot the solution of the two-phase flow problem with an interactive time slider."""
    import matplotlib.pyplot as plt

    # Convert solutions to numpy array for easier manipulation.
    solutions_array = jnp.array(solutions)
    n_time_steps = len(solutions)

    # Create figure and axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()  # Create a second y-axis for the saturation.
    plt.subplots_adjust(bottom=0.25)  # Make room for slider.

    # Cell centers for plotting.
    xx = jnp.linspace(0, 1, NUM_CELLS)

    # Initial plot with first time step.
    pressures = solutions_array[0, ::2]
    saturations = solutions_array[0, 1::2]

    (p_line,) = ax.plot(xx, pressures, "o-", color="tab:blue", label=r"$p_n$")
    (s_line,) = ax2.plot(xx, saturations, "v-", color="tab:orange", label=r"$s_w$")

    # Find min and max pressure values for consistent y-axis.
    p_min = min(solutions_array[:, ::2].min(), 0)  # type: ignore
    p_max = solutions_array[:, :].max() * 1.1
    ax.set_ylim(p_min, p_max)  # type: ignore

    ax2.set_ylim(0, 1)  # Saturation is between 0 and 1.

    ax.set_xlabel("x")
    ax.set_ylabel(r"Pressure $p_n$", color="tab:blue")
    ax2.set_ylabel(r"Wetting Saturation $s_w$", color="tab:orange")

    ax.set_title("Two-Phase Flow Solution (Time Step: 0)")
    ax.legend()
    ax2.legend()
    ax.grid(True)

    # Add slider
    ax_slider = plt.axes((0.15, 0.1, 0.7, 0.03))
    time_slider = Slider(
        ax=ax_slider,
        label="Time Step",
        valmin=0,
        valmax=n_time_steps,
        valinit=0,
        valstep=1,
    )

    # Update function for slider
    def update(val):
        time_idx = int(time_slider.val)
        p_line.set_ydata(solutions_array[time_idx, ::2])
        s_line.set_ydata(solutions_array[time_idx, 1::2])
        ax.set_title(f"Two-Phase Flow Solution (Time Step: {time_idx})")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


solutions = solve(INITIAL_CONDITIONS, final_time=1.0, n_time_steps=10)
plot_solution(solutions)


def flux_homotopy():
    pass


def dissipation_homotopy():
    pass


# def h_beta_deriv(x, x_prev=None):
#     """Compute the derivative of the homotopy problem with respect to beta."""
#     global beta
#     beta_save = beta

#     beta = 1.0
#     r_g = residual(x, x_prev=x_prev)
#     beta = 0.0
#     r_f = residual(x, x_prev=x_prev)

#     beta = beta_save
#     return r_g - r_f


# def curve_tangent(x, x_prev=None, jac=None):
#     """Compute the tangent vector of the homotopy curve at (x, beta)."""
#     b = h_beta_deriv(x, x_prev=x_prev)
#     A = jacobian(x, x_prev=x_prev) if jac is None else jac
#     tangent = jnp.linalg.solve(A, b)
#     return tangent


# def hessian(x, x_prev=None):
#     """Compute the Hessian of the homotopy problem at (x, beta)."""
#     n = len(x)

#     def hessian_tensor(v1, v2):
#         """Compute the Hessian tensor of the homotopy problem."""
#         hessian_matrix = jnp.zeros((n, n))
#         for i in range(n):
#             e_i = jnp.zeros(n)
#             e_i[i] = 1.0
#             hessian_matrix[:, i] = curve_tangent(x + e_i, x_prev=x_prev)

#         return hessian_matrix

#     return hessian_tensor


# def curve_curvature(x, x_prev=None, jac=None):
#     if jac is None:
#         jac = jacobian(x, x_prev=x_prev)
#     tangent = curve_tangent(x, x_prev=x_prev, jac=jac)
#     w2 = hessian(x, x_prev=x_prev)(tangent, tangent)
#     curvature_vector = jnp.linalg.solve(jac, -w2)
#     return jnp.linalg.norm(curvature_vector)
