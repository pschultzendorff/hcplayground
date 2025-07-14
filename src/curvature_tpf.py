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
import matplotlib.pyplot as plt
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


def hessian_tensor(f):
    """Return a function that computes the Hessian tensor of f at a point x."""

    def per_component_hessian(x, *args, **kwargs):
        return jnp.stack(
            [
                jax.hessian(lambda x_: f(x_, *args, **kwargs)[i])(x, *args, **kwargs)
                for i in range(f(x, *args, **kwargs).shape[0])
            ]
        )

    return per_component_hessian


def apply_hessian(H, u, v):
    """Apply the Hessian tensor H to vectors u and v."""
    return jnp.einsum("ijk,j,k->i", H, u, v)


class TPFModel:
    def __init__(self) -> None:
        self.num_cells = NUM_CELLS
        self.transmissibilities = TRANSMISSIBILITIES
        self.porosities = POROSITIES
        self.source_terms = SOURCE_TERMS
        self.neumann_bc = NEU_BC
        self.dirichlet_bc = DIR_BC
        self.initial_conditions = INITIAL_CONDITIONS
        self.mu_w = MU_W
        self.mu_n = MU_N

    def pc(self, s):
        """Capillary pressure function."""
        # return jnp.nan_to_num(0.1 * (1 - s) / s, nan=0.0)
        return s * 0.0

    def mobility_w(self, s):
        """Mobility of the wetting phase. Corey model."""
        k_w = jnp.where(s <= 1, jnp.where(s >= 0, s**2, 0.0), 1.0)  # type: ignore
        return k_w / self.mu_w  # type: ignore

    def mobility_n(self, s):
        """Mobility of the nonwetting phase. Corey model."""
        k_n = jnp.where(s >= 0, jnp.where(s <= 1, (1 - s) ** 2, 0.0), 1.0)  # type: ignore
        return k_n / self.mu_n  # type: ignore

    def compute_face_fluxes(self, p, s):
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
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        p_c = self.pc(s)

        # Calculate fluxes at each face. After adding Dirichlet bc, the indices work as
        # follows. Face i has cell i on the left and cell i + 1 on the right. The negative
        # pressure gradient across cell i is therefore given by p[i] - p[i + 1].

        # Phase potential upwinding of the mobilities. The mobilities for Neumann
        # boundaries are upwinded as well. This is not a problem, because the
        # transmissibilities are set to 0 at the Neumann boundary.
        m_w = jnp.zeros(self.num_cells + 1)
        m_n = jnp.zeros(self.num_cells + 1)

        for i in range(self.num_cells + 1):
            m_w = m_w.at[i].set(
                self.mobility_w(s[i]) if p[i] >= p[i + 1] else self.mobility_w(s[i + 1])
            )
            m_n = m_n.at[i].set(
                (
                    self.mobility_n(s[i])
                    if p[i] - p_c[i] >= p[i + 1] - p_c[i + 1]
                    else self.mobility_n(s[i + 1])
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

        F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        F_w = (
            F_t * (m_w / m_t)
            - self.transmissibilities[i] * m_w * m_n / m_t * p_c_gradient
        )

        # Add Neumann boundary conditions.
        F_t = F_t.at[0].add(self.neumann_bc[0, 0])
        F_w = F_w.at[0].add(self.neumann_bc[0, 1])
        F_t = F_t.at[-1].add(self.neumann_bc[1, 0])
        F_w = F_w.at[-1].add(self.neumann_bc[1, 1])

        return F_t, F_w

    def residual(self, x, dt, x_prev=None):
        """Compute the residual of the system at (x, beta)."""
        p = x[::2]
        s = x[1::2]

        if x_prev is None:
            s_prev = self.initial_conditions[1::2]
        else:
            s_prev = x_prev[1::2]

        # Compute fluxes.
        F_t, F_w = self.compute_face_fluxes(p, s)

        # Residuals for flow and transport equations.
        r_f = F_t[1:] - F_t[:-1] - self.source_terms[:, 0]
        r_t = (
            self.porosities * (s - s_prev) / dt
            + F_w[1:]
            - F_w[:-1]
            - self.source_terms[:, 1]
        )

        r = jnp.concatenate([r_f, r_t])
        return r

    def jacobian(self, x, dt, x_prev=None):
        """Compute the Jacobian of the system at (x, beta)."""
        J = jax.jacrev(self.residual, argnums=0)(x, dt, x_prev=x_prev)
        return J.reshape(-1, len(x))


def newton(model, x_init, x_prev, dt, max_iter=50, tol=1e-6):
    """Solve the system using Newton"s method."""
    x = x_init.copy()
    converged = False

    newton_progressbar = tqdm.trange(
        max_iter, desc="Newton iterations", position=0, leave=False
    )

    for i in newton_progressbar:
        r = model.residual(x, dt=dt, x_prev=x_prev)
        newton_progressbar.set_postfix({"residual_norm": jnp.linalg.norm(r)})

        if jnp.linalg.norm(r) < tol:
            converged = True
            break

        J = model.jacobian(x, dt=dt, x_prev=x_prev)
        dx = jnp.linalg.solve(J, -r)
        x += dx

    return x, converged


def hc(
    model,
    x_prev,
    x_init,
    dt,
    decay=0.1,
    max_iter=11,
    max_newton_iter=20,
    newton_tol=1e-6,
):
    """Solve the system using homotopy continuation with Newton"s method."""
    assert hasattr(model, "beta"), (
        "Model must have a beta attribute for homotopy continuation."
    )

    x = x_init.copy()
    model.beta = 1.0

    hc_progressbar = tqdm.trange(
        max_iter, desc="Homotopy continuation", position=1, leave=False
    )

    for i in hc_progressbar:
        hc_progressbar.set_postfix({r"$\lambda$": model.beta})

        # Previous solution for the predictor step. Newton's method for the corrector
        # step.
        x, converged = newton(
            model, x, x_prev, max_iter=max_newton_iter, tol=newton_tol, dt=dt
        )

        if not converged:
            break

        # Store data for the homotopy curve BEFORE updating beta.
        model.store_curve_data(x, dt, x_prev=x_prev)
        # Update the homotopy parameter beta only now.
        model.beta -= decay

    return x, converged


def solve(model, final_time, n_time_steps):
    """Solve the two-phase flow problem over a given time period."""
    dt = final_time / n_time_steps
    solutions = [model.initial_conditions]

    time_progressbar_position = 2 if hasattr(model, "beta") else 1
    time_progressbar = tqdm.trange(
        n_time_steps, desc="Time steps", position=time_progressbar_position, leave=True
    )

    solver = newton if not hasattr(model, "beta") else hc

    for i in time_progressbar:
        time_progressbar.set_postfix({"time_step": i + 1})

        # Solve the nonlinear problem at the current time step.
        x_prev = solutions[-1]
        # Use the previous time step solution as the initial guess for the solver.
        (
            x_next,
            converged,
        ) = solver(model, x_prev, x_prev, dt=dt)

        if converged:
            solutions.append(x_next)

        else:
            break

    return solutions


class HCModel(TPFModel):
    """Base class for homotopy continuation models."""

    def __init__(self):
        super().__init__()
        self.beta = (
            1.0  # Homotopy parameter, 0 for simpler system, 1 for actual system.
        )
        self.betas = [self.beta]  # Store beta values for the homotopy curve.
        self.tangents = []  # Store tangent vectors of the homotopy curve.
        self.curvatures = []  # Store curvature values of the homotopy curve.

    def h_beta_deriv(self, x, dt, x_prev=None):
        """Compute the derivative of the homotopy problem with respect to beta."""
        beta_save = self.beta

        self.beta = 1.0
        r_g = self.residual(x, dt, x_prev=x_prev)
        self.beta = 0.0
        r_f = self.residual(x, dt, x_prev=x_prev)

        self.beta = beta_save

        return r_g - r_f

    def curve_tangent(self, x, dt, x_prev=None, jac=None):
        """Compute the tangent vector of the homotopy curve at (x, beta)."""
        b = self.h_beta_deriv(x, dt, x_prev=x_prev)
        A = self.jacobian(x, dt, x_prev=x_prev) if jac is None else jac
        tangent = jnp.linalg.solve(A, b)
        return tangent

    def curve_curvature(self, x, dt, x_prev=None, jac=None):
        if jac is None:
            jac = self.jacobian(x, dt, x_prev=x_prev)
        tangent = self.curve_tangent(x, dt, x_prev=x_prev, jac=jac)
        # Compute the Hessian of the residual.
        H = hessian_tensor(self.residual)(x, dt, x_prev=x_prev)
        w2 = apply_hessian(H, tangent, tangent)
        curvature_vector = jnp.linalg.solve(jac, -w2)
        return jnp.linalg.norm(curvature_vector)

    def store_curve_data(self, x, dt, x_prev=None):
        """Store the beta, tangent, and curvature data for the homotopy curve."""
        self.betas.append(self.beta)
        self.tangents.append(self.curve_tangent(x, dt, x_prev=x_prev))
        self.curvatures.append(self.curve_curvature(x, dt, x_prev=x_prev))


class FluxHC(HCModel):
    """Flux homotopy continuation for the two-phase flow problem."""

    def mobility_w(self, s):
        """Mobility of the wetting phase."""
        k_w = self.beta * s + (1 - self.beta) * jnp.where(
            s <= 1,
            jnp.where(s >= 0, s**2, 0.0),  # type: ignore
            1.0,
        )
        return k_w / MU_W

    def mobility_n(self, s):
        """Mobility of the nonwetting phase."""
        k_n = self.beta * (1 - s) + (1 - self.beta) * jnp.where(
            s >= 0,
            jnp.where(s <= 1, (1 - s) ** 2, 0.0),  # type: ignore
            1.0,
        )
        return k_n / MU_N


class DiffusionHC(HCModel):
    r"""Vanishing diffusion homotopy continuation for the two-phase flow problem.

    Cf. Jiang, J. and Tchelepi, H.A. (2018) ‘Dissipation-based continuation method for
    multiphase flow in heterogeneous porous media’, Journal of Computational Physics,
    375, pp. 307–336. Available at: https://doi.org/10.1016/j.jcp.2018.08.044.

    We omit the additional control parameter :math:`\beta` in the paper (the
    continuation parameter :math:`beta` is called :math:`\kappa` in the paper).

    """

    def compute_face_fluxes(self, p, s):
        """Compute total and wetting phase fluxes at cell faces with vanishing diffusion."""
        # Base model fluxes.
        F_t, F_w = super().compute_face_fluxes(p, s)

        # Compute vanishing dissipation.
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])
        s_w_gradient = s[:-1] - s[1:]  # Negative pressure gradient across cells.
        F_w += self.beta * self.transmissibilities * s_w_gradient

        return F_t, F_w


def plot_solution(solutions):
    """Plot the solution of the two-phase flow problem with an interactive time slider."""
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


def plot_curvature(betas, tangents, curvatures, fig=None, ax=None):
    """Plot the curvature of the homotopy curve."""

    # Convert solutions to numpy array for easier manipulation.
    betas_array = jnp.array(betas)
    tangents_array = jnp.array(tangents)
    curvatures_array = jnp.array(curvatures)

    # Create figure and axis.
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curvature over time steps.
    ax.plot(betas_array, curvatures_array, marker="o", label="Curvature")
    ax.set_xlabel(r"\lambda")
    ax.set_ylabel(r"\kappa")
    ax.set_title("Homotopy Curve Curvature Over Time Steps")
    ax.grid(True)
    ax.legend()

    plt.show()

    return fig, ax


# solutions = solve(TPFModel(), final_time=1.0, n_time_steps=10)
# plot_solution(solutions)


model_fluxhc = FluxHC()
model_diffhc = DiffusionHC()
solve(model_fluxhc, final_time=1.0, n_time_steps=1)
solve(model_diffhc, final_time=1.0, n_time_steps=1)
fig, ax = plot_curvature(
    model_fluxhc.betas, model_fluxhc.tangents, model_fluxhc.curvatures
)
plot_curvature(
    model_diffhc.betas, model_diffhc.tangents, model_diffhc.curvatures, fig=fig, ax=ax
)
# solutions = solve(DiffusionHC(), final_time=1.0, n_time_steps=10)
# plot_solution(solutions)
