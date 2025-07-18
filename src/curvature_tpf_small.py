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

import logging
import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import viztracer
from matplotlib.widgets import Slider
from tqdm.contrib.logging import logging_redirect_tqdm

dirname = pathlib.Path(__file__).parent
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

key = jax.random.PRNGKey(0)  # Reproducability.

NUM_CELLS = 2

RAND = jax.random.uniform(key, shape=(NUM_CELLS + 1,), minval=-1.0, maxval=2.0)

TRANSMISSIBILITIES = jnp.pow(10, RAND)
TRANSMISSIBILITIES = TRANSMISSIBILITIES.at[0].set(0.0)
TRANSMISSIBILITIES = TRANSMISSIBILITIES.at[-1].set(TRANSMISSIBILITIES[-1] * 2)
# Transmissibility at the Dirichlet boundary has to be doubled.
# Transmissibility at the Neumann boundary has to be set to 0.

POROSITIES = (RAND + 2.5) / 5.0  # Scale to [0.1, 0.5] range.
POROSITIES = POROSITIES[1:]  # Porosities are cellwise, not facewise.

SOURCE_TERMS = jnp.full(
    (NUM_CELLS, 2), 0.0
)  # Source terms for cells c0, c1,... . Total and wetting source.

NEU_BC = jnp.array(
    [
        [1.0, 0.1],
        [0.0, 0.0],
    ]
)  # Neumann boundary conditions for left and right boundary. Total flux and wetting flux.
DIR_BC = jnp.array(
    [
        [0.0, 0.0],
        [0.0, 0.98],
    ]
)  # Dirichlet boundary conditions for for left and right boundary. Pressure and saturation.

INITIAL_CONDITIONS = jnp.stack(
    [
        jnp.zeros((NUM_CELLS,)),
        jnp.full((NUM_CELLS,), 0.98),
    ],
    axis=1,
).flatten()  # Initial conditions for pressure and saturation at cell centers.

MU_W = 1.0  # Viscosity of the wetting phase.
MU_N = 1.0  # Viscosity of the nonwetting phase.


def smooth_heaviside(x, k=20.0):
    """Smooth approximation to the Heaviside function."""
    return 0.5 * (1 + jnp.tanh(k * x))


def hessian_tensor(f):
    """Return a function that computes the Hessian tensor of f at a point x."""

    @jax.jit
    def per_component_hessian(x, *args, **kwargs):
        return jnp.stack(
            [
                jax.hessian(
                    lambda x_, *args, **kwargs: f(x_, *args, **kwargs)[i], argnums=0
                )(x, *args, **kwargs)
                for i in range(
                    f(x, *args, **kwargs).shape[0]
                )  # TODO: Is this independent of beta? Or do we get the wrong Hessian at later betas, because beta updates but the Hessian function does not.
            ]
        )

    return per_component_hessian


def apply_hessian(H, u, v):
    """Apply the Hessian tensor H to vectors u and v."""
    return jnp.einsum("ijk,j,k->i", H, u, v)


class TPFModel:
    def __init__(self, **kwargs) -> None:
        self.num_cells = NUM_CELLS
        self.transmissibilities = TRANSMISSIBILITIES
        self.porosities = POROSITIES
        self.source_terms = SOURCE_TERMS
        self.neumann_bc = NEU_BC
        self.dirichlet_bc = DIR_BC
        self.initial_conditions = INITIAL_CONDITIONS
        self.mu_w = MU_W
        self.mu_n = MU_N

        self.nb = kwargs.get("nb", 2)
        self.p_e = kwargs.get("p_e", 5.0)
        self.n1 = kwargs.get("n1", 2)
        self.n2 = kwargs.get("n2", 1 + 2 / self.nb)
        self.n3 = kwargs.get("n3", 1)

        self.linear_system = (
            None  # Placeholder for the linear system (Jacobian, residual).
        )

    def pc(self, s):
        """Capillary pressure function. Brooks-Corey model.

        Limit to above to avoid problems with Newton when :math:`s_w = 0`.

        """
        return jnp.minimum(
            jnp.nan_to_num(
                self.p_e * s ** (-1 / self.nb),
                nan=0.0,
                posinf=self.p_e * 10,
                neginf=0.0,
            ),
            self.p_e * 10,
        )

    def mobility_w(self, s):
        """Mobility of the wetting phase. Brooks-Corey model."""
        k_w = jnp.where(
            s <= 1,
            jnp.where(s >= 0, s ** (self.n1 + self.n2 * self.n3), 0.0),  # type: ignore
            1.0,
        )
        return k_w / self.mu_w  # type: ignore

    def mobility_n(self, s):
        """Mobility of the nonwetting phase. Brooks-Corey model."""
        k_n = jnp.where(
            s >= 0,
            jnp.where(s <= 1, (1 - s) ** self.n1 * (1 - s**self.n2) ** self.n3, 0.0),  # type: ignore
            1.0,
        )
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

        # TPFA fluxes at the faces.
        p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
        p_c_gradient = (
            p_c[:-1] - p_c[1:]
        )  # Negative capillary pressure gradient across cells.

        # Phase potential upwinding of the mobilities. The mobilities for Neumann
        # boundaries are upwinded as well. This is not a problem, because the
        # transmissibilities are set to 0 at the Neumann boundary.
        m_w = jnp.where(
            p_n_gradient >= 0, self.mobility_w(s[:-1]), self.mobility_w(s[1:])
        )
        m_n = jnp.where(
            p_n_gradient - p_c_gradient >= 0,
            self.mobility_n(s[:-1]),
            self.mobility_n(s[1:]),
        )

        # Handle NaN values in mobilities.
        m_n = jnp.nan_to_num(m_n, nan=0.0)
        m_w = jnp.nan_to_num(m_w, nan=0.0)

        m_t = m_n + m_w

        F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        F_w = (
            F_t * (m_w / m_t) - self.transmissibilities * m_w * m_n / m_t * p_c_gradient
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
        # jax.make_jaxpr(jax.jacrev(self.residual, argnums=0))(x, dt, x_prev=x_prev)
        return J.reshape(-1, len(x))


def newton(model, x_init, x_prev, dt, max_iter=100, tol=1e-6):
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
                {"residual_norm": jnp.linalg.norm(r) / jnp.sqrt(len(r))}
            )

            if jnp.linalg.norm(r) / jnp.sqrt(len(r)) < tol:
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
                raise RuntimeError("Newton step resulted in NaN or Inf values.")
            x += dx

            # Ensure physical saturation values plus small epsilon to avoid numerical issues.
            p = x[::2]
            s = jnp.clip(x[1::2], 1e-5, 1 - 1e-5)
            x = jnp.stack([p, s], axis=1).flatten()

    return x, converged


def hc(
    model,
    x_prev,
    x_init,
    dt,
    decay=1 / 30,
    max_iter=31,
    max_newton_iter=100,
    newton_tol=3e-5,
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

    with logging_redirect_tqdm([logger]):
        for i in hc_progressbar:
            hc_progressbar.set_postfix({r"$\lambda$": model.beta})

            # Previous solution for the predictor step. Newton's method for the corrector
            # step.
            try:
                x, converged = newton(
                    model, x, x_prev, max_iter=max_newton_iter, tol=newton_tol, dt=dt
                )
            except RuntimeError as _:
                converged = False

            if converged:
                # Store data for the homotopy curve BEFORE updating beta.
                model.store_curve_data(x, dt, x_prev=x_prev)
                model.store_intermediate_solutions(x)

                # Update the homotopy parameter beta only now.
                model.beta -= decay
                # Just for convenience, ensure beta is non-negative and equal to zero at
                # the end of the loop.
                if abs(model.beta) < 1e-3 or model.beta < 0:
                    model.beta = 0.0
            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at continuation"
                    + f" step {i + 1}, lambda={model.beta}."
                )
                break

    return x, converged


def solve(model, final_time, n_time_steps):
    """Solve the two-phase flow problem over a given time period."""
    dt = final_time / n_time_steps
    solutions = [model.initial_conditions]

    time_progressbar_position = 2 if hasattr(model, "beta") else 1
    time_progressbar = tqdm.trange(
        n_time_steps, desc="Time steps", position=time_progressbar_position, leave=False
    )

    solver = newton if not hasattr(model, "beta") else hc

    with logging_redirect_tqdm([logger]):
        for i in time_progressbar:
            time_progressbar.set_postfix({"time_step": i + 1})

            # Solve the nonlinear problem at the current time step.
            x_prev = solutions[-1]
            # Use the previous time step solution as the initial guess for the solver.
            try:
                (
                    x_next,
                    converged,
                ) = solver(model, x_prev, x_prev, dt=dt)
            except RuntimeError as _:
                converged = False

            if converged:
                solutions.append(x_next)
            else:
                logger.info(
                    f"Model {model.__class__.__name__} did not converge at time step {i + 1}."
                )
                break

    return solutions, converged


class HCModel(TPFModel):
    """Base class for homotopy continuation models."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = (
            1.0  # Homotopy parameter, 0 for simpler system, 1 for actual system.
        )
        self.betas = []  # Store beta values for the homotopy curve.
        self.tangents = []  # Store tangent vectors of the homotopy curve.
        self.curvature_vectors = []  # Store curvature vectors of the homotopy curve.)
        self.intermediate_solutions = []  # Store intermediate solver states.

        self.per_component_hessian = hessian_tensor(self.residual)

    def reset(self):
        self.beta = 1.0
        self.betas = []
        self.tangents = []
        self.curvature_vectors = []
        self.intermediate_solutions = []

    def h_beta_deriv(self, x, dt, x_prev=None):
        """Compute the derivative of the homotopy problem with respect to beta."""
        beta_save = self.beta

        self.beta = 1.0
        r_g = self.residual(x, dt, x_prev=x_prev)
        self.beta = 0.0
        r_f = self.residual(x, dt, x_prev=x_prev)

        self.beta = beta_save

        return r_g - r_f

    def tangent(self, x, dt, x_prev=None, jac=None):
        """Compute the tangent vector of the homotopy curve at (x, beta)."""
        b = self.h_beta_deriv(x, dt, x_prev=x_prev)
        A = self.jacobian(x, dt, x_prev=x_prev) if jac is None else jac
        tangent = jnp.linalg.solve(A, b)
        return tangent

    def curvature_vector(self, x, dt, x_prev=None, jac=None, tangent=None):
        if jac is None:
            jac = self.jacobian(x, dt, x_prev=x_prev)
        if tangent is None:
            tangent = self.tangent(x, dt, x_prev=x_prev, jac=jac)
        # Compute the Hessian of the residual.
        H = self.per_component_hessian(x, dt, x_prev=x_prev)
        w2 = apply_hessian(H, tangent, tangent)
        curvature_vector = jnp.linalg.solve(jac, -w2)
        return curvature_vector

    def store_curve_data(self, x, dt, x_prev=None):
        """Store the beta, tangent, and curvature data for the homotopy curve."""
        self.betas.append(self.beta)
        self.tangents.append(
            self.tangent(x, dt, x_prev=x_prev, jac=self.linear_system[0])  # type: ignore
        )
        self.curvature_vectors.append(
            self.curvature_vector(
                x,
                dt,
                x_prev=x_prev,
                jac=self.linear_system[0],  # type: ignore
                tangent=self.tangents[-1],
            )
        )

    def store_intermediate_solutions(self, x):
        self.intermediate_solutions.append(x)

    def curve_curvature_approx(self):
        hh = (jnp.asarray(self.betas[1:]) - jnp.asarray(self.betas[:-1]))[..., None]
        curvatures_approx = (
            jnp.asarray(self.tangents[1:]) - jnp.asarray(self.tangents[:-1])
        ) * (2 / (hh + hh))
        return jnp.linalg.norm(curvatures_approx, axis=-1)


class FluxHC1(HCModel):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for both capillary pressure and relative
    permeability.

    """

    def pc(self, s):
        return self.beta * self.p_e * (2 - s) + (1 - self.beta) * super().pc(s)

    def mobility_w(self, s):
        k_w = self.beta * s + (1 - self.beta) * super().mobility_w(s)
        return k_w / MU_W

    def mobility_n(self, s):
        k_n = self.beta * (1 - s) + (1 - self.beta) * super().mobility_n(s)
        return k_n / MU_N


class FluxHC2(FluxHC1):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for relative permeability and zero capillary
    pressure.

    """

    def pc(self, s):
        return (1 - self.beta) * super().pc(s)


class FluxHC3(FluxHC1):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for relative permeability and capillary pressure
    and averaging instead of upwinding for mobility.

    """

    def compute_face_fluxes(self, p, s):
        """Convex combination of fluxes with mobility averaging and upwinding."""
        # Get upwinded fluxes.
        up_F_t, up_F_w = super().compute_face_fluxes(p, s)

        # Compute fluxes with mobility averaging. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        p_c = self.pc(s)

        p_n_gradient = p[:-1] - p[1:]
        p_c_gradient = p_c[:-1] - p_c[1:]

        # Mobility averaging. The mobilities at Neumann boundaries are averaged as well.
        # This is not a problem, because the transmissibilities are set to 0 at the
        # Neumann boundary.
        m_w = (self.mobility_w(s[:-1]) + self.mobility_w(s[1:])) / 2
        m_n = (self.mobility_n(s[:-1]) + self.mobility_n(s[1:])) / 2

        m_n = jnp.nan_to_num(m_n, nan=0.0)
        m_w = jnp.nan_to_num(m_w, nan=0.0)

        m_t = m_n + m_w

        av_F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        av_F_w = (
            av_F_t * (m_w / m_t)
            - self.transmissibilities * m_w * m_n / m_t * p_c_gradient
        )

        av_F_t = av_F_t.at[0].add(self.neumann_bc[0, 0])
        av_F_w = av_F_w.at[0].add(self.neumann_bc[0, 1])
        av_F_t = av_F_t.at[-1].add(self.neumann_bc[1, 0])
        av_F_w = av_F_w.at[-1].add(self.neumann_bc[1, 1])

        # Form convex combination.
        F_t = self.beta * av_F_t + (1 - self.beta) * up_F_t
        F_w = self.beta * av_F_w + (1 - self.beta) * up_F_w

        return F_t, F_w


class FluxHC4(FluxHC1):
    r"""Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for relative permeability and capillary pressure
    and smoothened upwinding for mobility, i.e., using :math:`\tanh` insted of the
    Heaviside function.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = kwargs.get("k", 5.0)  # Smoothing parameter for the Heaviside function.

    def compute_face_fluxes(self, p, s):
        """Convex combination of fluxes with smooth Heaviside and upwinding (Heaviside)
        evaluation for the phase mobilities."""
        # Get upwinded fluxes.
        up_F_t, up_F_w = super().compute_face_fluxes(p, s)

        # Compute fluxes with mobility averaging. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        p_c = self.pc(s)

        p_n_gradient = p[:-1] - p[1:]
        p_c_gradient = p_c[:-1] - p_c[1:]

        # Smooth Heaviside convex combination of mobilities. Replacing the smooth
        # Heaviside approximation with the Heaviside function, this would give phase
        # potential upwinding.
        smooth_heaviside_pn = smooth_heaviside(p_n_gradient, k=self.k)
        smooth_heaviside_pw = smooth_heaviside(p_n_gradient - p_c_gradient, k=self.k)

        m_w = smooth_heaviside_pw * self.mobility_w(s[:-1]) + (
            1 - smooth_heaviside_pw
        ) * self.mobility_w(s[1:])
        m_n = smooth_heaviside_pn * self.mobility_n(s[:-1]) + (
            1 - smooth_heaviside_pn
        ) * self.mobility_n(s[1:])

        m_n = jnp.nan_to_num(m_n, nan=0.0)
        m_w = jnp.nan_to_num(m_w, nan=0.0)

        m_t = m_n + m_w

        hs_F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        hs_F_w = (
            hs_F_t * (m_w / m_t)
            - self.transmissibilities * m_w * m_n / m_t * p_c_gradient
        )

        hs_F_t = hs_F_t.at[0].add(self.neumann_bc[0, 0])
        hs_F_w = hs_F_w.at[0].add(self.neumann_bc[0, 1])
        hs_F_t = hs_F_t.at[-1].add(self.neumann_bc[1, 0])
        hs_F_w = hs_F_w.at[-1].add(self.neumann_bc[1, 1])

        # Form convex combination.
        F_t = self.beta * hs_F_t + (1 - self.beta) * up_F_t
        F_w = self.beta * hs_F_w + (1 - self.beta) * up_F_w

        return F_t, F_w


class DiffusionHC(HCModel):
    r"""Vanishing diffusion homotopy continuation for the two-phase flow problem.

    Cf. Jiang, J. and Tchelepi, H.A. (2018) ‘Dissipation-based continuation method for
    multiphase flow in heterogeneous porous media’, Journal of Computational Physics,
    375, pp. 307–336. Available at: https://doi.org/10.1016/j.jcp.2018.08.044.

    The parameter controlling the strength of the diffusion, called :math:`\beta` in the
    paper, is denoted :math:`\kappa` here (the continuation parameter
    :math:`\lambda`/:math:`\beta` is called :math:`\kappa` in the paper).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kappa = kwargs.get("kappa", 1.0)

    def compute_face_fluxes(self, p, s):
        """Compute total and wetting phase fluxes at cell faces with vanishing diffusion."""
        # Base model fluxes.
        F_t, F_w = super().compute_face_fluxes(p, s)

        # Compute vanishing dissipation.
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])
        s_w_gradient = s[:-1] - s[1:]  # Negative pressure gradient across cells.
        F_w += self.kappa * self.beta * self.transmissibilities * s_w_gradient

        return F_t, F_w


def plot_solution(model, solutions, plot_pw=False):
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
    if plot_pw:
        wetting_pressures = solutions_array[:, ::2] - model.pc(solutions_array[:, 1::2])
    else:
        wetting_pressures = jnp.zeros_like(solutions_array[:, ::2])

    (pn_line,) = ax.plot(xx, pressures, "o-", color="tab:blue", label=r"$p_n$")
    (pw_line,) = ax.plot(
        xx, wetting_pressures[0], "x-", color="tab:green", label=r"$p_w$"
    )
    (s_line,) = ax2.plot(xx, saturations, "v-", color="tab:orange", label=r"$s_w$")

    # Find min and max pressure values for consistent y-axis.
    p_min = min(
        jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).min(),  # type: ignore
        0,
    )
    p_max = jnp.concatenate([solutions_array[:, ::2], wetting_pressures]).max() * 1.1
    ax.set_ylim(p_min, p_max)  # type: ignore

    ax2.set_ylim(0, 1)  # Saturation is between 0 and 1.

    ax.set_xlabel("x")
    ax.set_ylabel("Pressures", color="tab:blue")
    ax2.set_ylabel(r"Wetting saturation $s_w$", color="tab:orange")

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
        pn_line.set_ydata(solutions_array[time_idx, ::2])
        pw_line.set_ydata(wetting_pressures[time_idx])
        s_line.set_ydata(solutions_array[time_idx, 1::2])
        ax.set_title(f"Two-Phase Flow Solution (Time Step: {time_idx})")
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    plt.show()


def weighted_distance(approximations, exact_solution):
    """Weight distance in pressure and saturations separately by the magnitude of the
    exact solution."""
    approximations = jnp.asarray(approximations)

    pressure_norm = jnp.linalg.norm(exact_solution[::2])
    saturation_norm = jnp.linalg.norm(exact_solution[1::2])

    distances = (
        jnp.linalg.norm(
            approximations[:, ::2] - exact_solution[None, ...][:, ::2],
            axis=-1,
        )
        / pressure_norm
        + jnp.linalg.norm(
            approximations[:, 1::2] - exact_solution[None, ...][:, 1::2],
            axis=-1,
        )
        / saturation_norm
    )

    return distances


def plot_curvature(betas, curvatures=None, distances=None, fig=None, **kwargs):
    """Plot the curvature of the homotopy curve."""
    betas = jnp.asarray(betas)

    # Create figure and axis.
    if fig is None:
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax2 = ax1.twinx()
    else:
        ax1, ax2 = fig.axes

    # Plot curvature and distance over \lambda range.
    if curvatures is not None:
        curvatures = jnp.asarray(curvatures)
        ax1.plot(betas, curvatures, **kwargs)
    if distances is not None:
        distances = jnp.asarray(distances)
        ax2.plot(betas, distances, **kwargs)

    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(r"$\kappa$")
    ax2.set_ylabel(
        r"$\frac{\|\mathbf{x}_{\lambda=0} - \mathbf{x}_{\lambda=1}\|}{\|\mathbf{x}_{\lambda=0}\|}$"
    )
    ax1.set_xlim(1, 0)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_title("Homotopy Curve Curvature and Distance to Solution")
    ax1.grid(True)
    ax1.legend()
    ax2.legend()

    return fig


# model = TPFModel(p_e=0.2)
# solutions, _ = solve(model, final_time=1.0, n_time_steps=10)
# plot_solution(model, solutions)

# model = TPFModel(p_e=2.0)
# solutions, _ = solve(model, final_time=1.0, n_time_steps=10)
# plot_solution(model, solutions, plot_pw=True)

# model = TPFModel(p_e=2.0)
# solutions, _ = solve(model, final_time=0.5, n_time_steps=1)
# # plot_solution(model, solutions, plot_pw=True)

# solutions_a = solutions.copy()

# tracer = viztracer.VizTracer(
#     min_duration=1e3,  # μs
#     ignore_c_function=True,
#     ignore_frozen=True,
# )
# tracer.start()
# tracer.stop()
# tracer.save(str(dirname / "traces.json"))

for p_e in [0.05, 0.2, 2.0]:
    # for p_e in [0.05, 2.0, 10.0]:
    model_fluxhc1 = FluxHC1(p_e=p_e)
    model_fluxhc2 = FluxHC2(p_e=p_e)
    model_fluxhc3 = FluxHC3(p_e=p_e)
    model_fluxhc4 = FluxHC4(p_e=p_e, k=3.0)
    model_diffhc1 = DiffusionHC(p_e=p_e, kappa=0.01)
    model_diffhc2 = DiffusionHC(p_e=p_e, kappa=0.1)
    model_diffhc3 = DiffusionHC(p_e=p_e, kappa=1.0)

    for final_time in [0.5, 1.0, 10.0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()

        color_palette_fluxhc = sns.color_palette("rocket", 4)
        color_palette_diffhc = sns.color_palette("mako", 3)

        for i, (model_fluxhc, color) in enumerate(
            zip(
                [model_fluxhc1, model_fluxhc2, model_fluxhc3, model_fluxhc4],
                color_palette_fluxhc,
            ),
            start=1,
        ):
            model_fluxhc.reset()
            _, converged = solve(model_fluxhc, final_time=final_time, n_time_steps=1)
            if converged:
                # plot_solution(
                #     model_fluxhc1, model_fluxhc1.intermediate_solutions, plot_pw=False
                # )
                print(
                    f"Flux HC {i}: Relative distance between solutions:"
                    + f" {jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1] - model_fluxhc.intermediate_solutions[0]) / jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1])}"
                )
                print(
                    f"Flux HC {i}: Relative distance between pressure solutions:"
                    + f" {jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1][::2] - model_fluxhc.intermediate_solutions[0][::2]) / jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1][::2])}"
                )
                print(
                    f"Flux HC {i}: Relative distance between saturation solutions:"
                    + f" {jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1][1::2] - model_fluxhc.intermediate_solutions[0][1::2]) / jnp.linalg.norm(model_fluxhc.intermediate_solutions[-1][1::2])}"
                )
            if len(model_fluxhc.betas) > 1:
                fig = plot_curvature(
                    model_fluxhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_fluxhc.curvature_vectors), axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa$ (Flux HC {i})",
                    color=color,
                    marker="o",
                )
                fig = plot_curvature(
                    model_fluxhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_fluxhc.curvature_vectors)[:, ::2], axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa_p$ (Flux HC {i})",
                    color=color,
                    ls="--",
                    marker="v",
                )
                fig = plot_curvature(
                    model_fluxhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_fluxhc.curvature_vectors)[:, 1::2], axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa_s$ (Flux HC {i})",
                    color=color,
                    ls="-.",
                    marker="x",
                )
                intermediate_solutions = jnp.asarray(
                    model_fluxhc.intermediate_solutions
                )
                distances = weighted_distance(
                    model_fluxhc.intermediate_solutions[:-1],
                    model_fluxhc.intermediate_solutions[-1],
                )
                fig = plot_curvature(
                    model_fluxhc.betas[:-1],
                    distances=distances,
                    fig=fig,
                    label=rf"$\frac{{\|\mathbf{{x}}_{{\lambda=0}} - \mathbf{{x}}_{{\lambda=1}}\|}}{{\|\mathbf{{x}}_{{\lambda=0}}\|}}$ (Flux HC {i})",
                    color=color,
                    ls=":",
                    marker="^",
                )
        try:
            fig.savefig(
                dirname / f"curvature_small_flux_hc_T_{final_time}_pe_{p_e}.png",
                bbox_inches="tight",
            )
        except Exception as _:
            pass
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()
        for i, (model_diffhc, color) in enumerate(
            zip([model_diffhc1, model_diffhc2, model_diffhc3], color_palette_diffhc),
            start=1,
        ):
            model_diffhc.reset()
            _, converged = solve(model_diffhc, final_time=final_time, n_time_steps=1)
            if converged:
                # plot_solution(
                #     model_diffhc, model_diffhc.intermediate_solutions, plot_pw=False
                # )
                print(
                    f"Diffusion HC {i}: Relative distance between solutions:"
                    + f" {jnp.linalg.norm(model_diffhc.intermediate_solutions[-1] - model_diffhc.intermediate_solutions[0]) / jnp.linalg.norm(model_diffhc.intermediate_solutions[-1])}"
                )
                print(
                    f"Diffusion HC {i}: Relative distance between pressure solutions:"
                    + f" {jnp.linalg.norm(model_diffhc.intermediate_solutions[-1][::2] - model_diffhc.intermediate_solutions[0][::2]) / jnp.linalg.norm(model_diffhc.intermediate_solutions[-1][::2])}"
                )
                print(
                    f"Diffusion HC {i}: Relative distance between saturation solutions:"
                    + f" {jnp.linalg.norm(model_diffhc.intermediate_solutions[-1][1::2] - model_diffhc.intermediate_solutions[0][1::2]) / jnp.linalg.norm(model_diffhc.intermediate_solutions[-1][1::2])}"
                )
            if len(model_diffhc.betas) > 1:
                fig = plot_curvature(
                    model_diffhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_diffhc.curvature_vectors), axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa$ (Diffusion HC {i})",
                    color=color,
                    marker="o",
                )
                fig = plot_curvature(
                    model_diffhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_diffhc.curvature_vectors)[:, ::2], axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa_p$ (Diffusion HC {i})",
                    color=color,
                    ls="--",
                    marker="v",
                )
                fig = plot_curvature(
                    model_diffhc.betas,
                    curvatures=jnp.linalg.norm(
                        jnp.asarray(model_diffhc.curvature_vectors)[:, 1::2], axis=-1
                    ),
                    fig=fig,
                    label=rf"$\kappa_s$ (Diffusion HC {i})",
                    color=color,
                    ls="-.",
                    marker="x",
                )
                intermediate_solutions = jnp.asarray(
                    model_diffhc.intermediate_solutions
                )
                distances = weighted_distance(
                    model_diffhc.intermediate_solutions[:-1],
                    model_diffhc.intermediate_solutions[-1],
                )
                fig = plot_curvature(
                    model_diffhc.betas[:-1],
                    distances=distances,
                    fig=fig,
                    label=rf"$\frac{{\|\mathbf{{x}}_{{\lambda=0}} - \mathbf{{x}}_{{\lambda=1}}\|}}{{\|\mathbf{{x}}_{{\lambda=0}}\|}}$ (Diffusion HC {i})",
                    color=color,
                    ls=":",
                    marker="^",
                )

        try:
            fig.savefig(
                dirname / f"curvature_small_diff_hc_T_{final_time}_pe_{p_e}.png",
                bbox_inches="tight",
            )
        except Exception as _:
            pass
        plt.close(fig)
