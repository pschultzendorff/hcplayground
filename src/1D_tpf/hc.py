r"""Homotopy continuation models and evaluation functions for 1D two-phase flow problems.

Note: Throughout the module, the homotopy parameter variable is denoted as `beta`, while
    in the docstrings it is referred to as $\lambda$, as is common in the literature.

For more information on curvatures, see
Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

"""

import logging

import jax
import jax.numpy as jnp
import tqdm
from model import TPFModel, newton
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def smooth_heaviside(x, k=20.0):
    """Smooth approximation to the Heaviside function."""
    return 0.5 * (1 + jnp.tanh(k * x))


def hc(
    model,
    x_prev,
    x_init,
    dt,
    decay=1 / 30,
    max_iter=31,
    max_newton_iter=100,
    newton_tol=1e-4,
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


def solve(model, final_time, n_time_steps, **kwargs):
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
                ) = solver(model, x_prev, x_prev, dt=dt, **kwargs)
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
    r"""Base class for homotopy continuation models.

    Note: For computing the tangent and curvature, it would be a cleaner implementation
    to define two residual functions, `residual_initial` and `residual_target`, and
    compute Jacobian and Hessian for both of them. However, this would require
    duplicating the code for the residuals for all concrete homotopies.

    Instead, we leave the concrete implementation of the homotopy to the subclasses and
    compute tangent and curvatures by temporarily shifting :math:`\lambda` to 0 and 1,
    respectively.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = (
            1.0  # Homotopy parameter, 0 for simpler system, 1 for actual system.
        )
        self.betas = []  # Store beta values for the homotopy curve.
        self.tangents = []  # Store tangent vectors of the homotopy curve.
        self.curvature_vectors = []  # Store curvature vectors of the homotopy curve.
        self.intermediate_solutions = []  # Store intermediate solver states.

        self.per_component_hessian = hessian_tensor(self.residual)

    def reset(self):
        self.beta = 1.0
        self.betas = []
        self.tangents = []
        self.curvature_vectors = []
        self.intermediate_solutions = []

    def h_beta_deriv(self, x, dt, x_prev=None):
        r"""Compute the derivative of the homotopy problem with respect to beta.

        .. math::
            \frac{\partial \mathcal{R}_H}{\partial \lambda} = \mathcal{R}_g - \mathcal{R}_F,


        where :math:`\mathcal{R}_G` is the residual of the simpler system and
        :math:`\mathcal{r}_F` is the residual of the actual system.

        """
        beta_save = self.beta

        self.beta = 1.0
        r_g = self.residual(x, dt, x_prev=x_prev)
        self.beta = 0.0
        r_f = self.residual(x, dt, x_prev=x_prev)

        self.beta = beta_save

        return r_g - r_f

    def tangent(self, x, dt, x_prev=None, jac=None):
        r"""Compute the tangent vector of the homotopy curve at :math:`(\mathbf{x},
         \lambda)`.

        Note: The full tangent vector includes the derivative :math:`\lambda' = -1`,
        which is **not** included by this method. :meth:`curvature_vector` concatenates
        :math:`\mathbf{x}'` with :math:`-1` to form the full tangent vector.

        """
        b = self.h_beta_deriv(x, dt, x_prev=x_prev)
        A = self.jacobian(x, dt, x_prev=x_prev) if jac is None else jac
        tangent = jnp.linalg.solve(A, b)
        return tangent

    def curvature_vector(self, x, dt, x_prev=None, jac=None, tangent_star=None):
        # First, we compute the Hessian w.r.t. to :math:`\mathbf{x}` and apply it to the
        # corresponding tangent vector :math:`\mathbf{x}'`.
        if jac is None:
            jac = self.jacobian(x, dt, x_prev=x_prev)
        if tangent_star is None:
            tangent_star = self.tangent(x, dt, x_prev=x_prev, jac=jac)

        # Compute the Hessian of the residual.
        H_star = self.per_component_hessian(x, dt, x_prev=x_prev)
        w2_star = apply_hessian(H_star, tangent_star, tangent_star)

        # The above Hessian is a tensor of shape (n, n, n), where n is the number of
        # components of the residual. However, the full Hessian of :math:`\mathcal{H}`
        # is a tensor of shape (n + 1, n + 1, n), where the last dimension corresponds
        # to the partial derivatives w.r.t. :math:`\lambda`.
        # To compute the missing components of the Hessian, we note that
        # :math:`\partial_{\mathbf{x}_j}\partial_{\lambda}\mathcal{H} =
        # \partial_{\mathbf{x}_j}\mathcal{R}_G - \partial_{\mathbf{x}_j}\mathcal{R}_F`
        # and :math:`\partial_{\lambda}^2 \mathcal{H} = 0`.
        beta_save = self.beta
        self.beta = 1.0
        jac_g = self.jacobian(x, dt, x_prev=x_prev)
        self.beta = 0.0
        jac_f = self.jacobian(x, dt, x_prev=x_prev)
        self.beta = beta_save

        # Since :math:`\lambda` decreases, we know that :math:`\lambda' = -1` (w.r.t. to
        # an auxiliary parameter :math:`r`).
        # The missing components of the Hessian applied to the tangent vector are
        # :math:`\partial_{\mathbf{x}_j}\partial_{\lambda}\mathcal{H}_i \cdot \tau_j
        # \cdot -1`,
        # :math:`\partial_{\lambda}\partial_{\mathbf{x}_j} \mathcal{H}_i \cdot -1 \cdot
        # \tau_j`,
        # and :math:`\partial_{\lambda}^2 \mathcal{H}_i \cdot -1 \cdot -1 = 0`.
        # The first two terms are equal.
        w2 = w2_star - 2 * (jac_g - jac_f) @ tangent_star

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
                tangent_star=self.tangents[-1],
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
        k_w = s
        return self.beta * (k_w / self.mu_w) + (1 - self.beta) * super().mobility_w(s)

    def mobility_n(self, s):
        k_n = 1 - s
        return self.beta * (k_n / self.mu_n) + (1 - self.beta) * super().mobility_n(s)


class FluxHC2(FluxHC1):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for relative permeability and zero capillary
    pressure.

    """

    def pc(self, s):
        return (1 - self.beta) * TPFModel.pc(self, s)


class FluxHC3(FluxHC1):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear models for relative permeability and capillary pressure
    and averaging instead of upwinding for mobility.

    """

    # As :meth:`compute_face_fluxes` is overridden for the HC, we need to define **two**
    # capillary pressure methods. One for the target problem and one for the initial
    # problem. Otherwise, both the target and initial :meth`compute_face_fluxes` use the
    # same capillary pressure.
    def pc(self, s):
        return TPFModel.pc(self, s)

    def linear_pc(self, s):
        """Linear capillary pressure function."""
        return FluxHC1.pc(self, s)

    def compute_face_fluxes(self, p, s):
        """Convex combination of fluxes with mobility averaging and upwinding."""
        # Get upwinded fluxes. These use the target capillary pressure function.
        # :class:`FLuxHC1` does not override :meth:`TPFModel.compute_face_fluxes`, so we
        # can call it by using `super()`.
        up_F_t, up_F_w = super().compute_face_fluxes(p, s)

        # Compute fluxes with mobility averaging. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        # Use the linear capillary pressure function for the initial problem.
        p_c = self.linear_pc(s)

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


class FluxHC4(FluxHC3):
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
        # Get upwinded fluxes. These use the target capillary pressure function.
        # We explicitely call the method from the base class, not from the super class,
        # which is already a homotopy itself.
        up_F_t, up_F_w = TPFModel.compute_face_fluxes(self, p, s)

        # Compute fluxes with mobility averaging. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        # Use the linear capillary pressure function for the initial problem.
        p_c = self.linear_pc(s)

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


class FluxHC5(FluxHC2):
    r"""Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear modes for relative permeability and zero capillary
    pressure and smoothened upwinding for mobility, i.e., using :math:`\tanh` insted of
    the Heaviside function.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = kwargs.get("k", 5.0)  # Smoothing parameter for the Heaviside function.

    # As :meth:`compute_face_fluxes` is overridden for the HC, we need to define **two**
    # capillary pressure methods. One for the target problem and one for the initial
    # problem. Otherwise, both the target and initial :meth`compute_face_fluxes` use the
    # same capillary pressure.
    def pc(self, s):
        return TPFModel.pc(self, s)

    def zero_pc(self, s):
        """Zero capillary pressure."""
        return FluxHC2.pc(self, s)

    def compute_face_fluxes(self, p, s):
        """Convex combination of fluxes with smooth Heaviside and upwinding (Heaviside)
        evaluation for the phase mobilities."""
        # Get upwinded fluxes. These use the target capillary pressure function.
        # :class:`FLuxHC2` does not override :meth:`TPFModel.compute_face_fluxes`, so we
        # can call it by using `super()`.
        up_F_t, up_F_w = super().compute_face_fluxes(p, s)

        # Compute fluxes with mobility averaging. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        # Use the zero capillary pressure function for the initial problem.
        p_c = self.zero_pc(s)

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
