"""Homotopy continuation models and evaluation functions for 1D two-phase flow problems.

For more information on evaluating the curvature of homotopy solution curves, see

Brown, D.A. and Zingg, D.W. (2016) ‘Efficient numerical differentiation of
implicitly-defined curves for sparse systems’, Journal of Computational and Applied
Mathematics, 304, pp. 138–159. Available at: https://doi.org/10.1016/j.cam.2016.03.002.

and

Brown, D.A. and Zingg, D.W. (2017) ‘Design and evaluation of homotopies for efficient
and robust continuation’, Applied Numerical Mathematics, 118, pp. 150–181. Available at:
https://doi.org/10.1016/j.apnum.2017.03.001.


"""

import logging
import pathlib
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import seaborn as sns
from jax.typing import ArrayLike as ArrayLike_jax
from matplotlib import pyplot as plt
from model import TPFModel

parent_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from convex_hull import HullSide, convex_hull

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sns.set_theme(style="whitegrid")


def hessian_tensor(f: Callable) -> Callable:
    """Return a function that computes the Hessian tensor of f at a point x."""

    @jax.jit
    def per_component_hessian(x, *args, **kwargs):
        return jnp.stack(
            [
                jax.hessian(
                    lambda x_, *args, **kwargs: f(x_, *args, **kwargs)[i], argnums=0
                )(x, *args, **kwargs)
                for i in range(f(x, *args, **kwargs).shape[0])
                # TODO: Is this independent of beta? Or do we get the wrong Hessian at
                # later betas, because beta updates but the Hessian function does not.
            ]
        )

    return per_component_hessian


def apply_hessian(H: Callable, u: jax.Array, v: jax.Array) -> jax.Array:
    """Apply the Hessian tensor H to vectors u and v."""
    return jnp.einsum("ijk,j,k->i", H, u, v)


def smooth_heaviside(x: ArrayLike_jax, k=20.0):
    """Smooth approximation to the Heaviside function."""
    return 0.5 * (1 + jnp.tanh(k * x))


class HCModel(TPFModel):
    """Base class for homotopy continuation models.

    Note: For computing the tangent and curvature, it would be cleaner to define two
    residual functions, ``residual_initial`` and ``residual_target``, and compute
    Jacobian and Hessian for both of them. However, this would require duplicating the
    code for the residuals for all concrete homotopies.
    Instead, we leave the concrete implementation of the homotopy to the subclasses and
    compute tangent and curvatures by temporarily shifting :math:`\beta` to 0 and 1,
    respectively.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Homotopy parameter, 1 for simpler system, 0 for actual system.
        self.beta = 1.0

        # Store curve data.
        self.betas: list[float] = []
        self.tangents: list[jnp.ndarray] = []
        self.curvature_vectors: list[jnp.ndarray] = []
        self.intermediate_solutions: list[jnp.ndarray] = []

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
            \frac{\partial \mathcal{R}_H}{\partial \beta} = \mathcal{R}_g - \mathcal{R}_F,


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
         \beta)`.

        Note: The full tangent vector includes the derivative :math:`\beta' = -1`,
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
        # to the partial derivatives w.r.t. :math:`\beta`.
        # To compute the missing components of the Hessian, we note that
        # :math:`\partial_{\mathbf{x}_j}\partial_{\beta}\mathcal{H} =
        # \partial_{\mathbf{x}_j}\mathcal{R}_G - \partial_{\mathbf{x}_j}\mathcal{R}_F`
        # and :math:`\partial_{\beta}^2 \mathcal{H} = 0`.
        beta_save = self.beta
        self.beta = 1.0
        jac_g = self.jacobian(x, dt, x_prev=x_prev)
        self.beta = 0.0
        jac_f = self.jacobian(x, dt, x_prev=x_prev)
        self.beta = beta_save

        # Since :math:`\beta` decreases, we know that :math:`\beta' = -1` (w.r.t. to
        # an auxiliary parameter :math:`r`).
        # The missing components of the Hessian applied to the tangent vector are
        # :math:`\partial_{\mathbf{x}_j}\partial_{\beta}\mathcal{H}_i \cdot \tau_j
        # \cdot -1`,
        # :math:`\partial_{\beta}\partial_{\mathbf{x}_j} \mathcal{H}_i \cdot -1 \cdot
        # \tau_j`,
        # and :math:`\partial_{\beta}^2 \mathcal{H}_i \cdot -1 \cdot -1 = 0`.
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


class LinearCapPressHC(HCModel):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear capillary pressure function. Can be combined with
    :class:`LinearRelPermHC`.

    """

    def pc(self, s):
        return self.beta * TPFModel.pc(self, s, cp_model="linear") + (
            1 - self.beta
        ) * TPFModel.pc(self, s)


class ZeroCapPressHC(HCModel):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has zero capillary flow. Can be combined with
    :class:`LinearRelPermHC`.

    """

    def pc(self, s):
        return (1 - self.beta) * TPFModel.pc(self, s)


class LinearRelPermHC(HCModel):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has linear relative permeabilities. Can be combined with
    :class:`LinearCapPressHC` or :class:`ZeroCapPressHC`.

    """

    def mobility_w(self, s):
        return self.beta * TPFModel.mobility_w(self, s, rp_model="linear") + (
            1 - self.beta
        ) * TPFModel.mobility_w(self, s)

    def mobility_n(self, s):
        return self.beta * TPFModel.mobility_n(self, s, rp_model="linear") + (
            1 - self.beta
        ) * TPFModel.mobility_n(self, s)


class SmoothUpwindHC(HCModel):
    """Flux homotopy continuation for the two-phase flow problem.

    Initial problem has smoothed upwinding, linear relative permeabilities, and linear
    or zero capillary pressure function.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.k = kwargs.get("k", 5.0)  # Smoothing parameter for the Heaviside function.
        self.initial_cp_model = kwargs.get("initial_cp_model", "linear")

    def compute_face_fluxes(self, p, s):
        """Convex combination of fluxes with smooth Heaviside and upwinding (Heaviside)
        evaluation for the phase mobilities."""
        # Get upwinded fluxes with target constitutive laws.
        up_F_t, up_F_w = super().compute_face_fluxes(p, s)

        # Compute fluxes with smoothed Heaviside. For documentation of the code check
        # :meth:`TPFModel.compute_face_fluxes`.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        # Use the zero capillary pressure function for the initial problem.
        p_c = self.pc(s, cp_model=self.initial_cp_model)

        p_n_gradient = p[:-1] - p[1:]
        p_c_gradient = p_c[:-1] - p_c[1:]

        # Smooth Heaviside convex combination of mobilities. Replacing the smooth
        # Heaviside approximation with the Heaviside function, this would give phase
        # potential upwinding.
        smooth_heaviside_pn = smooth_heaviside(p_n_gradient, k=self.k)
        smooth_heaviside_pw = smooth_heaviside(p_n_gradient - p_c_gradient, k=self.k)

        # Use linear relative permeabilities for the initial problem.
        m_w = smooth_heaviside_pw * self.mobility_w(s[:-1], rp_model="linear") + (
            1 - smooth_heaviside_pw
        ) * self.mobility_w(s[1:], rp_model="linear")
        m_n = smooth_heaviside_pn * self.mobility_n(s[:-1], rp_model="linear") + (
            1 - smooth_heaviside_pn
        ) * self.mobility_n(s[1:], rp_model="linear")

        m_n = jnp.nan_to_num(m_n, nan=0.0)
        m_w = jnp.nan_to_num(m_w, nan=0.0)

        m_t = m_n + m_w

        hs_F_t = self.transmissibilities * (m_t * p_n_gradient - m_w * p_c_gradient)
        hs_F_w = (
            hs_F_t * (m_w / m_t)
            - self.transmissibilities * m_w * m_n / m_t * p_c_gradient
        )

        # Add Neumann boundary conditions.
        if self.bc[0] == "neumann":
            hs_F_t = hs_F_t.at[0].set(self.neumann_bc[0, 0])
            hs_F_w = hs_F_w.at[0].set(self.neumann_bc[0, 1])
        if self.bc[1] == "neumann":
            hs_F_t = hs_F_t.at[-1].set(self.neumann_bc[1, 0])
            hs_F_w = hs_F_w.at[-1].set(self.neumann_bc[1, 1])

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
    paper, is called `diffusion_coeffficient` here (the continuation parameter
    :math:`\beta` is called :math:`\kappa` in the paper).

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adaptive_diffusion_coeff = kwargs.get("adaptive_diffusion_coeff", 1.0)
        self.fixed_diffusion_coeff = kwargs.get("fixed_diffusion_coeff", 1e-2)

    def compute_face_fluxes(self, p, s):
        """Compute total and wetting phase fluxes at cell faces with vanishing
        diffusion.

        At domain boundaries the diffusion is nulled.

        """
        # Base model fluxes.
        F_t, F_w = super().compute_face_fluxes(p, s)

        # Diffusion flux.
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])
        s_w_gradient = s[:-1] - s[1:]  # Negative saturation gradient across cells.
        diffusive_flux = (
            self.adaptive_diffusion_coeff * self.transmissibilities * s_w_gradient
        )
        # No diffusion at the boundaries.
        diffusive_flux = diffusive_flux.at[0].set(0.0)
        diffusive_flux = diffusive_flux.at[-1].set(0.0)

        F_w += self.beta * diffusive_flux

        return F_t, F_w

    def update_adaptive_diffusion_coeff(self, dt, x_prev):
        r"""Optimize the diffusion parameter kappa for the given problem.

        We follow the approach from Jiang and Tchelepi (2018), which defines the
        vanishing artificial diffusion flux for the fully coupled flow and transport
        problem as

        .. math::
            F_{diff, ij} = \kappa \alpha_{ij} (S_i - S_j),

        with the locally adaptive diffusion coefficient :math:`\alpha_{ij}` defined via
        a CFL-like formula as

        .. math::
            \alpha_{ij} = \omega \frac{\Delta t}{\phi_{ij} \Delta x_{ij}} \max |\left(\frac{\beta_{w, ij}{\beta_{t, ij} u_{T,ij}\right)'|,

        where :math:`\Delta t` is the time step size, :math:`\Delta x_{ij}` the
        (averaged) cell size, :math:`\phi_{ij}` the (averaged?) porosity,
        :math:`\left(\frac{\beta_{w, ij}{\beta_{t, ij} u_{T,ij}\right)'` the
        derivative of the wetting flux, and :math:`omega` a constant tuning factor
        chosen as $1.0 \times 10^{-5}$ in the paper.

        The method was derived for a sequential-implicit method, which first solves the
        flow to find :math:`u_T` and then the transport problem. For the fully implicit
        method, we do not know the value of :math:`u_T` a priori. **However,** to not
        degrade performance, the paper assumes :math:`u_T` to be **constant** during
        each time step. This enables to use the method directly for the fully implicit
        scheme.

        In the presence of capillary forces, :math:`F_{w,ij}` becomes a function of the
        saturations on both sizes of the interface. In particular, due to the upwinding
        it is no longer smooth and the maximum of its gradient is not defined.

        We assume unity cell size.

        """
        dx = 1.0

        # Map cell porosities to cell faces by averaging.
        face_porosities = (self.porosities[1:] + self.porosities[:-1]) / 2
        face_porosities = jnp.concatenate(
            [self.porosities[:1], face_porosities, self.porosities[-1:]]
        )

        # Compute the maximal flow gradient based on the total and capillary flux from
        # the previous time step.
        # TODO: Repeated code from :meth:`compute_face_fluxes`. Refactor.

        p = x_prev[::2]
        s = x_prev[1::2]

        # Add Dirichlet boundary conditions.
        p = jnp.concatenate([self.dirichlet_bc[0, :1], p, self.dirichlet_bc[1, :1]])
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s, self.dirichlet_bc[1, 1:]])

        p_c = self.pc(s)

        # TPFA fluxes at the faces.
        p_n_gradient = p[:-1] - p[1:]  # Negative pressure gradient across cells.
        p_c_gradient = (
            p_c[:-1] - p_c[1:]
        )  # Negative capillary pressure gradient across cells.

        # Phase potential upwinding of the mobilities. The mobilities for Neumann
        # boundaries are upwinded as well. This is not a problem, because the
        # transmissibilities are set to 0 at the Neumann boundary.
        m_w = jnp.where(
            p_n_gradient > 0, self.mobility_w(s[:-1]), self.mobility_w(s[1:])
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
        capillary_potential = p_c_gradient

        # Approximate capillary potential at Neumann boundaries with difference
        # between nonwetting and wetting flux divided by transmissibility.
        if self.bc[0] == "neumann":
            F_t = F_t.at[0].set(self.neumann_bc[0, 0])
            capillary_potential = capillary_potential.at[0].set(
                (self.neumann_bc[0, 1] - 2 * self.neumann_bc[0, 1])
                / self.transmissibilities[0]
            )
        if self.bc[1] == "neumann":
            F_t = F_t.at[-1].set(self.neumann_bc[1, 0])
            capillary_potential = capillary_potential.at[-1].set(
                (self.neumann_bc[0, 1] - 2 * self.neumann_bc[0, 1])
                / self.transmissibilities[-1]
            )
        # Dividing by 0 transmissibility can give NaN values.
        capillary_potential = jnp.nan_to_num(
            capillary_potential, nan=0.0, posinf=0.0, neginf=0.0
        )

        F_t = jnp.maximum(F_t, 1.0)

        max_flow_gradient = self.Fw_scalar_max_grad(F_t, capillary_potential)
        self.adaptive_diffusion_coeff = (
            self.fixed_diffusion_coeff * dt / (face_porosities * dx) * max_flow_gradient
        )

        if jnp.isnan(self.adaptive_diffusion_coeff).any():
            raise ValueError(
                "Adaptive diffusion coefficient contains NaN values. "
                "Check the input data and the model parameters."
            )

    def Fw_scalar(self, s, F_t, capillary_potential, transmissibility):
        r"""Compute the wetting flux function for fixed total and capillary flow.

        .. math::
            F_w(s_w) = \frac{\beta_w}{\beta_t} F_T - \kappa *\frac{\beta_w \beta_n}{\beta_t} \nabla p_c,

        where :math:`F_t` and :math:`\kappa *\frac{\beta_w \beta_n}{\beta_t}
        \nabla p_c` are fixed.


        """
        return (
            self.mobility_w(s) / (self.mobility_w(s) + self.mobility_n(s)) * F_t
            - transmissibility
            * (self.mobility_w(s) * self.mobility_n(s))
            / (self.mobility_w(s) + self.mobility_n(s))
            * capillary_potential
        )

    def Fw_scalar_max_grad(self, F_t, capillary_potential, n_points=100):
        # Sample saturations, while disregarding s=0 and s=1, as mobilities are cut off
        # there and gradients can return NaN.
        s = jnp.linspace(0.0, 1.0, n_points)[1:-1]

        # Compute the wetting flux for the given saturation values.
        # Somewhat messy construction to compute the maximum gradient along the
        # cartesian product of all saturation and all ``F_t`` and ``capillary_flow`` values.
        # ``.item()`` produces an error, instead we use ``.reshape(, ())``.
        batched_grad = jax.vmap(
            jax.vmap(
                jax.grad(
                    lambda a, b, c, d: jnp.reshape(self.Fw_scalar(a, b, c, d), ()),
                    argnums=0,
                ),
                in_axes=(0, None, None, None),
            ),
            in_axes=(None, 0, 0, 0),
        )
        grad_Fw = batched_grad(
            s,
            F_t,
            capillary_potential,
            self.transmissibilities.flatten(),
        )  # ``shape=(n_faces, n_points)``
        assert grad_Fw.shape == (F_t.shape[0], s.shape[0]), (
            f"Expected ``grad_Fw`` shape {(F_t.shape[0], s.shape[0])},"
            + f" got {grad_Fw.shape}."
        )
        return jnp.abs(grad_Fw).max(axis=-1)


class ConvexHullFluxHC(HCModel):
    r"""Convex hull homotopy continuation for the two-phase flow problem.

    Take the convex hull of the wetting flow function
    :math:`f_w = \frac{\beta_w}{\beta_t}` as the initial wetting flow function.

    Note: This assumes **ZERO** buoyancy and capillary forces, s.t. the numerical
    wetting flow function :math:`\frac{F_w}{F_t}` is a one-dimensional function of only
    the saturation in upstream direction.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convex_hull_side: HullSide = kwargs.get("convex_hull_side", "upper")
        self.convex_hull_saturation_interval: tuple[float, float] = kwargs.get(
            "convex_hull_saturation_interval", (0.0, 1.0)
        )
        self.initialize_convex_hull()

    def initialize_convex_hull(self):
        """Compute the convex hull of the wetting flow function."""

        def wetting_flow(s):
            return self.mobility_w(s) / (self.mobility_w(s) + self.mobility_n(s))

        self.wetting_flow_convex_hull = convex_hull(  #  type: ignore
            wetting_flow,
            self.convex_hull_saturation_interval,
            self.convex_hull_side,
            xp=jnp,
        )

    def compute_face_fluxes(self, p, s):
        """Compute total and wetting phase fluxes at cell faces with convex hull
        homotopy continuation.

        At domain boundaries the diffusion is nulled.

        """
        # Base model fluxes.
        F_t, target_F_w = super().compute_face_fluxes(p, s)

        # Compute the convex hull of the wetting flow function. The total flow goes from
        # left to right, so the upwinded mobility is evaluated left of the interface.
        s = jnp.concatenate([self.dirichlet_bc[0, 1:], s])
        initial_F_w = self.wetting_flow_convex_hull(s) * F_t

        if self.bc[0] == "neumann":
            initial_F_w = initial_F_w.at[0].set(self.neumann_bc[0, 1])
        if self.bc[1] == "neumann":
            initial_F_w = initial_F_w.at[-1].set(self.neumann_bc[1, 1])

        # Form convex combination.
        F_w = self.beta * initial_F_w + (1 - self.beta) * target_F_w

        return F_t, F_w

    def plot_convex_hull(self):
        """Plot the convex hull of the wetting flow function."""

        def wetting_flow(s):
            return self.mobility_w(s) / (self.mobility_w(s) + self.mobility_n(s))

        def linear_wetting_flow(s):
            return self.mobility_w(s, rp_model="linear") / (
                self.mobility_w(s, rp_model="linear")
                + self.mobility_n(s, rp_model="linear")
            )

        s_vals = jnp.linspace(0, 1, 100)
        fw_vals = wetting_flow(s_vals)
        fw_hull_vals = self.wetting_flow_convex_hull(s_vals)
        fw_linear_vals = linear_wetting_flow(s_vals)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(s_vals, fw_vals, label="$f_w(s)$", ls="--", color="gray")
        ax.plot(s_vals, fw_hull_vals, label=r"$\text{conv } f_w(s)$", color="blue")
        ax.plot(
            s_vals, fw_linear_vals, label="$f_{w,linear}(s)$", ls="--", color="green"
        )
        ax.set_xlabel("$s$")
        ax.set_title("Convex Hull of Wetting Flow Function")
        ax.legend()

        return fig
