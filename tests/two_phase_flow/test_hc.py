# import pathlib
# import sys

# import jax.numpy as jnp

# parent_dir = pathlib.Path(__file__).resolve().parent.parent
# sys.path.append(str(parent_dir))

# import jax.numpy as jnp
# from src.1D_tpf.model import TPFModel
# from src.1D_tpf.hc import DiffusionHC, FluxHC1, FluxHC2, FluxHC3, FluxHC4, FluxHC5, solve


# def test_hc_models_at_beta_zero():
#     """Test that HC models with beta=0 give the same results as the base TPFModel."""

#     # Basic parameters for model setup
#     nx = 10
#     L = 1.0
#     dt = 0.1
#     phi = 0.2
#     K = 1.0
#     mu_w = 1.0
#     mu_n = 2.0
#     p_e = 0.1
#     s_wr = 0.2
#     s_nr = 0.1

#     # Create a reference TPFModel
#     reference_model = TPFModel(
#         nx=nx,
#         L=L,
#         phi=phi,
#         K=K,
#         mu_w=mu_w,
#         mu_n=mu_n,
#         p_e=p_e,
#         s_wr=s_wr,
#         s_nr=s_nr,
#     )

#     # Generate some test data
#     p = jnp.linspace(0.5, 1.5, nx)
#     s = jnp.linspace(0.3, 0.8, nx)
#     x = jnp.concatenate([p, s])

#     # Test all HC models
#     hc_models = {
#         "FluxHC1": FluxHC1(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#         "FluxHC2": FluxHC2(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#         "FluxHC3": FluxHC3(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#         "FluxHC4": FluxHC4(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#         "FluxHC5": FluxHC5(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#         "DiffusionHC": DiffusionHC(
#             nx=nx,
#             L=L,
#             phi=phi,
#             K=K,
#             mu_w=mu_w,
#             mu_n=mu_n,
#             p_e=p_e,
#             s_wr=s_wr,
#             s_nr=s_nr,
#         ),
#     }

#     for name, model in hc_models.items():
#         model.beta = 0.0

#         # Test constitutive relationships for models that override them
#         if name in ["FluxHC1", "FluxHC2"]:
#             # Test capillary pressure
#             ref_pc = reference_model.pc(s)
#             model_pc = model.pc(s)
#             assert jnp.allclose(ref_pc, model_pc), f"{name}: pc differs at beta=0"

#             # Test mobilities
#             ref_mob_w = reference_model.mobility_w(s)
#             model_mob_w = model.mobility_w(s)
#             assert jnp.allclose(ref_mob_w, model_mob_w), (
#                 f"{name}: mobility_w differs at beta=0"
#             )

#             ref_mob_n = reference_model.mobility_n(s)
#             model_mob_n = model.mobility_n(s)
#             assert jnp.allclose(ref_mob_n, model_mob_n), (
#                 f"{name}: mobility_n differs at beta=0"
#             )

#         # Test residual for all models
#         ref_residual = reference_model.residual(x, dt)
#         model_residual = model.residual(x, dt)
#         assert jnp.allclose(ref_residual, model_residual), (
#             f"{name}: residual differs at beta=0"
#         )


# def test_hc_models_at_beta_one():
#     """Test that HC models with beta=1 use the initial constitutive relationships."""

#     # Basic parameters for model setup
#     nx = 10
#     L = 1.0
#     phi = 0.2
#     K = 1.0
#     mu_w = 1.0
#     mu_n = 2.0
#     p_e = 0.1
#     s_wr = 0.2
#     s_nr = 0.1

#     s = jnp.linspace(0.3, 0.8, nx)

#     # FluxHC1 has linear cap pressure and linear rel perm
#     model1 = FluxHC1(
#         nx=nx, L=L, phi=phi, K=K, mu_w=mu_w, mu_n=mu_n, p_e=p_e, s_wr=s_wr, s_nr=s_nr
#     )
#     model1.beta = 1.0

#     # Check initial capillary pressure (linear)
#     assert jnp.allclose(model1.pc(s), model1.initial_pc(s)), (
#         "FluxHC1: pc != initial_pc at beta=1"
#     )

#     # Check initial mobilities (linear)
#     assert jnp.allclose(model1.mobility_w(s), model1.initial_mobility_w(s)), (
#         "FluxHC1: mobility_w != initial_mobility_w at beta=1"
#     )
#     assert jnp.allclose(model1.mobility_n(s), model1.initial_mobility_n(s)), (
#         "FluxHC1: mobility_n != initial_mobility_n at beta=1"
#     )

#     # FluxHC2 has zero cap pressure and linear rel perm
#     model2 = FluxHC2(
#         nx=nx, L=L, phi=phi, K=K, mu_w=mu_w, mu_n=mu_n, p_e=p_e, s_wr=s_wr, s_nr=s_nr
#     )
#     model2.beta = 1.0

#     # Check initial capillary pressure (zero)
#     assert jnp.allclose(model2.pc(s), model2.initial_pc(s)), (
#         "FluxHC2: pc != initial_pc at beta=1"
#     )

#     # Check initial mobilities (linear)
#     assert jnp.allclose(model2.mobility_w(s), model2.initial_mobility_w(s)), (
#         "FluxHC2: mobility_w != initial_mobility_w at beta=1"
#     )
#     assert jnp.allclose(model2.mobility_n(s), model2.initial_mobility_n(s)), (
#         "FluxHC2: mobility_n != initial_mobility_n at beta=1"
#     )


# def test_hc_model_tangent_and_curvature():
#     """Test that tangent and curvature computation works for HC models."""

#     # Basic parameters for model setup
#     nx = 10
#     L = 1.0
#     dt = 0.1
#     phi = 0.2
#     K = 1.0
#     mu_w = 1.0
#     mu_n = 2.0
#     p_e = 0.1
#     s_wr = 0.2
#     s_nr = 0.1

#     # Create a model
#     model = FluxHC1(
#         nx=nx, L=L, phi=phi, K=K, mu_w=mu_w, mu_n=mu_n, p_e=p_e, s_wr=s_wr, s_nr=s_nr
#     )

#     # Generate some test data
#     p = jnp.linspace(0.5, 1.5, nx)
#     s = jnp.linspace(0.3, 0.8, nx)
#     x = jnp.concatenate([p, s])

#     # Test tangent computation
#     tangent = model.tangent(x, dt)
#     assert tangent.shape == x.shape, "Tangent shape doesn't match state shape"

#     # Test curvature computation
#     jac = model.jacobian(x, dt)
#     tangent = model.tangent(x, dt, jac=jac)
#     curvature = model.curvature_vector(x, dt, jac=jac, tangent_star=tangent)
#     assert curvature.shape == x.shape, "Curvature shape doesn't match state shape"

#     # Test h_beta_deriv
#     h_beta = model.h_beta_deriv(x, dt)
#     assert h_beta.shape == x.shape, "h_beta_deriv shape doesn't match state shape"
