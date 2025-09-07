import numpy as np


# ==================================================================================================
def test_bilaplacian_builder(prior_build_setup) -> None:
    prior_object, built_prior_object, size = prior_build_setup
    rng = np.random.default_rng(1)
    state_array = rng.random(size)
    direction_array = rng.random(size)

    cost = prior_object.evaluate_cost(state_array)
    built_cost = built_prior_object.evaluate_cost(state_array)
    grad = prior_object.evaluate_gradient(state_array)
    built_grad = built_prior_object.evaluate_gradient(state_array)
    hvp = prior_object.evaluate_hessian_vector_product(direction_array)
    built_hvp = built_prior_object.evaluate_hessian_vector_product(direction_array)

    assert np.isclose(cost, built_cost)
    assert np.allclose(grad, built_grad)
    assert np.allclose(hvp, built_hvp)
