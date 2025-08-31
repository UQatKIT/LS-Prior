import numpy as np

from . import components


# ==================================================================================================
class Prior:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        precision_operator: components.NumpyComponent,
        covariance_operator: components.NumpyComponent,
        covariance_factorization: components.NumpyComponent,
        seed: int,
    ):
        self._precision_operator = precision_operator
        self._covariance_operator = covariance_operator
        self._covariance_factorization = covariance_factorization
        self._prng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        cost = 0.5 * np.inner(parameter_vector, self._precision_operator.apply(parameter_vector))
        assert cost > 0
        return cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        gradient = self._precision_operator.apply(parameter_vector)
        assert gradient.shape == parameter_vector.shape
        return gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        hessian_vector_product = self._precision_operator.apply(direction_vector)
        assert hessian_vector_product.shape == direction_vector.shape
        return hessian_vector_product

    # ----------------------------------------------------------------------------------------------
    def generate_sample(self):
        random_vector_size = self._covariance_factorization.input_dimension
        random_vector = self._prng.normal(loc=0.0, scale=1.0, size=random_vector_size)
        sample_vector = self._covariance_factorization.apply(random_vector)
        return sample_vector
