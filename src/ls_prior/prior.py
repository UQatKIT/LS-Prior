import numpy as np


# ==================================================================================================
class Prior:
    
    # ----------------------------------------------------------------------------------------------
    def __init__(self, precision_operator, covariance_operator, covariance_factorization, prng):
        self._precision_operator = precision_operator
        self._covariance_operator = covariance_operator
        self._covariance_factorization = covariance_factorization
        self._prng = prng

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        cost = 0.5 * parameter_vector.T @ self._precision_operator.apply(parameter_vector)
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
        random_vector = self._prng.generate_gaussian_vector()
        sample_vector = self._covariance_factorization.apply(random_vector)
        return sample_vector
