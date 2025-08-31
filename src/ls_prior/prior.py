import numpy as np

from . import components


# ==================================================================================================
class Prior:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        precision_operator: components.NumpyComponent,
        covariance_operator: components.NumpyComponent,
        covariance_factorization: components.NumpyComponent,
        dimension: int,
        seed: int,
    ):
        if not mean_vector.shape == (dimension,):
            raise ValueError(
                f"Mean vector size {mean_vector.shape} does not match "
                f"the prescribed dimension {dimension}."
            )
        if not precision_operator.shape == (dimension, dimension):
            raise ValueError(
                f"Precision operator shape {precision_operator.shape} does not match "
                f"the prescribed shape {(dimension, dimension)}."
            )
        if not covariance_operator.shape == (dimension, dimension):
            raise ValueError(
                f"Covariance operator shape {covariance_operator.shape} does not match "
                f"the prescribed shape {(dimension, dimension)}."
            )
        if not covariance_factorization.shape[0] == dimension:
            raise ValueError(
                f"Covariance factorization output dimension {covariance_factorization.shape[0]} "
                f"does not match the prescribed dimension {dimension}."
            )
        self.dimension = dimension
        self._mean_vector = mean_vector
        self._precision_operator = precision_operator
        self._covariance_operator = covariance_operator
        self._covariance_factorization = covariance_factorization
        self._prng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        if not parameter_vector.shape == self._mean_vector.shape:
            raise ValueError(
                f"Parameter vector shape {parameter_vector.shape} does not match "
                f"the prescribed shape {(self.dimension,)}."
            )
        difference_vector = parameter_vector - self._mean_vector
        cost = 0.5 * np.inner(difference_vector, self._precision_operator.apply(difference_vector))
        assert cost >= 0
        return cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if not parameter_vector.shape == (self.dimension,):
            raise ValueError(
                f"Parameter vector shape {parameter_vector.shape} does not match "
                f"the prescribed shape {(self.dimension,)}."
            )
        difference_vector = parameter_vector - self._mean_vector
        gradient = self._precision_operator.apply(difference_vector)
        assert gradient.shape == (self.dimension,)
        return gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if not direction_vector.shape == (self.dimension,):
            raise ValueError(
                f"Direction vector shape {direction_vector.shape} does not match "
                f"the prescribed shape {(self.dimension,)}."
            )
        hessian_vector_product = self._precision_operator.apply(direction_vector)
        assert hessian_vector_product.shape == (self.dimension,)
        return hessian_vector_product

    # ----------------------------------------------------------------------------------------------
    def generate_sample(self):
        random_vector_size = self._covariance_factorization.shape[1]
        random_vector = self._prng.normal(loc=0.0, scale=1.0, size=random_vector_size)
        sample_vector = self._covariance_factorization.apply(random_vector)
        assert sample_vector.shape == (self.dimension,)
        sample_vector += self._mean_vector
        return sample_vector
