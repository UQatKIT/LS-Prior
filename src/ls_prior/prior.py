"""."""

import numpy as np

from . import components


# ==================================================================================================
class Prior:
    """."""

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        precision_operator: components.InterfaceComponent,
        covariance_operator: components.InterfaceComponent,
        covariance_factorization: components.InterfaceComponent,
        seed: int,
    ) -> None:
        """_summary_.

        Args:
            mean_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): _description_
            precision_operator (components.InterfaceComponent): _description_
            covariance_operator (components.InterfaceComponent): _description_
            covariance_factorization (components.InterfaceComponent): _description_
            seed (int): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """
        mean_vector_dim = mean_vector.shape[0]
        if not precision_operator.shape == (mean_vector_dim, mean_vector_dim):
            raise ValueError(
                f"Precision operator shape {precision_operator.shape} does not match "
                f"the mean vector dimension {mean_vector}."
            )
        if not covariance_operator.shape == (mean_vector_dim, mean_vector_dim):
            raise ValueError(
                f"Covariance operator shape {covariance_operator.shape} does not match "
                f"the mean vector dimension {mean_vector_dim}."
            )
        if not covariance_factorization.shape[0] == mean_vector_dim:
            raise ValueError(
                f"Covariance factorization output dimension {covariance_factorization.shape[0]} "
                f"does not match the mean vector dimension {mean_vector_dim}."
            )
        self._mean_vector = mean_vector
        self._precision_operator = precision_operator
        self._covariance_operator = covariance_operator
        self._covariance_factorization = covariance_factorization
        self._prng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        """_summary_.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): _description_

        Returns:
            float: _description_
        """
        self._check_input_dimension(parameter_vector)
        difference_vector = parameter_vector - self._mean_vector
        cost = 0.5 * np.inner(difference_vector, self._precision_operator.apply(difference_vector))
        assert cost >= 0, f"Cost needs to be non-negative, but is {cost}."
        return cost

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """_summary_.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): _description_

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: _description_
        """
        self._check_input_dimension(parameter_vector)
        difference_vector = parameter_vector - self._mean_vector
        gradient = self._precision_operator.apply(difference_vector)
        self._check_output_dimension(gradient)
        return gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """_summary_.

        Args:
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): _description_

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: _description_
        """
        self._check_input_dimension(direction_vector)
        hessian_vector_product = self._precision_operator.apply(direction_vector)
        self._check_output_dimension(hessian_vector_product)
        return hessian_vector_product

    # ----------------------------------------------------------------------------------------------
    def generate_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """_summary_.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: _description_
        """
        random_vector_size = self._covariance_factorization.shape[1]
        random_vector = self._prng.normal(loc=0.0, scale=1.0, size=random_vector_size)
        sample_vector = self._covariance_factorization.apply(random_vector)
        self._check_output_dimension(sample_vector)
        sample_vector += self._mean_vector
        return sample_vector

    # ----------------------------------------------------------------------------------------------
    def _check_input_dimension(self, vector: np.ndarray) -> None:
        """_summary_.

        Args:
            vector (np.ndarray): _description_

        Raises:
            ValueError: _description_
        """
        if not vector.shape == self._mean_vector.shape:
            raise ValueError(
                f"Vector shape {vector.shape} does not match "
                f"the mean vector shape {self._mean_vector.shape}."
            )

    # ----------------------------------------------------------------------------------------------
    def _check_output_dimension(self, vector: np.ndarray) -> None:
        """_summary_.

        Args:
            vector (np.ndarray): _description_
        """
        assert vector.shape == self._mean_vector.shape, (
            f"Sample vector shape {vector.shape} does not match "
            f"mean vector shape {self._mean_vector.shape}."
        )
