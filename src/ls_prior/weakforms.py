from numbers import Real
from typing import Annotated

import ufl
from beartype.vale import Is


def generate_forms(
    trial_function: ufl.TrialFunction,
    test_function: ufl.TestFunction,
    kappa: Annotated[Real, Is[lambda x: x > 0]],
    tau: Annotated[Real, Is[lambda x: x > 0]],
    robin_const: float | None = None,
) -> tuple[ufl.Form, ufl.Form]:
    mass_matrix_form = ufl.inner(trial_function, test_function) * ufl.dx
    stiffness_matrix_form = ufl.inner(ufl.grad(trial_function), ufl.grad(test_function)) * ufl.dx
    spde_matrix_form = kappa**2 * tau * mass_matrix_form + tau * stiffness_matrix_form
    if robin_const is not None:
        robin_boundary_form = robin_const * ufl.inner(trial_function, test_function) * ufl.ds
        spde_matrix_form += robin_boundary_form

    return mass_matrix_form, spde_matrix_form
