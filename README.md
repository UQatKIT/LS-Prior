![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FUQatKIT%2FLS-Prior%2Fmain%2Fpyproject.toml)
![License](https://img.shields.io/github/license/UQatKIT/LS-Prior)
![Beartype](https://github.com/beartype/beartype-assets/raw/main/badge/bear-ified.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# Large-Scale Prior Fields via SPDEs

LS-Prior is a python package for large-sace Gaussian prior measures The implementation is based on the SPDE approach to Gaussian Matérn fields popularized by
[Lindgren et al.](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.00777.x).
This work introduces the generation of random field via transformation of white noise $W$ by an elliptic operator, resembled by equations of the form

$$
\begin{equation*}
    \tau (\kappa^2 - \Delta)^{\nu/2} = W
\end{equation*}
$$

The discrete representation of prior distributions in LS-Prior on computationl meshes is realized through the finite element method with [FenicsX](https://fenicsproject.org/). The package is
intended to be a more modern, modular, and flexible substitute for the prior component in the [hIPPYlib](https://hippylib.github.io/) library. In particular, it can be combined with any other component in a inverse problem workflow, without explicit reliance on the internal FEM
representation.

### Key Features
- **Matrix-free Gaussian prior distribution via SPDE approach**
- **FenicsX FEM backend, PETSc linear algebra and solvers**
- **Fully MPI parallelized, fully modular**
- **Easily embeddable into Bayesian inverse problem workflows**

## Getting Started

LS-Prior is currently managed as a [Pixi](https://pixi.sh/latest/) project, a conda package is planned. To
start using LS-Prior, simply run
```bash
pixi install
```
in the top-level directory.

## Documentation

The [documentation](https://uqatkit.github.io/LS-Prior/) provides further information regarding usage, technical setup and API. Alternatively, you can check out the notebooks under [`examples`](https://github.com/UQatKIT/LS-Prior/tree/main/examples)

## Acknowledgement and License

LS-Prior is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT. It is distributed as free software under the [MIT License](https://choosealicense.com/licenses/mit/). Major portions of the implementation are inspired by the
prior implementation of the [hIPPYlib](https://hippylib.github.io/) library.