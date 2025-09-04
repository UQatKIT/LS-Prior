# LS-Prior [<img src="images/uq_logo.png" width="200" height="100" alt="UQ at KIT" align="right">](https://www.scc.kit.edu/forschung/uq.php)

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
:material-checkbox-marked-circle-outline: &nbsp; **Matrix-free Gaussian prior distribution via SPDE approach** <br>
:material-checkbox-marked-circle-outline: &nbsp; **FenicsX FEM backend, PETSc linear algebra and solvers** <br>
:material-checkbox-marked-circle-outline: &nbsp; **Fully MPI parallelized, fully modular** <br>
:material-checkbox-marked-circle-outline: &nbsp; **Easily embeddable into Bayesian inverse problem workflows**

## Installation

LS-Prior is currently managed as a [Pixi](https://pixi.sh/latest/) project, a conda package is planned. To
start using LS-Prior, simply run
```bash
pixi install
```
in the top-level directory.

## Documentation

#### Usage

Under Usage, we provide walkthroughs of the functionalities of LS-Prior.
The [Component Interface](usage/components.md) tutorial describes the flexible low-level interface
of the packae. [Builder Interface](usage/builder.md) demonstrates how to wuickly set up
pre-condigured prior distributions using LS-Prior's builder module.

#### API Reference

The API reference contains detailed explanations of all software components of Eikonax, and how to use them.

#### Examples

We provide [runnable examples](https://github.com/UQatKIT/LS-Prior/tree/main/examples) in our Github repository.

## Acknowledgement and License

LS-Prior is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT. It is distributed as free software under the [MIT License](https://choosealicense.com/licenses/mit/). Major portions of the implementation are inspired by the
prior implementation of the [hIPPYlib](https://hippylib.github.io/) library.
