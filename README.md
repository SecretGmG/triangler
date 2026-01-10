# Numerical Evaluation of the Scalar Triangle with Complex Masses

This repository contains **all code used to generate the numerical results and figures** presented in the thesis *Regularising complex-valued thresholds in numerical integration of loop integrals*

The focus is on the **numerical evaluation of the scalar triangle integral** in the Cross-Free Family representation, including **threshold subtraction for complex internal masses**.


## Overview

The repository provides:

Generation and compilation of the following expressions using the symbolica library is implemented in `integrand_builder.py`

* The CFF representation of the scalar triangle
* Counterterms for threshold subtraction for E-surfaces, even with complex thresholds
* The integrated counterterm, scaled by a mask, such that it can be integrated alongside the threshold subtracted integrand

The integrand builder additionally supports toggling of the following features

* Threshold subtraction on/off
* Centered vs sliver subtraction regions
* Constant vs width-dependent subtraction masks
* Per-sample vs minimum-value E-surface existence conditions
* Gaussian vs step-function smearing of integrated counterterms

These expressions can be compiled into optimized assembly. `wrapped_eval.py` implements utility functions for building and loading such compiled evaluators.

To facilitate complex integration with symbolica a custom integration routine is needed, which is implemented in `integrator.py`. It splits the integration of the real and complex parts, while still using the absolute value of the integrand to update the grid sampler of the **VEGAS** implementation of symbolica.

The `TriangleIntegrandContext` class, which is implemented in `context.py` implements methods to define the hyperparameters and parameters for the integration. This includes the external momenta and (complex valued) masses of the propagators.


## Dependencies

### Symbolica

This project **requires Symbolica** for:

* Compiled symbolic expressions
* VEGAS integration

Follow the installation instructions at:
[https://symbolica.io/](https://symbolica.io/)

### Reference Values (OneLOop)

To validate the numerical integration, results are compared against **OneLOop**.

This repo expects the separate package **OneLOopBridge** which proves a Python and Rust API to the original Fortran code.

* **OneLOopBridge**
  [https://github.com/SecretGmG/OneLOopBridge](https://github.com/SecretGmG/OneLOopBridge)

Install it according to its README.


## License

MIT License