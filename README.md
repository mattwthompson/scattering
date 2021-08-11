# scattering
[![CI with Anaconda](https://github.com/mattwthompson/scattering/actions/workflows/main.yml/badge.svg)](https://github.com/mattwthompson/scattering/actions/workflows/main.yml)

Functions for analyzing molecular simulations according to scattering experiments.  Currently, functions are available to compute:
- Static structure factor
- Van Hove Correlation Function

Currently, the majority of form factors are esimated from the atomic
number.  Specific form factors can additionally be used for water,
taken from `DOI: 10.1126/sciadv.1603079`.

## Requirements
Package requirements are listed in `environment.yml` (Development environment contained in `environment-dev.yml`.
A conda environment with all dependencies for this package can be installed with `conda create -f environment.yml` or `conda create -f environment-dev.yml`.

## Installation
Installation is currently only available from source.
Package can be cloned from GitHub with the following command:
```
https://github.com/mattwthompson/scattering.git
```
Once cloned, the package can be installed through Pip:
```
cd scattering
pip install -e .
```

## Van Hove Function
Add details later
