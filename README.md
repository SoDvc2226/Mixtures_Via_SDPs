# Mixtures Closest to a Given Measure: A Semidefinite Programming Approach

We provide the `code` and the `data` used for the numerical experiments in:

[1] Srećko Ðurašinović, Jean-Bernard Lasserre, Victor Magron. "*Mixtures Closest to a Given Measure: A Semidefinite Programming Approach*" arXiv, 2025. 

The `code/` directory includes all the algorithms and functions required to run the notebooks and evaluate various benchmarks. The implementation is written in [Julia](https://julialang.org).

The `data/` directory contains all the datasets necessary to replicate the experimental results presented in the paper. It includes Polynomial Optimization (POP) instances, initial relaxation bounds, known local solutions, and logs from the SDP solver. For detailed descriptions, refer to the accompanying `_EXPLANATION.txt` files. All data is provided in `.jld2` format.


## Getting started

The code requires Julia  1.10.5+ version and, among others, following packages/libraries:

- Linear Algebra
- Random
- IOLogging
- SparseArrays
- IJulia and Jupyter (for running the notebooks)
- DynamicPolynomials
- [TSSOS](https://github.com/wangjie212/TSSOS/) (polynomial optimization library based on the sparsity adapted Moment-SOS hierarchies)
  
Our optimization problems were solved using the SDP solver [Mosek](https://www.mosek.com/) (licence required).

All the experiments can be reproduced by running `code/notebook_name.ipynb`, where `notebook_name` is reflective of the experimental tables from the paper. All instructions on how to run the experiments are detailed in the notebooks and associated `.jl` function files.
