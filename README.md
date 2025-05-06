# pfm-cracks: A parallel-adaptive framework for phase-field fracture propagation

This is a finite element code based on the finite element library deal.II to
simulate crack propagation in elastic and porous media.

Features:
- phase field approach for the crack location
- support for 2d and 3d computations
- primal-dual active set method for the irreversibility constraint
- novel adaptive mesh refinement technique

This project was originally developed for the papers mentioned below, but has
been extended considerably since then. Please cite these papers if you make
use of our work. Thank you!

The original code versions are available as separate branches in this
repository:

1. https://github.com/tjhei/cracks/tree/paper-2015 for

> T. Heister, M. F. Wheeler, T. Wick:
> A primal-dual active set method and predictor-corrector mesh adaptivity for computing fracture propagation using a phase-field approach.
> Comp. Meth. Appl. Mech. Engrg., Vol. 290 (2015), pp. 466-495
> http://dx.doi.org/10.1016/j.cma.2015.03.009

A preprint is available here: http://www.math.clemson.edu/~heister/preprints/HeWheWi15_CMAME_accepted.pdf

2. https://github.com/tjhei/cracks/tree/paper-2018-parallel for

> T. Heister, T. Wick:
> Parallel solution, adaptivity, computational convergence, and open-source code of 2d and 3d pressurized phase-field fracture problems
> Proc. Appl. Math. Mech., 2018, e201800353
> https://doi.org/10.1002/pamm.201800353

A preprint is available here: https://arxiv.org/abs/1806.09924

# How to run

You need to install deal.II (see http://www.dealii.org) with external
dependencies p4est and Trilinos. Minimum required version for deal.II is 9.5.
Then configure with:

```
  cmake -D DEAL_II_DIR=/your/dealii-installation/ .
```
Compile with:
```
  make
```
and finally run with:
```
  mpirun -n 2 ./cracks parameters_sneddon_2d.prm
```

# Notes

The code is published under GPL v2 or newer.

Authors: Timo Heister, Thomas Wick.
