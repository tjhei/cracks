# Crack propagation

This is a finite element code based in deal.II to simulate crack propagation
in elastic and porous media.

Features:
- phase field approach for the crack location
- primal-dual active set method for the irreversibility constraint
- novel adaptive mesh refinement technique

This project was originally developed for the two papers mentioned below, but has been extended considerably since then. The original code versions are available as separate branches in this repository:

1. https://github.com/tjhei/cracks/tree/paper-2015 for

> T. Heister, M. F. Wheeler, T. Wick:
> A primal-dual active set method and predictor-corrector mesh adaptivity for computing fracture propagation using a phase-field approach.
> Comp. Meth. Appl. Mech. Engrg., Vol. 290 (2015), pp. 466-495
> http://dx.doi.org/10.1016/j.cma.2015.03.009

A preprint is available here: http://www.math.clemson.edu/~heister/preprints/HeWheWi15_CMAME_accepted.pdf

2. https://github.com/tjhei/cracks/tree/paper-2018-parallel for

> T. Heister, T. Wick:
> Parallel solution, adaptivity, computational convergence, and open-source code of 2d and 3d pressurized phase-field fracture problems
> ArXiv preprint
> https://arxiv.org/abs/1806.09924

# How to run

You need to install deal.II (see http://www.dealii.org) with external dependencies p4est and Trilinos. Then configure with:

  cmake -D DEAL_II_DIR=/your/dealii-installation/ .

Compile with:

  make

and finally run with:

  mpirun -n 2 ./cracks parameters_sneddon_2d.prm

If the code crashes with an exception "ExcIO", create an empty directory
called "output".

# Notes

The code is published under GPL v2 or newer.

Authors: Timo Heister, Thomas Wick.
