# Crack propagation

This is a finite element code based in deal.II to simulate crack propagation
in elastic and porous media.

Features:
- phase field approach for the crack location
- primal-dual active set method for the irreversibility constraint
- novel adaptive mesh refinement technique

This is the example program to accompany the paper

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
