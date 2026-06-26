# Research Notes

## Project Aim

This project compares Tikhonov and total variation regularization for noisy
linearized EIT reconstructions using controlled synthetic phantoms.

## Numerical Setup

- Domain: unit square.
- Mesh: 40 by 40 square grid split into triangular elements.
- Electrodes: 16 boundary electrodes.
- Drive pattern: adjacent current injection.
- Measurements: adjacent electrode voltage differences.
- Linearization point: homogeneous conductivity `sigma0 = 1`.

## Phantoms

- Phantom A uses a circular piecewise-constant inclusion with elevated
  conductivity.
- Phantom B uses a smooth Gaussian inclusion centered in the domain.

## Reconstruction Methods

- Tikhonov reconstruction solves the regularized least-squares problem using a
  dual system whose dimension matches the data vector.
- TV reconstruction uses a graph difference matrix over adjacent triangles and a
  matrix-free primal-dual interior-point iteration.

## Evaluation

The scripts evaluate relative reconstruction error against the known
triangle-wise conductivity perturbation. Experiments are repeated across
relative noise levels of 0%, 1%, 3%, 5%, and 10%.
