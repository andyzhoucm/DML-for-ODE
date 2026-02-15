# DML-for-ODE

The repository includes four main scripts corresponding to the four experimental sections in the paper:

| File Name | Description | Paper Section |
| :--- | :--- | :--- |
| `test_for_unbiased.py` | **Bias Correction Experiment.** Compares the Iterative DML estimator against a Naive Non-linear Least Squares (NLS) estimator. Generates KDE plots showing the removal of confounding bias. | Experiment 1 |
| `test_for_asymptotic_normality.py` | **Inference Validity.** Verifies the asymptotic normality of the estimator and the coverage rate of the confidence intervals derived via the "Sandwich Formula" across different sample sizes ($N=1000, 4000, 8000$). | Experiment 2 |
| `test_for_high_dim.py` | **High-Dimensional Extension.** Validates the algorithm on a multi-dimensional parameter space ($\theta \in \mathbb{R}^3$) using unregularized OLS updates, confirming independence between parameter estimates. | Experiment 3 |
| `test_for_ode.py` | **Application to ODEs.** Applies the "Differentiation-then-Regression" strategy to a confounded Michaelis-Menten Kinetics system, demonstrating the recovery of physical parameters from trajectory data. | None |
| `test_for_integral_method.py` | **New: Trajectory-Based Integral DML.** Implements the advanced integral-based framework to handle time-varying confounders. Demonstrates superior robustness to noise and valid inference across sample sizes ($N=100, 500$) without numerical differentiation. | Experiment 4 |
