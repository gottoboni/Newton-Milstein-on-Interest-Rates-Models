# Newton--Milstein Scheme for Short-Rate Models: Numerical Experiments

This repository contains the numerical experiments accompanying the semester paper

> **Newton--Milstein Methods Applied to Short-Rate Models**
> Giovanni Ottoboni, ETH Zurich, Department of Mathematics, 2026.

The paper studies the Newton--Milstein discretization scheme (Amano, 2016) applied to stochastic differential equations arising in interest rate modelling. After $n$ Newton corrections on a grid of step size $h$, the scheme achieves a pathwise error of order $h^{2^n/p}$ in probability under a fixed regularity condition, independently of the number of iterations performed. The experiments below verify this theoretical prediction on zero-coupon bond pricing problems under the Vasicek and Cox--Ingersoll--Ross (CIR) models.

## Repository Structure

```
numerical_experiments/
    bond_pricing_experiment.py        # Experiment 1: convergence study
    efficiency_experiment.py          # Experiment 2: theoretical efficiency frontier
    efficiency_wallclock_experiment.py # Experiment 3: wall-clock efficiency frontier
    figures/                          # Generated plots (PNG)
```

## Experiments

### Experiment 1: Zero-Coupon Bond Pricing Convergence (`bond_pricing_experiment.py`)

Compares Monte Carlo estimates of the zero-coupon bond price $P(0,T) = \mathbb{E}[\exp(-\int_0^T r_s\, ds)]$ against exact analytical formulas for the Vasicek and CIR models. Four discretization schemes are compared across a range of step counts $m \in \{8, 16, 32, 64, 128, 256, 512\}$:

- Euler--Maruyama (strong order 1/2)
- Classical Milstein (strong order 1)
- Modified Milstein (Amano, uniform order 1)
- Newton--Milstein with $n = 1, 2, 3$ Newton corrections

All schemes share the same Brownian increments (common random numbers) for a fair comparison. The script reports bias and RMSE as functions of the step count and produces log-log convergence plots.

### Experiment 2: Theoretical Efficiency Frontier (`efficiency_experiment.py`)

Plots accuracy (bias) against theoretical computational cost on the CIR model. The cost model counts the number of full passes over the time grid:

- Euler--Maruyama / Milstein / Modified Milstein: 1 pass, cost $= m$
- Newton--Milstein with $n$ corrections: $(n+1)$ passes, cost $= (n+1) \cdot m$

This produces a log-log "efficiency frontier" showing that a single Newton correction typically reaches a given bias target at lower total cost than refining the classical grid.

### Experiment 3: Wall-Clock Efficiency Frontier (`efficiency_wallclock_experiment.py`)

Measures actual wall-clock time (averaged over multiple runs) rather than theoretical operation counts. This captures real per-step cost differences between the schemes (e.g., the exponential integrating factor evaluation in the Newton iteration). The script produces bias-vs-time plots on a log-log scale and a table reporting, for several target bias levels, the minimum step count and wall-clock time each scheme requires.

## Model Parameters

Both models use calibrations following Brigo and Mercurio (2006):

| Parameter | Vasicek | CIR |
|-----------|---------|-----|
| $\kappa$ (mean-reversion speed) | 0.5 | 0.5 |
| $\theta$ (long-run mean) | 0.05 | 0.05 |
| $\sigma$ (volatility) | 0.02 | 0.1 |
| $r_0$ (initial rate) | 0.03 | 0.03 |
| $T$ (maturity) | 5.0 | 5.0 |

The CIR parameters satisfy the Feller condition $2\kappa\theta \geq \sigma^2$.

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- Matplotlib

Install with:

```bash
pip install numpy scipy matplotlib
```

## Usage

Run each experiment from the `numerical_experiments/` directory:

```bash
cd numerical_experiments

# Experiment 1: convergence study (Vasicek + CIR)
python bond_pricing_experiment.py

# Experiment 2: theoretical efficiency frontier (CIR only)
python efficiency_experiment.py

# Experiment 3: wall-clock efficiency frontier (CIR only)
python efficiency_wallclock_experiment.py
```

All figures are saved to `numerical_experiments/figures/`.

Note: `efficiency_experiment.py` and `efficiency_wallclock_experiment.py` import shared simulation infrastructure from `bond_pricing_experiment.py`, so all three files must reside in the same directory.

## References

- Amano, K. (2016). Newton--Milstein scheme.
- Brigo, D. and Mercurio, F. (2006). *Interest Rate Models -- Theory and Practice*. Springer.
- Kloeden, P. E. and Platen, E. (1992). *Numerical Solution of Stochastic Differential Equations*. Springer.

## Author

Giovanni Ottoboni
Department of Mathematics, ETH Zurich
gottoboni23@gmail.com
