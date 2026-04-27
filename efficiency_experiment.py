"""
=============================================================================
Computational Efficiency Frontier Experiment
=============================================================================

This script compares the accuracy-vs-cost trade-off of different discretization
schemes applied to zero-coupon bond pricing under the CIR short-rate model.

For each scheme and each step count m, we:
    1. Simulate M paths and compute the Monte Carlo bond price bias.
    2. Compute the theoretical operation count as:
           cost = (number of passes over the grid) * m
       where:
           - Euler-Maruyama:        1 pass   => cost = m
           - Classical Milstein:    1 pass   => cost = m
           - Modified Milstein:     1 pass   => cost = m
           - Newton-Milstein (n=k): (k+1) passes => cost = (k+1) * m
    3. Plot bias vs theoretical cost on a log-log scale.

The Vasicek model is excluded because its constant diffusion coefficient
makes all schemes collapse to the same method (see Section 6.1.4).

References:
    - Amano, K. (2016). Newton-Milstein scheme.
    - Brigo, D. and Mercurio, F. (2006). Interest Rate Models.
    - Kloeden, P. E. and Platen, E. (1992). Numerical Solution of SDEs.

Author: Giovanni Ottoboni (with Claude assistance)
Date:   March 2026
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os

# ============================================================================
# Import shared infrastructure from the bond pricing experiment
# ============================================================================

from bond_pricing_experiment import (
    CIR_PARAMS,
    cir_bond_price,
    generate_brownian_increments,
    simulate_cir_euler,
    simulate_cir_milstein,
    simulate_cir_modified_milstein,
    simulate_cir_newton,
    compute_bond_price_mc,
)

# ============================================================================
# Configuration
# ============================================================================

# Random seed for reproducibility
SEED = 42

# Number of Monte Carlo paths (50k balances accuracy with memory constraints)
NUM_PATHS = 50_000

# Step counts to test: from coarse (m=16) to fine (m=512).
# We start at m=16 to stay in the regime where the Newton theory applies
# (m >= 20 for CIR with these parameters, see Section 6.1.4).
STEP_COUNTS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# Output directory for figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# Theoretical cost model
# ============================================================================

def theoretical_cost(num_steps, num_passes):
    """
    Compute the theoretical operation count for a scheme.

    Each pass over the time grid costs O(m) operations (one evaluation of
    the drift, diffusion, and any correction terms per step). The total
    cost is therefore proportional to the number of passes times m.

    Parameters
    ----------
    num_steps  : int -- number of time steps m
    num_passes : int -- number of full passes over the grid
                        (1 for Euler/Milstein, n+1 for Newton-Milstein)

    Returns
    -------
    int : theoretical operation count proportional to num_passes * m
    """
    return num_passes * num_steps


# ============================================================================
# Scheme registry with pass counts
# ============================================================================

# Each entry: (display_name, simulator_function, num_passes, needs_J, newton_steps)
# needs_J indicates whether the scheme requires the integrated BM increments J.
# newton_steps is the number of Newton corrections (only for Newton-Milstein).
SCHEMES = [
    ("Euler-Maruyama",        simulate_cir_euler,             1, False, None),
    ("Milstein",              simulate_cir_milstein,          1, False, None),
    ("Modified Milstein",     simulate_cir_modified_milstein, 1, True,  None),
    ("Newton-Milstein (n=1)", simulate_cir_newton,            2, True,  1),
    ("Newton-Milstein (n=2)", simulate_cir_newton,            3, True,  2),
    ("Newton-Milstein (n=3)", simulate_cir_newton,            4, True,  3),
]


# ============================================================================
# Run the experiment
# ============================================================================

def run_efficiency_experiment(params, analytical_price, step_counts, num_paths, seed):
    """
    For each scheme and step count, compute the bond price bias and
    the theoretical operation count.

    Parameters
    ----------
    params           : dict  -- CIR model parameters
    analytical_price : float -- exact bond price P(0, T)
    step_counts      : list of int -- values of m to test
    num_paths        : int   -- number of Monte Carlo paths
    seed             : int   -- random seed

    Returns
    -------
    results : dict -- {scheme_name: {"bias": [...], "cost": [...]}}
              where each list has one entry per step count
    """
    maturity = params["T"]

    # Initialize storage for each scheme
    results = {
        name: {"bias": [], "cost": []}
        for name, _, _, _, _ in SCHEMES
    }

    for m in step_counts:
        step_size = maturity / m
        print(f"  m = {m:5d} (h = {step_size:.5f}) ... ", end="", flush=True)

        # Generate common Brownian increments shared by all schemes
        rng = np.random.default_rng(seed)
        dW, J = generate_brownian_increments(num_paths, m, step_size, rng)

        for scheme_name, simulator, num_passes, needs_J, newton_steps in SCHEMES:
            # Call the simulator with the appropriate arguments
            if newton_steps is not None:
                paths = simulator(params, m, dW, J, newton_steps)
            elif needs_J:
                paths = simulator(params, m, dW, J)
            else:
                paths = simulator(params, m, dW)

            # Compute Monte Carlo bond price and bias
            mc_price, _, _ = compute_bond_price_mc(paths, maturity)
            bias = abs(mc_price - analytical_price)

            # Compute theoretical cost
            cost = theoretical_cost(m, num_passes)

            results[scheme_name]["bias"].append(bias)
            results[scheme_name]["cost"].append(cost)

        print("done")

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_efficiency_frontier(results, fig_dir):
    """
    Create a log-log plot of bias vs theoretical operation count
    for all schemes.

    Parameters
    ----------
    results : dict -- output of run_efficiency_experiment
    fig_dir : str  -- directory to save the figure
    """
    # Visual style for each scheme: color, marker, linestyle
    style_map = {
        "Euler-Maruyama":        {"color": "#1f77b4", "marker": "o",  "linestyle": "-"},
        "Milstein":              {"color": "#ff7f0e", "marker": "s",  "linestyle": "-"},
        "Modified Milstein":     {"color": "#2ca02c", "marker": "^",  "linestyle": "-"},
        "Newton-Milstein (n=1)": {"color": "#d62728", "marker": "D",  "linestyle": "--"},
        "Newton-Milstein (n=2)": {"color": "#9467bd", "marker": "v",  "linestyle": "--"},
        "Newton-Milstein (n=3)": {"color": "#8c564b", "marker": "P",  "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheme_name, data in results.items():
        style = style_map.get(
            scheme_name,
            {"color": "gray", "marker": "x", "linestyle": "-"}
        )
        cost_values = data["cost"]
        bias_values = data["bias"]

        # Replace any zero bias with a tiny value for log-log plotting
        bias_plot = [max(b, 1e-16) for b in bias_values]

        ax.loglog(
            cost_values, bias_plot,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=6,
            label=scheme_name,
        )

    # Add reference slope lines for cost^{-gamma} scaling.
    # For a scheme of order gamma, bias ~ h^gamma = (T/m)^gamma ~ m^{-gamma}.
    # Since cost ~ c * m for c passes, we have m ~ cost/c, so
    # bias ~ (cost/c)^{-gamma} ~ cost^{-gamma}.
    # The constant c only shifts the line horizontally on log-log scale,
    # so the slope is -gamma regardless of the number of passes.
    cost_ref = np.array([20, 5000])

    # Order 1/2 reference (Euler-Maruyama scaling)
    scale_half = 0.02  # vertical positioning constant
    ref_half = scale_half * cost_ref**(-0.5)
    ax.loglog(cost_ref, ref_half, "k:", linewidth=0.8, alpha=0.5,
              label=r"$O(\mathrm{cost}^{-1/2})$")

    # Order 1 reference (Milstein scaling)
    scale_one = 0.1  # vertical positioning constant
    ref_one = scale_one * cost_ref**(-1.0)
    ax.loglog(cost_ref, ref_one, "k-.", linewidth=0.8, alpha=0.5,
              label=r"$O(\mathrm{cost}^{-1})$")

    ax.set_xlabel("Theoretical operation count (passes $\\times$ $m$)", fontsize=12)
    ax.set_ylabel(r"$|\mathrm{Bias}|$", fontsize=12)
    ax.set_title("CIR Model: Accuracy vs Computational Cost", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    filepath = os.path.join(fig_dir, "efficiency_frontier_cir.png")
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """
    Run the computational efficiency frontier experiment for the CIR model.
    """
    print("=" * 70)
    print("  Computational Efficiency Frontier Experiment")
    print("  CIR Model: Bias vs Theoretical Operation Count")
    print("=" * 70)

    params = CIR_PARAMS
    analytical_price = cir_bond_price(
        **{k: params[k] for k in ["kappa", "theta", "sigma", "r0", "T"]}
    )
    print(f"  Analytical P(0, {params['T']}) = {analytical_price:.10f}")

    # Verify Feller condition
    feller_lhs = 2.0 * params["kappa"] * params["theta"]
    sigma_squared = params["sigma"] ** 2
    print(f"  Feller condition: 2*kappa*theta = {feller_lhs:.4f} "
          f">= sigma^2 = {sigma_squared:.4f} "
          f"=> {'SATISFIED' if feller_lhs >= sigma_squared else 'VIOLATED'}")

    print(f"\n  Running {len(STEP_COUNTS)} step counts "
          f"x {len(SCHEMES)} schemes "
          f"x {NUM_PATHS} paths ...\n")

    results = run_efficiency_experiment(
        params, analytical_price, STEP_COUNTS, NUM_PATHS, SEED
    )

    # Print a summary table to the console
    print(f"\n{'='*90}")
    print(f"  CIR Model -- Bias vs Theoretical Cost")
    print(f"{'='*90}")
    header = f"{'m':>6s} {'h':>8s}"
    for name, _, num_passes, _, _ in SCHEMES:
        short = name.replace("Newton-Milstein", "NM").replace("Modified Milstein", "Mod.Mil.")
        header += f"  {short:>16s}"
    print(f"\n  {header}")
    print(f"  {'-' * len(header)}")

    for j, m in enumerate(STEP_COUNTS):
        step_size = params["T"] / m
        row = f"  {m:6d} {step_size:8.4f}"
        for name, _, _, _, _ in SCHEMES:
            bias = results[name]["bias"][j]
            cost = results[name]["cost"][j]
            row += f"  {bias:.2e} ({cost:5d})"
        print(row)

    print()

    # Generate the efficiency frontier plot
    plot_efficiency_frontier(results, FIG_DIR)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
