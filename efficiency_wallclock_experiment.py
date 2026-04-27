"""
=============================================================================
Computational Efficiency Frontier -- Wall-Clock Timing Experiment
=============================================================================

This script measures the actual wall-clock time of each discretization scheme
applied to the CIR zero-coupon bond pricing problem, and plots accuracy
(bias) against measured computation time.

Unlike the theoretical cost model in efficiency_experiment.py (which counts
passes x m), this experiment captures real per-step cost differences:
    - Euler-Maruyama and Milstein: lightweight arithmetic per step
    - Modified Milstein: additional Ma * J correction
    - Newton-Milstein: full Newton iteration with exponential integrating
      factor evaluation, repeated (n+1) times over the grid

For each scheme and step count m, we:
    1. Generate Brownian increments (timed separately, not counted).
    2. Time the path simulation (averaged over TIMING_REPEATS runs).
    3. Compute the bond price bias against the analytical CIR formula.
    4. Record (measured_time, bias) pairs.

We then:
    - Plot bias vs wall-clock time on a log-log scale.
    - Produce a table: for target bias levels, report the minimum m
      and wall-clock time each scheme needs to achieve that accuracy.

The Vasicek model is excluded because its constant diffusion coefficient
makes all schemes collapse to the same method (see Section 6.1.3).

References:
    - Amano, K. (2016). Newton-Milstein scheme.
    - Brigo, D. and Mercurio, F. (2006). Interest Rate Models.

Author: Giovanni Ottoboni (with Claude assistance)
Date:   March 2026
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
import time

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

# Random seed for reproducibility of bias measurements
SEED = 42

# Number of Monte Carlo paths
NUM_PATHS = 50_000

# Number of times to repeat each simulation for stable timing
TIMING_REPEATS = 5

# Step counts to test: from coarse (m=16) to fine (m=512).
# We start at m=16 to stay in the regime where the Newton theory
# applies (m >= 20 for CIR with these parameters, see Section 6.1.4).
STEP_COUNTS = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

# Target bias levels for the "required steps" table
TARGET_BIAS_LEVELS = [1e-3, 5e-4, 2e-4]

# Output directory for figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# Scheme registry
# ============================================================================

# Each entry: (display_name, simulator, needs_J, newton_steps)
# needs_J: whether the scheme needs the integrated BM increments J
# newton_steps: number of Newton corrections (None for non-Newton schemes)
SCHEMES = [
    ("Euler-Maruyama",        simulate_cir_euler,             False, None),
    ("Milstein",              simulate_cir_milstein,          False, None),
    ("Modified Milstein",     simulate_cir_modified_milstein, True,  None),
    ("Newton-Milstein (n=1)", simulate_cir_newton,            True,  1),
    ("Newton-Milstein (n=2)", simulate_cir_newton,            True,  2),
    ("Newton-Milstein (n=3)", simulate_cir_newton,            True,  3),
]


# ============================================================================
# Timing utility
# ============================================================================

def time_simulation(simulator, params, num_steps, dW, J, needs_J, newton_steps,
                    num_repeats):
    """
    Measure the wall-clock time of a single scheme simulation,
    averaged over multiple repeats for stability.

    We time ONLY the simulation (path generation), not the Brownian
    increment generation or the bond price computation, since those
    costs are identical across schemes.

    Parameters
    ----------
    simulator    : callable -- the scheme's simulation function
    params       : dict     -- CIR model parameters
    num_steps    : int      -- number of time steps m
    dW           : ndarray  -- Brownian increments (num_paths, num_steps)
    J            : ndarray  -- integrated BM increments (num_paths, num_steps)
    needs_J      : bool     -- whether the scheme uses J
    newton_steps : int or None -- number of Newton corrections
    num_repeats  : int      -- number of timing repeats

    Returns
    -------
    avg_time_seconds : float   -- average wall-clock time in seconds
    paths            : ndarray -- simulated paths from the last run
                                  (used to compute bias afterward)
    """
    elapsed_times = []
    paths = None

    for _ in range(num_repeats):
        # Call the simulator with the correct signature
        if newton_steps is not None:
            start = time.perf_counter()
            paths = simulator(params, num_steps, dW, J, newton_steps)
            end = time.perf_counter()
        elif needs_J:
            start = time.perf_counter()
            paths = simulator(params, num_steps, dW, J)
            end = time.perf_counter()
        else:
            start = time.perf_counter()
            paths = simulator(params, num_steps, dW)
            end = time.perf_counter()

        elapsed_times.append(end - start)

    # Use median rather than mean to reduce sensitivity to outliers
    # (e.g., garbage collection spikes)
    avg_time_seconds = np.median(elapsed_times)

    return avg_time_seconds, paths


# ============================================================================
# Run the experiment
# ============================================================================

def run_wallclock_experiment(params, analytical_price, step_counts, num_paths,
                             seed, timing_repeats):
    """
    For each scheme and step count, measure wall-clock simulation time
    and compute the bond price bias.

    Parameters
    ----------
    params           : dict       -- CIR model parameters
    analytical_price : float      -- exact bond price P(0, T)
    step_counts      : list[int]  -- values of m to test
    num_paths        : int        -- number of Monte Carlo paths
    seed             : int        -- random seed
    timing_repeats   : int        -- number of repeats per timing measurement

    Returns
    -------
    results : dict -- {scheme_name: {"bias": [...], "time": [...], "m": [...]}}
              where each list has one entry per step count
    """
    maturity = params["T"]

    # Initialize storage for each scheme
    results = {
        name: {"bias": [], "time": [], "m": []}
        for name, _, _, _ in SCHEMES
    }

    for m in step_counts:
        step_size = maturity / m
        print(f"  m = {m:5d} (h = {step_size:.5f}) ... ", end="", flush=True)

        # Generate common Brownian increments shared by all schemes.
        # This is done OUTSIDE the timing loop so that increment generation
        # cost does not pollute the scheme timing.
        rng = np.random.default_rng(seed)
        dW, J = generate_brownian_increments(num_paths, m, step_size, rng)

        for scheme_name, simulator, needs_J, newton_steps in SCHEMES:
            # Time the simulation and get the paths from the last run
            avg_time, paths = time_simulation(
                simulator, params, m, dW, J, needs_J, newton_steps,
                timing_repeats
            )

            # Compute Monte Carlo bond price and bias
            mc_price, _, _ = compute_bond_price_mc(paths, maturity)
            bias = abs(mc_price - analytical_price)

            results[scheme_name]["bias"].append(bias)
            results[scheme_name]["time"].append(avg_time)
            results[scheme_name]["m"].append(m)

        print("done")

    return results


# ============================================================================
# Interpolation: find required m for a target bias
# ============================================================================

def find_required_steps_for_target_bias(results, target_bias):
    """
    For each scheme, find the minimum step count m and the corresponding
    wall-clock time needed to achieve a bias at or below the target level.

    We use log-linear interpolation on the (m, bias) data to estimate
    the crossing point. If no tested m achieves the target, we report None.

    Parameters
    ----------
    results     : dict  -- output of run_wallclock_experiment
    target_bias : float -- target bias level

    Returns
    -------
    table : dict -- {scheme_name: {"m_required": int or None,
                                    "time_required": float or None,
                                    "bias_achieved": float or None}}
    """
    table = {}

    for scheme_name, data in results.items():
        m_values = np.array(data["m"])
        bias_values = np.array(data["bias"])
        time_values = np.array(data["time"])

        # Find the first m where bias <= target
        # We look for where the bias crosses below the target.
        # Because bias is noisy (Monte Carlo), we use the first m
        # where bias is at or below the target.
        achieved_indices = np.where(bias_values <= target_bias)[0]

        if len(achieved_indices) == 0:
            # No tested m achieves the target bias
            table[scheme_name] = {
                "m_required": None,
                "time_required": None,
                "bias_achieved": None,
            }
        else:
            # Take the first m that achieves the target
            first_idx = achieved_indices[0]
            table[scheme_name] = {
                "m_required": int(m_values[first_idx]),
                "time_required": time_values[first_idx],
                "bias_achieved": bias_values[first_idx],
            }

    return table


# ============================================================================
# Plotting: efficiency frontier (bias vs wall-clock time)
# ============================================================================

def plot_wallclock_efficiency(results, fig_dir):
    """
    Create a log-log plot of bond price bias versus measured wall-clock
    simulation time for all schemes under the CIR model.

    Parameters
    ----------
    results : dict -- output of run_wallclock_experiment
    fig_dir : str  -- directory to save the figure
    """
    # Visual style for each scheme: color, marker, linestyle
    style_map = {
        "Euler-Maruyama":        {"color": "#1f77b4", "marker": "o",  "ls": "-"},
        "Milstein":              {"color": "#ff7f0e", "marker": "s",  "ls": "-"},
        "Modified Milstein":     {"color": "#2ca02c", "marker": "^",  "ls": "-"},
        "Newton-Milstein (n=1)": {"color": "#d62728", "marker": "D",  "ls": "--"},
        "Newton-Milstein (n=2)": {"color": "#9467bd", "marker": "v",  "ls": "--"},
        "Newton-Milstein (n=3)": {"color": "#8c564b", "marker": "P",  "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheme_name, data in results.items():
        style = style_map.get(
            scheme_name,
            {"color": "gray", "marker": "x", "ls": "-"}
        )
        time_values = data["time"]
        bias_values = data["bias"]

        # Replace any zero bias with a tiny value for log-log plotting
        bias_plot = [max(b, 1e-16) for b in bias_values]

        ax.loglog(
            time_values, bias_plot,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=1.5,
            markersize=6,
            label=scheme_name,
        )

    ax.set_xlabel("Wall-clock simulation time (seconds)", fontsize=12)
    ax.set_ylabel(r"$|\mathrm{Bias}|$", fontsize=12)
    ax.set_title("CIR Model: Accuracy vs Wall-Clock Time", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    filepath = os.path.join(fig_dir, "efficiency_wallclock_cir.png")
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================================
# Console output: required steps table
# ============================================================================

def print_required_steps_table(results, target_levels):
    """
    For each target bias level, print a table showing the minimum m
    and wall-clock time each scheme needs to achieve that accuracy.

    Parameters
    ----------
    results       : dict       -- output of run_wallclock_experiment
    target_levels : list[float] -- target bias levels
    """
    scheme_names = [name for name, _, _, _ in SCHEMES]

    for target in target_levels:
        table = find_required_steps_for_target_bias(results, target)

        print(f"\n  Target bias <= {target:.0e}")
        print(f"  {'Scheme':<25s} {'m_req':>8s} {'Time (s)':>12s} {'Bias':>12s}")
        print(f"  {'-'*57}")

        for name in scheme_names:
            entry = table[name]
            if entry["m_required"] is None:
                print(f"  {name:<25s} {'---':>8s} {'---':>12s} {'---':>12s}")
            else:
                print(f"  {name:<25s} {entry['m_required']:>8d} "
                      f"{entry['time_required']:>12.4f} "
                      f"{entry['bias_achieved']:>12.2e}")


# ============================================================================
# Console output: full timing table (for LaTeX transcription)
# ============================================================================

def print_full_timing_table(results, step_counts):
    """
    Print a detailed table of wall-clock times and biases for all
    schemes and step counts, formatted for easy LaTeX transcription.

    Parameters
    ----------
    results     : dict      -- output of run_wallclock_experiment
    step_counts : list[int] -- values of m tested
    """
    scheme_names = [name for name, _, _, _ in SCHEMES]

    print(f"\n  {'='*100}")
    print(f"  Full timing results (median of {TIMING_REPEATS} repeats)")
    print(f"  {'='*100}")

    # Header row
    header = f"  {'m':>5s}"
    for name in scheme_names:
        short = (name.replace("Newton-Milstein", "NM")
                     .replace("Modified Milstein", "Mod.Mil."))
        header += f"  {short:>18s}"
    print(header)

    # Sub-header: show what is reported
    sub = f"  {'':>5s}"
    for _ in scheme_names:
        sub += f"  {'time(s) / bias':>18s}"
    print(sub)
    print(f"  {'-' * (len(header) + 5)}")

    for j, m in enumerate(step_counts):
        row = f"  {m:5d}"
        for name in scheme_names:
            t = results[name]["time"][j]
            b = results[name]["bias"][j]
            row += f"  {t:7.3f} / {b:.2e}"
        print(row)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """
    Run the wall-clock efficiency frontier experiment for the CIR model.
    """
    print("=" * 70)
    print("  Wall-Clock Efficiency Frontier Experiment")
    print("  CIR Model: Bias vs Measured Computation Time")
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

    print(f"\n  Configuration:")
    print(f"    Paths:          {NUM_PATHS}")
    print(f"    Timing repeats: {TIMING_REPEATS}")
    print(f"    Step counts:    {STEP_COUNTS}")
    print(f"\n  Running {len(STEP_COUNTS)} step counts "
          f"x {len(SCHEMES)} schemes ...\n")

    # Run the experiment
    results = run_wallclock_experiment(
        params, analytical_price, STEP_COUNTS, NUM_PATHS, SEED, TIMING_REPEATS
    )

    # Print full timing table
    print_full_timing_table(results, STEP_COUNTS)

    # Print required-steps tables for each target bias level
    print(f"\n  {'='*70}")
    print(f"  Required steps to achieve target bias levels")
    print(f"  {'='*70}")
    print_required_steps_table(results, TARGET_BIAS_LEVELS)

    # Generate the efficiency frontier plot
    print()
    plot_wallclock_efficiency(results, FIG_DIR)

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
