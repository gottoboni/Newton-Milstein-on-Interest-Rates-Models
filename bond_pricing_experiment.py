"""
=============================================================================
Zero-Coupon Bond Pricing Experiment
=============================================================================

This script compares Monte Carlo estimates of the zero-coupon bond price

    P(0, T) = E[ exp( -int_0^T r_s ds ) ]

against exact analytical formulas for the Vasicek and CIR short-rate models.

Four discretization schemes are compared:
    1. Euler--Maruyama          (strong order 1/2)
    2. Classical Milstein       (strong order 1)
    3. Amano's modified Milstein (uniform order 1, used as Newton initialization)
    4. Newton--Milstein with n = 1, 2, 3 Newton corrections

For each scheme and each step count m, we simulate M paths using common
random numbers (the same Brownian increments for all schemes), approximate
the integral of r by the trapezoidal rule, and compute the MC bond price
estimate. We then report:
    - Bias:  |MC estimate - analytical price|
    - RMSE:  sqrt( mean( (individual estimate - analytical)^2 ) )

References:
    - Brigo, D. and Mercurio, F. (2006). Interest Rate Models.
    - Amano, K. (2016). Newton--Milstein scheme.
    - Kloeden, P. E. and Platen, E. (1992). Numerical Solution of SDEs.

=============================================================================
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os
import time

# ============================================================================
# Global configuration
# ============================================================================

# Random seed for reproducibility
SEED = 42

# Number of Monte Carlo paths
NUM_PATHS = 100_000

# Step counts to test (powers of 2 for clean halving)
STEP_COUNTS = [8, 16, 32, 64, 128, 256, 512]

# Output directory for figures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# CIR regularization parameter (epsilon for the smooth cutoff near zero)
CIR_EPSILON = 1e-8


# ============================================================================
# Model parameter sets (Brigo--Mercurio style calibrations)
# ============================================================================

VASICEK_PARAMS = {
    "kappa": 0.5,       # mean-reversion speed
    "theta": 0.05,      # long-run mean rate
    "sigma": 0.02,      # volatility
    "r0":    0.03,      # initial short rate
    "T":     5.0,       # bond maturity (years)
}

CIR_PARAMS = {
    "kappa": 0.5,       # mean-reversion speed
    "theta": 0.05,      # long-run mean rate
    "sigma": 0.1,       # volatility
    "r0":    0.03,      # initial short rate
    "T":     5.0,       # bond maturity (years)
    # Feller condition check: 2 * kappa * theta = 0.05 >= sigma^2 = 0.01  => satisfied
}


# ============================================================================
# Analytical bond price formulas
# ============================================================================

def vasicek_bond_price(kappa, theta, sigma, r0, T):
    """
    Exact zero-coupon bond price P(0, T) under the Vasicek model.

    The Vasicek model: dr = kappa * (theta - r) dt + sigma dW
    admits the affine bond price formula:

        P(0, T) = A(0, T) * exp( -B(0, T) * r0 )

    where:
        B(0, T) = (1 - exp(-kappa * T)) / kappa
        A(0, T) = exp( (B - T) * (kappa^2 * theta - sigma^2 / 2) / kappa^2
                       - sigma^2 * B^2 / (4 * kappa) )

    Reference: Brigo and Mercurio (2006), Section 3.2.1.

    Parameters
    ----------
    kappa : float  -- mean-reversion speed
    theta : float  -- long-run mean
    sigma : float  -- volatility
    r0    : float  -- initial rate
    T     : float  -- maturity

    Returns
    -------
    float : exact bond price P(0, T)
    """
    B = (1.0 - np.exp(-kappa * T)) / kappa
    A = np.exp(
        (B - T) * (kappa**2 * theta - 0.5 * sigma**2) / kappa**2
        - sigma**2 * B**2 / (4.0 * kappa)
    )
    return A * np.exp(-B * r0)


def cir_bond_price(kappa, theta, sigma, r0, T):
    """
    Exact zero-coupon bond price P(0, T) under the CIR model.

    The CIR model: dr = kappa * (theta - r) dt + sigma * sqrt(r) dW
    admits the affine bond price formula:

        P(0, T) = A(0, T) * exp( -B(0, T) * r0 )

    where:
        gamma = sqrt(kappa^2 + 2 * sigma^2)
        B(0, T) = 2 * (exp(gamma * T) - 1)
                   / ((gamma + kappa) * (exp(gamma * T) - 1) + 2 * gamma)
        A(0, T) = ( 2 * gamma * exp((kappa + gamma) * T / 2)
                     / ((gamma + kappa) * (exp(gamma * T) - 1) + 2 * gamma)
                   )^(2 * kappa * theta / sigma^2)

    Reference: Brigo and Mercurio (2006), Section 3.2.3.

    Parameters
    ----------
    kappa : float  -- mean-reversion speed
    theta : float  -- long-run mean
    sigma : float  -- volatility
    r0    : float  -- initial rate
    T     : float  -- maturity

    Returns
    -------
    float : exact bond price P(0, T)
    """
    gamma = np.sqrt(kappa**2 + 2.0 * sigma**2)
    exp_gamma_T = np.exp(gamma * T)
    denominator = (gamma + kappa) * (exp_gamma_T - 1.0) + 2.0 * gamma

    B = 2.0 * (exp_gamma_T - 1.0) / denominator

    # Exponent for A: 2 * kappa * theta / sigma^2
    exponent = 2.0 * kappa * theta / sigma**2
    A = (2.0 * gamma * np.exp((kappa + gamma) * T / 2.0) / denominator) ** exponent

    return A * np.exp(-B * r0)


# ============================================================================
# Brownian increment generation
# ============================================================================

def generate_brownian_increments(num_paths, num_steps, step_size, rng):
    """
    Generate Brownian increments dW_i = W(t_{i+1}) - W(t_i) ~ N(0, h)
    for all paths simultaneously.

    Also generates the integrated Brownian motion increments needed by
    the modified Milstein scheme:
        J_i = int_{t_i}^{t_{i+1}} (W_s - W_{t_i}) ds

    which is jointly Gaussian with dW_i:
        J_i = (h / 2) * dW_i + (h^{3/2} / (2 * sqrt(3))) * xi_i
    where xi_i ~ N(0,1), independent of dW_i.

    Parameters
    ----------
    num_paths  : int   -- number of MC sample paths (M)
    num_steps  : int   -- number of time steps (m)
    step_size  : float -- time step h = T / m
    rng        : numpy Generator -- seeded random number generator

    Returns
    -------
    dW : ndarray of shape (num_paths, num_steps)
         Brownian increments sqrt(h) * Z, where Z ~ N(0,1)
    J  : ndarray of shape (num_paths, num_steps)
         Integrated Brownian motion increments on each subinterval
    """
    h = step_size

    # Standard normal draws for the Brownian increments
    z1 = rng.standard_normal((num_paths, num_steps))
    dW = np.sqrt(h) * z1

    # Independent standard normal draws for the integrated BM
    z2 = rng.standard_normal((num_paths, num_steps))
    J = (h / 2.0) * dW + (h**1.5 / (2.0 * np.sqrt(3.0))) * z2

    return dW, J


# ============================================================================
# CIR regularization helper
# ============================================================================

def psi_epsilon(x, epsilon):
    """
    Smooth regularization function psi_epsilon(x) for the CIR diffusion.

    We use a smooth transition:
        psi_eps(x) = epsilon                          for x <= epsilon / 2
        psi_eps(x) = x                                for x >= 2 * epsilon
        psi_eps(x) = smooth interpolation             in between

    The smooth interpolation is a C^infinity bump constructed via:
        psi_eps(x) = epsilon + (x - epsilon/2) * phi( (x - epsilon/2) / (3*epsilon/2) )
    where phi is a smooth monotone function from 0 to 1 built from the
    standard bump function. For simplicity, we use a polynomial approximation
    that is C^2 and satisfies the required boundary conditions.

    In practice, for Monte Carlo bond pricing, a simple clamp
    psi_eps(x) = max(x, epsilon) works well because the paths rarely
    visit the regularization region when the Feller condition holds.
    We implement the simple clamp here for clarity and efficiency.

    Parameters
    ----------
    x       : ndarray -- input values (current rate values across paths)
    epsilon : float   -- regularization parameter

    Returns
    -------
    psi_val  : ndarray -- regularized values, >= epsilon
    psi_deriv: ndarray -- derivative psi'(x), in {0, 1}
    """
    # Simple clamp: psi(x) = max(x, epsilon)
    psi_val = np.maximum(x, epsilon)
    # Derivative: 1 where x > epsilon, 0 where x <= epsilon
    psi_deriv = np.where(x > epsilon, 1.0, 0.0)
    return psi_val, psi_deriv


# ============================================================================
# Scheme implementations: Vasicek model
# ============================================================================

def simulate_vasicek_euler(params, num_steps, dW):
    """
    Euler--Maruyama scheme for the Vasicek model.

    dr = kappa * (theta - r) * dt + sigma * dW

    Discretization:
        r_{i+1} = r_i + kappa * (theta - r_i) * h + sigma * dW_i

    Parameters
    ----------
    params    : dict   -- model parameters (kappa, theta, sigma, r0, T)
    num_steps : int    -- number of time steps m
    dW        : ndarray of shape (num_paths, num_steps) -- Brownian increments

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1) -- simulated rate paths
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    # Allocate path array: columns 0, 1, ..., m correspond to times 0, h, ..., T
    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = r0

    for i in range(num_steps):
        r_current = paths[:, i]
        # Euler--Maruyama update
        paths[:, i + 1] = (
            r_current
            + kappa * (theta - r_current) * h
            + sigma * dW[:, i]
        )

    return paths


def simulate_vasicek_milstein(params, num_steps, dW):
    """
    Classical Milstein scheme for the Vasicek model.

    Since Db = d(sigma)/dx = 0, the Milstein correction term
    Mb * [(dW)^2 - h] / 2 vanishes. The Milstein scheme thus
    coincides with the Euler--Maruyama scheme for the Vasicek model.

    We implement it separately for clarity and to confirm this identity.

    Parameters
    ----------
    params    : dict   -- model parameters
    num_steps : int    -- number of time steps
    dW        : ndarray of shape (num_paths, num_steps)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    # For Vasicek, Milstein = Euler--Maruyama because Db = 0
    return simulate_vasicek_euler(params, num_steps, dW)


def simulate_vasicek_modified_milstein(params, num_steps, dW, J):
    """
    Amano's modified Milstein scheme for the Vasicek model.

    The scheme adds a correction term involving Ma and the integrated
    Brownian motion:

        r_{i+1} = r_i + a_i * h + b_i * dW_i
                  + Ma_i * J_i + Mb_i * [(dW_i)^2 - h] / 2

    For Vasicek:
        a(t, x)  = kappa * (theta - x)
        b(t, x)  = sigma
        Ma       = b * Da = sigma * (-kappa) = -kappa * sigma
        Mb       = b * Db = 0

    So the scheme becomes:
        r_{i+1} = r_i + kappa * (theta - r_i) * h + sigma * dW_i
                  - kappa * sigma * J_i

    Parameters
    ----------
    params    : dict   -- model parameters
    num_steps : int    -- number of time steps
    dW        : ndarray of shape (num_paths, num_steps)
    J         : ndarray of shape (num_paths, num_steps) -- integrated BM increments

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    # Precompute the constant Ma = -kappa * sigma
    Ma = -kappa * sigma

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = r0

    for i in range(num_steps):
        r_current = paths[:, i]
        # Modified Milstein update
        paths[:, i + 1] = (
            r_current
            + kappa * (theta - r_current) * h
            + sigma * dW[:, i]
            + Ma * J[:, i]
        )

    return paths


def simulate_vasicek_newton(params, num_steps, dW, J, num_newton_steps):
    """
    Newton--Milstein scheme for the Vasicek model.

    As shown in the Application chapter, the Newton iteration for the
    Vasicek model converges in a single step: the linearized SDE coincides
    with the original SDE because the drift is affine and the diffusion
    is constant. Therefore, for n >= 1 Newton steps, the result is the
    exact solution (up to time discretization of the integrals).

    We implement the general Newton iteration nonetheless, to verify
    this one-step convergence numerically.

    The Newton iterate Z^{(n+1)} solves the linearized SDE:
        dZ = (a0_n + a1_n * Z) dt + (b0_n + b1_n * Z) dW

    For Vasicek:
        a0_n = kappa * theta,  a1_n = -kappa
        b0_n = sigma,          b1_n = 0

    Since b1_n = 0, the integrating factor is deterministic:
        eta_n(t) = -kappa * t
    and the explicit solution on each subinterval [t_i, t_{i+1}] is:
        Z_{i+1} = exp(-kappa * h) * Z_i
                  + (kappa * theta / kappa) * (1 - exp(-kappa * h))
                  + sigma * int_{t_i}^{t_{i+1}} exp(-kappa * (t_{i+1} - s)) dW_s

    The last stochastic integral is approximated consistently with the
    Milstein discretization.

    Parameters
    ----------
    params           : dict -- model parameters
    num_steps        : int  -- number of time steps
    dW               : ndarray of shape (num_paths, num_steps)
    J                : ndarray of shape (num_paths, num_steps)
    num_newton_steps : int  -- number of Newton corrections (n)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    # Step 0: Initialize with the modified Milstein approximation
    z_current = simulate_vasicek_modified_milstein(params, num_steps, dW, J)

    # Newton iterations
    for newton_iter in range(num_newton_steps):
        z_new = np.empty_like(z_current)
        z_new[:, 0] = r0

        for i in range(num_steps):
            z_n = z_current[:, i]

            # Linearized coefficients at the current iterate
            # For Vasicek these are constant (independent of z_n):
            a0_n = kappa * theta
            a1_n = -kappa
            b0_n = sigma
            b1_n = 0.0

            # Since b1_n = 0, the integrating factor is deterministic:
            #   exp(eta) = exp(a1_n * h) = exp(-kappa * h)
            exp_eta = np.exp(a1_n * h)

            # Explicit solution of the linearized SDE on [t_i, t_{i+1}]:
            #   Z_{i+1} = exp(eta) * [ Z_i + (a0_n - b0_n * b1_n) * I_0 + b0_n * I_1 ]
            # where I_0 = int exp(-eta(s)) ds and I_1 = int exp(-eta(s)) dW_s
            # are approximated at first order:
            #   I_0 approx h * exp(-a1_n * h/2)  (midpoint)
            #   I_1 approx dW_i
            # For the simple case b1_n = 0:
            z_new[:, i + 1] = (
                exp_eta * z_current[:, i]
                + (a0_n / (-a1_n)) * (1.0 - exp_eta)
                + b0_n * exp_eta * dW[:, i]
                # Higher-order stochastic correction (from modified Milstein):
                + exp_eta * (-kappa * sigma) * J[:, i]
            )

        z_current = z_new

    return z_current


# ============================================================================
# Scheme implementations: CIR model
# ============================================================================

def simulate_cir_euler(params, num_steps, dW):
    """
    Euler--Maruyama scheme for the CIR model.

    dr = kappa * (theta - r) * dt + sigma * sqrt(r) * dW

    Discretization:
        r_{i+1} = r_i + kappa * (theta - r_i) * h + sigma * sqrt(max(r_i, 0)) * dW_i

    The max(r_i, 0) clamp prevents taking the square root of a negative
    number, which can happen with the Euler scheme even when the Feller
    condition is satisfied.

    Parameters
    ----------
    params    : dict   -- model parameters
    num_steps : int    -- number of time steps
    dW        : ndarray of shape (num_paths, num_steps)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = r0

    for i in range(num_steps):
        r_current = paths[:, i]
        # Clamp to zero to prevent sqrt of negative values
        r_positive = np.maximum(r_current, 0.0)
        # Euler--Maruyama update
        paths[:, i + 1] = (
            r_current
            + kappa * (theta - r_current) * h
            + sigma * np.sqrt(r_positive) * dW[:, i]
        )

    return paths


def simulate_cir_milstein(params, num_steps, dW):
    """
    Classical Milstein scheme for the CIR model.

    The Milstein correction for CIR involves:
        Mb = b * Db = sigma * sqrt(x) * sigma / (2 * sqrt(x)) = sigma^2 / 2

    So the scheme is:
        r_{i+1} = r_i + kappa * (theta - r_i) * h
                  + sigma * sqrt(r_i) * dW_i
                  + (sigma^2 / 4) * [(dW_i)^2 - h]

    Note: Mb = sigma^2 / 2, and the Milstein term is Mb * [(dW)^2 - h] / 2
          = (sigma^2 / 4) * [(dW)^2 - h].

    Parameters
    ----------
    params    : dict
    num_steps : int
    dW        : ndarray of shape (num_paths, num_steps)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    # Precompute the Milstein correction coefficient: Mb / 2 = sigma^2 / 4
    milstein_coeff = sigma**2 / 4.0

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = r0

    for i in range(num_steps):
        r_current = paths[:, i]
        r_positive = np.maximum(r_current, 0.0)
        dW_i = dW[:, i]
        # Milstein update
        paths[:, i + 1] = (
            r_current
            + kappa * (theta - r_current) * h
            + sigma * np.sqrt(r_positive) * dW_i
            + milstein_coeff * (dW_i**2 - h)
        )

    return paths


def simulate_cir_modified_milstein(params, num_steps, dW, J):
    """
    Amano's modified Milstein scheme for the CIR model.

    Adds the Ma * J correction to the classical Milstein scheme.

    For CIR:
        Ma = b * Da = sigma * sqrt(x) * (-kappa) = -kappa * sigma * sqrt(x)

    The scheme becomes:
        r_{i+1} = r_i + kappa * (theta - r_i) * h
                  + sigma * sqrt(r_i) * dW_i
                  + (sigma^2 / 4) * [(dW_i)^2 - h]
                  - kappa * sigma * sqrt(r_i) * J_i

    Parameters
    ----------
    params    : dict
    num_steps : int
    dW        : ndarray of shape (num_paths, num_steps)
    J         : ndarray of shape (num_paths, num_steps)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    num_paths = dW.shape[0]

    milstein_coeff = sigma**2 / 4.0

    paths = np.empty((num_paths, num_steps + 1))
    paths[:, 0] = r0

    for i in range(num_steps):
        r_current = paths[:, i]
        r_positive = np.maximum(r_current, 0.0)
        sqrt_r = np.sqrt(r_positive)
        dW_i = dW[:, i]

        # Modified Milstein update:
        # Euler part + Milstein correction + Ma * J correction
        paths[:, i + 1] = (
            r_current
            + kappa * (theta - r_current) * h
            + sigma * sqrt_r * dW_i
            + milstein_coeff * (dW_i**2 - h)
            - kappa * sigma * sqrt_r * J[:, i]
        )

    return paths


def simulate_cir_newton(params, num_steps, dW, J, num_newton_steps):
    """
    Newton--Milstein scheme for the CIR model with regularization.

    This is the key nontrivial case. Unlike Vasicek, the CIR diffusion
    b(x) = sigma * sqrt(x) is nonlinear, so the Newton iteration does
    NOT converge in one step. The linearized coefficients at each iterate
    Z^{(n)} are (using the regularized diffusion):

        a0_n = kappa * theta
        a1_n = -kappa
        b1_n = sigma * psi'(Z^{(n)}) / (2 * sqrt(psi(Z^{(n)})))
        b0_n = sigma * sqrt(psi(Z^{(n)})) - b1_n * Z^{(n)}

    The Newton iterate Z^{(n+1)} is obtained by solving this linear SDE
    on each subinterval [t_i, t_{i+1}] via the integrating factor method.

    Since b1_n != 0, the integrating factor involves a stochastic integral:
        eta_n(t) = int_0^t (a1_n - b1_n^2 / 2) ds + int_0^t b1_n dW_s

    On each subinterval, we approximate:
        eta_i = (a1_n_i - b1_n_i^2 / 2) * h + b1_n_i * dW_i

    Parameters
    ----------
    params           : dict
    num_steps        : int
    dW               : ndarray of shape (num_paths, num_steps)
    J                : ndarray of shape (num_paths, num_steps)
    num_newton_steps : int -- number of Newton corrections (1, 2, or 3)

    Returns
    -------
    paths : ndarray of shape (num_paths, num_steps + 1)
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    r0 = params["r0"]
    T = params["T"]
    h = T / num_steps
    epsilon = CIR_EPSILON
    num_paths = dW.shape[0]

    # Step 0: Initialize with the modified Milstein approximation
    z_current = simulate_cir_modified_milstein(params, num_steps, dW, J)

    # Newton iterations
    for newton_iter in range(num_newton_steps):
        z_new = np.empty_like(z_current)
        z_new[:, 0] = r0

        for i in range(num_steps):
            z_n = z_current[:, i]
            dW_i = dW[:, i]

            # Regularized diffusion at the current iterate
            psi_val, psi_der = psi_epsilon(z_n, epsilon)
            sqrt_psi = np.sqrt(psi_val)

            # Linearized coefficients
            a0_n = kappa * theta
            a1_n = -kappa
            b1_n = sigma * psi_der / (2.0 * sqrt_psi)
            b0_n = sigma * sqrt_psi - b1_n * z_n

            # Integrating factor increment on [t_i, t_{i+1}]:
            #   delta_eta = (a1_n - b1_n^2 / 2) * h + b1_n * dW_i
            delta_eta = (a1_n - 0.5 * b1_n**2) * h + b1_n * dW_i
            exp_delta_eta = np.exp(delta_eta)

            # Explicit solution of the linearized SDE on [t_i, t_{i+1}]:
            #   Z_{i+1} = exp(delta_eta) * Z_i
            #             + exp(delta_eta) * (a0_n - b0_n * b1_n) * h
            #             + exp(delta_eta) * b0_n * dW_i
            #
            # This is a first-order discretization of the integral
            # representation from Lemma (explicit solution via integrating factor).
            # The integrands exp(-eta_n(s)) are approximated as constant = 1
            # on each subinterval, which is consistent with the Milstein-level
            # local accuracy.
            z_new[:, i + 1] = (
                exp_delta_eta * z_n
                + exp_delta_eta * (a0_n - b0_n * b1_n) * h
                + exp_delta_eta * b0_n * dW_i
            )

        z_current = z_new

    return z_current


# ============================================================================
# Bond price estimation via Monte Carlo
# ============================================================================

def compute_bond_price_mc(paths, T):
    """
    Estimate the zero-coupon bond price P(0, T) = E[exp(-int_0^T r_s ds)]
    from simulated rate paths using the trapezoidal rule.

    Parameters
    ----------
    paths : ndarray of shape (num_paths, num_steps + 1)
            Simulated short-rate paths at times 0, h, 2h, ..., T
    T     : float -- maturity

    Returns
    -------
    mc_price    : float -- Monte Carlo estimate of P(0, T)
    mc_std_err  : float -- standard error of the MC estimate
    mc_prices   : ndarray of shape (num_paths,) -- individual path prices
    """
    num_paths, num_points = paths.shape
    num_steps = num_points - 1
    h = T / num_steps

    # Trapezoidal rule: int_0^T r_s ds approx h * [r_0/2 + r_1 + ... + r_{m-1} + r_m/2]
    integral = h * (0.5 * paths[:, 0] + paths[:, 1:-1].sum(axis=1) + 0.5 * paths[:, -1])

    # Discount factors for each path
    mc_prices = np.exp(-integral)

    # Monte Carlo estimate and standard error
    mc_price = np.mean(mc_prices)
    mc_std_err = np.std(mc_prices) / np.sqrt(num_paths)

    return mc_price, mc_std_err, mc_prices


# ============================================================================
# Main experiment runner
# ============================================================================

def run_experiment(model_name, params, analytical_price, scheme_simulators, step_counts, num_paths, seed):
    """
    Run the convergence experiment for a given model.

    For each step count m and each scheme, simulate num_paths paths using
    common Brownian increments, compute the MC bond price, and record
    the bias and RMSE.

    Parameters
    ----------
    model_name        : str  -- "Vasicek" or "CIR"
    params            : dict -- model parameters
    analytical_price  : float -- exact bond price
    scheme_simulators : dict  -- {scheme_name: simulator_function}
    step_counts       : list of int -- values of m to test
    num_paths         : int  -- number of MC paths
    seed              : int  -- random seed

    Returns
    -------
    results : dict -- {scheme_name: {"bias": [...], "rmse": [...], "std_err": [...]}}
    """
    T = params["T"]

    # Initialize results storage
    results = {name: {"bias": [], "rmse": [], "std_err": []} for name in scheme_simulators}

    for m in step_counts:
        h = T / m
        print(f"  m = {m:4d} (h = {h:.4f}) ... ", end="", flush=True)

        # Generate common Brownian increments for all schemes
        rng = np.random.default_rng(seed)
        dW, J = generate_brownian_increments(num_paths, m, h, rng)

        for scheme_name, simulator in scheme_simulators.items():
            # Determine which arguments the simulator needs
            if "newton" in scheme_name.lower():
                # Extract the number of Newton steps from the scheme name
                # Format: "Newton-Milstein (n=k)"
                n_newton = int(scheme_name.split("n=")[1].rstrip(")"))
                paths = simulator(params, m, dW, J, n_newton)
            elif "modified" in scheme_name.lower() or "amano" in scheme_name.lower():
                paths = simulator(params, m, dW, J)
            else:
                paths = simulator(params, m, dW)

            # Compute MC bond price
            mc_price, mc_std_err, mc_prices = compute_bond_price_mc(paths, T)

            # Bias: |MC estimate - exact price|
            bias = abs(mc_price - analytical_price)

            # RMSE: sqrt( mean( (individual price - exact)^2 ) )
            # This combines both bias and variance:
            # RMSE^2 = bias^2 + variance
            rmse = np.sqrt(np.mean((mc_prices - analytical_price)**2))

            results[scheme_name]["bias"].append(bias)
            results[scheme_name]["rmse"].append(rmse)
            results[scheme_name]["std_err"].append(mc_std_err)

        print("done")

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_bias_convergence(model_name, step_counts, results, analytical_price, fig_dir):
    """
    Create a log-log plot of |bias| vs step size h for all schemes.

    Parameters
    ----------
    model_name      : str
    step_counts     : list of int
    results         : dict -- output of run_experiment
    analytical_price: float
    fig_dir         : str -- directory to save the figure
    """
    T = 5.0  # maturity (used only for computing h from m)
    h_values = [T / m for m in step_counts]

    # Color and marker assignments for each scheme
    style_map = {
        "Euler-Maruyama":       {"color": "#1f77b4", "marker": "o",  "linestyle": "-"},
        "Milstein":             {"color": "#ff7f0e", "marker": "s",  "linestyle": "-"},
        "Modified Milstein":    {"color": "#2ca02c", "marker": "^",  "linestyle": "-"},
        "Newton-Milstein (n=1)":{"color": "#d62728", "marker": "D",  "linestyle": "--"},
        "Newton-Milstein (n=2)":{"color": "#9467bd", "marker": "v",  "linestyle": "--"},
        "Newton-Milstein (n=3)":{"color": "#8c564b", "marker": "P",  "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheme_name, data in results.items():
        style = style_map.get(scheme_name, {"color": "gray", "marker": "x", "linestyle": "-"})
        bias_values = data["bias"]
        # Replace zeros with a tiny value for log-log plotting
        bias_plot = [max(b, 1e-16) for b in bias_values]
        ax.loglog(
            h_values, bias_plot,
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=6,
            label=scheme_name,
        )

    # Reference slopes
    h_ref = np.array(h_values)
    # Order 1/2 reference line
    ref_scale_half = bias_plot[0] * (h_ref / h_ref[0])**0.5 if results else h_ref**0.5
    ax.loglog(h_ref, ref_scale_half * 0.5, "k:", linewidth=0.8, alpha=0.5, label=r"$O(h^{1/2})$")
    # Order 1 reference line
    ref_scale_one = bias_plot[0] * (h_ref / h_ref[0])**1.0 if results else h_ref
    ax.loglog(h_ref, ref_scale_one * 0.3, "k-.", linewidth=0.8, alpha=0.5, label=r"$O(h)$")

    ax.set_xlabel(r"Step size $h = T/m$", fontsize=12)
    ax.set_ylabel(r"$|\mathrm{Bias}|$", fontsize=12)
    ax.set_title(f"{model_name} Model: Bond Price Bias vs Step Size", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    filepath = os.path.join(fig_dir, f"bias_{model_name.lower()}.png")
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_rmse_convergence(model_name, step_counts, results, fig_dir):
    """
    Create a log-log plot of RMSE vs step size h for all schemes.

    Parameters
    ----------
    model_name  : str
    step_counts : list of int
    results     : dict
    fig_dir     : str
    """
    T = 5.0
    h_values = [T / m for m in step_counts]

    style_map = {
        "Euler-Maruyama":       {"color": "#1f77b4", "marker": "o",  "linestyle": "-"},
        "Milstein":             {"color": "#ff7f0e", "marker": "s",  "linestyle": "-"},
        "Modified Milstein":    {"color": "#2ca02c", "marker": "^",  "linestyle": "-"},
        "Newton-Milstein (n=1)":{"color": "#d62728", "marker": "D",  "linestyle": "--"},
        "Newton-Milstein (n=2)":{"color": "#9467bd", "marker": "v",  "linestyle": "--"},
        "Newton-Milstein (n=3)":{"color": "#8c564b", "marker": "P",  "linestyle": "--"},
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for scheme_name, data in results.items():
        style = style_map.get(scheme_name, {"color": "gray", "marker": "x", "linestyle": "-"})
        ax.loglog(
            h_values, data["rmse"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            markersize=6,
            label=scheme_name,
        )

    ax.set_xlabel(r"Step size $h = T/m$", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(f"{model_name} Model: Bond Price RMSE vs Step Size", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    filepath = os.path.join(fig_dir, f"rmse_{model_name.lower()}.png")
    fig.savefig(filepath, dpi=200)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def print_results_table(model_name, step_counts, results, analytical_price, T):
    """
    Print a formatted table of results to the console.

    Parameters
    ----------
    model_name      : str
    step_counts     : list of int
    results         : dict
    analytical_price: float
    T               : float
    """
    print(f"\n{'='*80}")
    print(f"  {model_name} Model -- Analytical P(0,{T:.0f}) = {analytical_price:.10f}")
    print(f"{'='*80}")

    scheme_names = list(results.keys())
    header = f"{'m':>5s} {'h':>8s}"
    for name in scheme_names:
        short = name.replace("Newton-Milstein", "NM").replace("Modified Milstein", "Mod.Mil.")
        header += f"  {short:>14s}"
    print(f"\n  BIAS:")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for j, m in enumerate(step_counts):
        h = T / m
        row = f"  {m:5d} {h:8.4f}"
        for name in scheme_names:
            bias = results[name]["bias"][j]
            row += f"  {bias:14.2e}"
        print(row)

    print(f"\n  RMSE:")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    for j, m in enumerate(step_counts):
        h = T / m
        row = f"  {m:5d} {h:8.4f}"
        for name in scheme_names:
            rmse = results[name]["rmse"][j]
            row += f"  {rmse:14.2e}"
        print(row)

    print()


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """
    Run the full bond pricing experiment for Vasicek and CIR models.
    """
    print("=" * 70)
    print("  Zero-Coupon Bond Pricing Experiment")
    print("  Monte Carlo vs Analytical Bond Prices")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Vasicek model
    # ------------------------------------------------------------------
    print("\n[1/2] Vasicek model")
    params_v = VASICEK_PARAMS
    price_v = vasicek_bond_price(**{k: params_v[k] for k in ["kappa", "theta", "sigma", "r0", "T"]})
    print(f"  Analytical P(0, {params_v['T']}) = {price_v:.10f}")

    # Define the schemes to compare
    vasicek_schemes = {
        "Euler-Maruyama":        simulate_vasicek_euler,
        "Milstein":              simulate_vasicek_milstein,
        "Modified Milstein":     simulate_vasicek_modified_milstein,
        "Newton-Milstein (n=1)": simulate_vasicek_newton,
        "Newton-Milstein (n=2)": simulate_vasicek_newton,
        "Newton-Milstein (n=3)": simulate_vasicek_newton,
    }

    t0 = time.time()
    results_v = run_experiment(
        "Vasicek", params_v, price_v, vasicek_schemes, STEP_COUNTS, NUM_PATHS, SEED
    )
    elapsed_v = time.time() - t0
    print(f"  Vasicek experiment completed in {elapsed_v:.1f}s")

    print_results_table("Vasicek", STEP_COUNTS, results_v, price_v, params_v["T"])
    plot_bias_convergence("Vasicek", STEP_COUNTS, results_v, price_v, FIG_DIR)
    plot_rmse_convergence("Vasicek", STEP_COUNTS, results_v, FIG_DIR)

    # ------------------------------------------------------------------
    # CIR model
    # ------------------------------------------------------------------
    print("\n[2/2] CIR model")
    params_c = CIR_PARAMS
    price_c = cir_bond_price(**{k: params_c[k] for k in ["kappa", "theta", "sigma", "r0", "T"]})
    print(f"  Analytical P(0, {params_c['T']}) = {price_c:.10f}")

    # Verify Feller condition
    feller = 2.0 * params_c["kappa"] * params_c["theta"]
    sigma_sq = params_c["sigma"]**2
    print(f"  Feller condition: 2*kappa*theta = {feller:.4f} >= sigma^2 = {sigma_sq:.4f} => {'SATISFIED' if feller >= sigma_sq else 'VIOLATED'}")

    cir_schemes = {
        "Euler-Maruyama":        simulate_cir_euler,
        "Milstein":              simulate_cir_milstein,
        "Modified Milstein":     simulate_cir_modified_milstein,
        "Newton-Milstein (n=1)": simulate_cir_newton,
        "Newton-Milstein (n=2)": simulate_cir_newton,
        "Newton-Milstein (n=3)": simulate_cir_newton,
    }

    t0 = time.time()
    results_c = run_experiment(
        "CIR", params_c, price_c, cir_schemes, STEP_COUNTS, NUM_PATHS, SEED
    )
    elapsed_c = time.time() - t0
    print(f"  CIR experiment completed in {elapsed_c:.1f}s")

    print_results_table("CIR", STEP_COUNTS, results_c, price_c, params_c["T"])
    plot_bias_convergence("CIR", STEP_COUNTS, results_c, price_c, FIG_DIR)
    plot_rmse_convergence("CIR", STEP_COUNTS, results_c, FIG_DIR)

    print("\nAll experiments complete. Figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
