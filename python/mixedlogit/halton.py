"""Halton draws for simulated maximum likelihood estimation.

Mixed Logit choice probabilities are integrals over the random-coefficient
distribution that have no closed form. We approximate them by simulation:

    P_n(i | beta_n) = integral over beta of P(i | beta) * f(beta | theta) d_beta
                   ~= (1/R) * sum_{r=1..R} P(i | beta_r),  beta_r ~ f(theta)

For this to converge fast in R (the number of draws), the draws should be
*low-discrepancy* rather than pseudo-random. Halton sequences are the standard
choice in transport and choice modeling: they fill the unit hypercube more
evenly than uniform random draws, so a Halton-based simulator with R = 200
typically matches the precision of a pseudo-random simulator with R = 2000+.

The "shuffled / scrambled Halton" variant of Bhat (2003) breaks the
correlation that plain Halton sequences develop in dimensions tied to large
primes, and is the recommended default in modern practice.

References
----------
Train, K.E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.,
    Cambridge University Press, Ch. 9.
Bhat, C.R. (2003). "Simulation Estimation of Mixed Discrete Choice Models
    Using Randomized and Scrambled Halton Sequences."
    Transportation Research Part B, 37(9), 837-855.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# First 100 primes - more than enough; we never expect 100 random parameters
# in a Mixed Logit model. (If you do, you have a bigger problem.)
_PRIMES = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541,
)


def _van_der_corput(n_points: int, base: int, *, skip: int = 10) -> np.ndarray:
    """Van der Corput sequence in a given prime base.

    The first ``skip`` elements are discarded because the start of every Halton
    sequence is too regular (always begins 1/b, 2/b, ...) and biases the draws.
    A skip of 10 is the conventional choice; some sources use the 100th prime.
    """
    n_total = n_points + skip
    out = np.zeros(n_total)
    for i in range(1, n_total + 1):
        f = 1.0
        r = 0.0
        k = i
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i - 1] = r
    return out[skip:]


def _scramble_digits(seq: np.ndarray, base: int, rng: np.random.Generator) -> np.ndarray:
    """Bhat-style scrambling: permute the digits of each number in base ``base``.

    For each position-d digit, we draw a single random permutation of {0,...,b-1}
    and apply it to that digit across the whole sequence. This breaks the
    spurious correlation that plain Halton sequences develop in high primes
    while preserving low discrepancy.
    """
    # Convert each entry of seq (in [0,1)) to a finite list of base-b digits
    # to a precision sufficient for our purposes.
    n_digits = max(8, int(np.ceil(np.log(len(seq) + 10) / np.log(base))) + 4)
    n = len(seq)

    # Extract digits
    digits = np.zeros((n, n_digits), dtype=np.int64)
    work = seq.copy()
    for d in range(n_digits):
        work *= base
        digits[:, d] = np.floor(work).astype(np.int64) % base
        work -= np.floor(work)

    # Apply an independent random permutation per digit position
    for d in range(n_digits):
        perm = rng.permutation(base)
        digits[:, d] = perm[digits[:, d]]

    # Reassemble back to a number in [0,1)
    weights = base ** -np.arange(1, n_digits + 1, dtype=np.float64)
    return digits @ weights


def halton_draws(
    n_draws: int,
    n_dims: int,
    *,
    scrambled: bool = True,
    skip: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Halton draws on the unit hypercube.

    Parameters
    ----------
    n_draws : int
        Number of draws (rows in the output).
    n_dims : int
        Number of dimensions (columns). Each dimension uses a different prime
        base; with ``scrambled=False`` and ``n_dims > ~10`` the high-prime
        sequences become visibly correlated, which is why scrambling is on
        by default.
    scrambled : bool, default True
        If True, apply Bhat-style digit scrambling per dimension. Highly
        recommended for n_dims > 5.
    skip : int, default 10
        Number of initial points to discard from each van der Corput sequence.
    seed : int, optional
        Seed for the scrambling RNG. Has no effect when ``scrambled=False``.

    Returns
    -------
    numpy.ndarray, shape (n_draws, n_dims)
        Values in [0, 1), suitable as inputs to inverse-CDF transformations.
    """
    if n_dims > len(_PRIMES):
        raise ValueError(f"n_dims={n_dims} exceeds the {len(_PRIMES)} primes available")
    if n_dims < 1 or n_draws < 1:
        raise ValueError("n_draws and n_dims must be >= 1")

    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, n_dims))
    for k in range(n_dims):
        seq = _van_der_corput(n_draws, _PRIMES[k], skip=skip)
        if scrambled:
            seq = _scramble_digits(seq, _PRIMES[k], rng)
        out[:, k] = seq
    return out


def standard_normal_draws(
    n_draws: int,
    n_dims: int,
    *,
    scrambled: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Halton draws transformed to N(0, 1) via the inverse normal CDF.

    This is the form actually consumed by MXL: standard-normal draws that get
    rescaled by (mean, sd) parameters per random coefficient inside the
    likelihood. Lognormal coefficients use the same draws then exponentiated.

    Returns
    -------
    numpy.ndarray, shape (n_draws, n_dims)
        Standard normal draws.
    """
    u = halton_draws(n_draws, n_dims, scrambled=scrambled, seed=seed)
    # Push values strictly into (0, 1) to avoid +/- inf at the tails of ndtri
    eps = 1e-10
    u = np.clip(u, eps, 1.0 - eps)
    return stats.norm.ppf(u)
