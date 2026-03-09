"""
Data generation for Spearman power simulations.

Provides three y-generation strategies:
  1. Gaussian copula with non-parametric marginals  (generate_y_copula)
  2. Linear model with Gaussian noise               (generate_y_linear)
  3. Non-parametric rank-mixing                      (generate_y_nonparametric)

The non-parametric method is the recommended default for Monte Carlo power
and CI estimation when x has ties.  It mixes standardized ranks of x with
random noise ranks at weight rho_s, then maps the resulting ordering onto
draws from the target y-marginal.  Because the mixing operates directly in
rank space, it handles tied x-values naturally and achieves the target
Spearman correlation in expectation without relying on the continuous-
marginals assumption that underpins the copula.

The Gaussian copula is retained as an option; however, when x has heavy
ties the jittering step collapses information and attenuates the realised
Spearman rho by 0.01-0.06, leading to underestimated power.  Distributional
transform and adaptive jitter alternatives were tested and do not fix the
issue for heavy ties (k=4); they only help for mild ties (k>=10).

X-values are generated either from a pre-specified frequency dictionary
(producing ties) or as all-distinct values.
"""

import functools

import numpy as np
from scipy.stats import norm, rankdata

from config import X_PARAMS, FREQ_DICT
from spearman_helpers import _rank_rows, _pearson_on_rank_arrays

try:
    from data.digitized import H_AL71, B_AL71, H_AL_OUTLIER, B_AL_OUTLIER_MIN, B_AL_OUTLIER_MAX
    _DIGITIZED_AVAILABLE = True
except ImportError:
    _DIGITIZED_AVAILABLE = False
    H_AL71 = B_AL71 = H_AL_OUTLIER = B_AL_OUTLIER_MIN = B_AL_OUTLIER_MAX = None


def digitized_available():
    """Return True if data/digitized.py was imported successfully (empirical generator can be used)."""
    return _DIGITIZED_AVAILABLE

GENERATORS = {
    "copula": "generate_y_copula",
    "linear": "generate_y_linear",
    "nonparametric": "generate_y_nonparametric",
    "empirical": "generate_y_empirical",
}

_NORM_PPF_075 = float(norm.ppf(0.75))


def get_generator(name):
    """Return the y-generation function for *name* ('copula', 'linear', or
    'nonparametric')."""
    funcs = {
        "copula": generate_y_copula,
        "linear": generate_y_linear,
        "nonparametric": generate_y_nonparametric,
        "empirical": generate_y_empirical,
    }
    if name not in funcs:
        raise ValueError(f"Unknown generator {name!r}; choose from {list(funcs)}")
    return funcs[name]


# ---------------------------------------------------------------------------
# X generation
# ---------------------------------------------------------------------------

_X_TEMPLATE_CACHE = {}


def _get_x_template(n, n_distinct, distribution_type, freq_dict, all_distinct):
    """Return a cached sorted x-array (the 'template') that only needs shuffling."""
    lo, hi = X_PARAMS["range"]
    if all_distinct:
        key = ("all_distinct", n)
        if key not in _X_TEMPLATE_CACHE:
            _X_TEMPLATE_CACHE[key] = np.linspace(lo, hi, n)
        return _X_TEMPLATE_CACHE[key]

    _fd = freq_dict if freq_dict is not None else FREQ_DICT
    counts = _fd[n][n_distinct][distribution_type]
    key = (n, n_distinct, distribution_type, tuple(counts))
    if key not in _X_TEMPLATE_CACHE:
        distinct_vals = np.linspace(lo, hi, n_distinct)
        _X_TEMPLATE_CACHE[key] = np.repeat(distinct_vals, counts)
    return _X_TEMPLATE_CACHE[key]


def generate_cumulative_aluminum(n, n_distinct, distribution_type=None,
                                 freq_dict=None, all_distinct=False, rng=None):
    """Return an array of cumulative vaccine aluminum values.

    Parameters
    ----------
    n : int
        Sample size.
    n_distinct : int
        Number of distinct x-values (ignored when *all_distinct* is True).
    distribution_type : str or None
        One of 'even', 'heavy_tail', 'heavy_center'.  Required when
        *all_distinct* is False.
    freq_dict : dict or None
        Nested frequency dictionary.  Falls back to ``config.FREQ_DICT``.
    all_distinct : bool
        If True, generate *n* unique values spanning the x range.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    x : ndarray of shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    template = _get_x_template(n, n_distinct, distribution_type, freq_dict,
                                all_distinct)
    x = template.copy()
    rng.shuffle(x)
    return x


# ---------------------------------------------------------------------------
# Y generation – Empirical
# ---------------------------------------------------------------------------

if _DIGITIZED_AVAILABLE:
    known_b_al_outliers = np.array([B_AL_OUTLIER_MIN, B_AL_OUTLIER_MAX])
else:
    known_b_al_outliers = None


def generate_b_al_outliers(n_missing=5, low=None, high=None, rng=None):
    """
    Sample remaining B-Al outliers using log-uniform distribution
    (heavy-tailed, realistic for extreme biomarker values).
    """
    if not _DIGITIZED_AVAILABLE:
        raise ImportError(
            "Digitized data not available. Create data/digitized.py or use --skip-empirical.")
    if low is None:
        low = B_AL_OUTLIER_MIN
    if high is None:
        high = B_AL_OUTLIER_MAX
    if rng is None:
        rng = np.random.default_rng()

    log_low = np.log(low)
    log_high = np.log(high)
    sampled = np.exp(rng.uniform(log_low, log_high, n_missing))
    return np.concatenate([known_b_al_outliers, sampled])

# ---------------------------------------------------------------------------
# Empirical pool construction
# ---------------------------------------------------------------------------

_POOL_CACHE = {}
_POOL_SEED = 42


def get_pool(n):
    """Return the empirical Y pool for sample size *n*.

    Requires digitized data (data/digitized.py). Use digitized_available() to check.
    The pool has exactly *n* values, matching the Karwowski case:
      n=73 -> B-Al clean  (71 digitized + 2 resampled)
      n=80 -> B-Al full   (73 clean + 2 known outliers + 5 log-uniform outliers)
      n=81 -> H-Al clean  (71 digitized + 10 resampled)
      n=82 -> H-Al full   (81 clean + 1 known outlier)

    Missing non-outlier values are filled by resampling with replacement
    from the 71 known values.  Unknown B-Al outliers are log-uniform
    between B_AL_OUTLIER_MIN and B_AL_OUTLIER_MAX.  All random generation
    uses a fixed seed (_POOL_SEED=42) for reproducibility.      Results are
    cached so the pool is identical across calls and runs.

    **Calibration only.** For data generation, use ``build_empirical_pool(n, rng)``
    which resamples the remainder per call.
    """
    if not _DIGITIZED_AVAILABLE:
        raise ImportError(
            "Digitized data not available. Create data/digitized.py or use --skip-empirical.")
    if n in _POOL_CACHE:
        return _POOL_CACHE[n]

    rng = np.random.default_rng(_POOL_SEED)

    if n == 73:
        fill = rng.choice(B_AL71, size=2, replace=True)
        pool = np.concatenate([B_AL71, fill])
    elif n == 80:
        base = get_pool(73)
        outliers = generate_b_al_outliers(n_missing=5, rng=rng)
        pool = np.concatenate([base, outliers])
    elif n == 81:
        fill = rng.choice(H_AL71, size=10, replace=True)
        pool = np.concatenate([H_AL71, fill])
    elif n == 82:
        base = get_pool(81)
        pool = np.append(base, H_AL_OUTLIER)
    else:
        raise ValueError(
            f"Empirical generator only supports n in {{73, 80, 81, 82}}, got {n}")

    _POOL_CACHE[n] = pool
    return pool


def build_empirical_pool(n, rng):
    """Build an empirical Y pool of size *n*, advancing *rng*.

    Unlike get_pool (which is cached with a fixed seed for calibration),
    this function is called once per sim/rep so the 'remainder' values
    (beyond the 71 digitized) vary across replications.

    The 71 digitized values (B_AL71 or H_AL71) appear exactly once.
    Only the remainder is resampled from the 71 via rng.
    """
    if not _DIGITIZED_AVAILABLE:
        raise ImportError(
            "Digitized data not available. Create data/digitized.py or use --skip-empirical.")
    if n == 73:
        fill = rng.choice(B_AL71, size=2, replace=True)
        return np.concatenate([B_AL71, fill])
    elif n == 80:
        base = build_empirical_pool(73, rng)
        outliers = generate_b_al_outliers(n_missing=5, rng=rng)
        return np.concatenate([base, outliers])
    elif n == 81:
        fill = rng.choice(H_AL71, size=10, replace=True)
        return np.concatenate([H_AL71, fill])
    elif n == 82:
        base = build_empirical_pool(81, rng)
        return np.append(base, H_AL_OUTLIER)
    else:
        raise ValueError(
            f"Empirical generator only supports n in {{73, 80, 81, 82}}, got {n}")


def generate_y_empirical(x, rho_s, y_params, rng=None, _calibrated_rho=None):
    """Generate y via rank-mixing with empirical marginal.

    Builds a fresh pool per call via build_empirical_pool: the 71 digitized
    values appear exactly once, remainder is resampled using rng.
    """
    if rng is None:
        rng = np.random.default_rng()
    pool = build_empirical_pool(len(x), rng)
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks = rankdata(x, method="average")
    return _raw_rank_mix(x_ranks, rho_input, y_params, rng,
                         marginal="empirical", pool=pool)


def _build_empirical_pool_batch(n, n_reps, rng):
    """Vectorized pool construction: returns (n_reps, n) array.

    Replaces the Python loop ``[build_empirical_pool(n, rng) for _ in range(n_reps)]``
    with batched numpy calls, eliminating per-rep Python overhead.
    The 71 digitized values appear exactly once in every row; only the
    remainder is resampled.
    """
    if not _DIGITIZED_AVAILABLE:
        raise ImportError(
            "Digitized data not available. Create data/digitized.py or use --skip-empirical.")
    if n == 73:
        fill = rng.choice(B_AL71, size=(n_reps, 2), replace=True)       # (n_reps, 2)
        base = np.tile(B_AL71, (n_reps, 1))                               # (n_reps, 71)
        return np.concatenate([base, fill], axis=1)                       # (n_reps, 73)
    elif n == 80:
        pool73 = _build_empirical_pool_batch(73, n_reps, rng)             # (n_reps, 73)
        log_low = np.log(B_AL_OUTLIER_MIN)
        log_high = np.log(B_AL_OUTLIER_MAX)
        outlier_fill = np.exp(rng.uniform(log_low, log_high, (n_reps, 5)))  # (n_reps, 5)
        known_out = np.tile(known_b_al_outliers, (n_reps, 1))             # (n_reps, 2)
        return np.concatenate([pool73, known_out, outlier_fill], axis=1)  # (n_reps, 80)
    elif n == 81:
        fill = rng.choice(H_AL71, size=(n_reps, 10), replace=True)       # (n_reps, 10)
        base = np.tile(H_AL71, (n_reps, 1))                               # (n_reps, 71)
        return np.concatenate([base, fill], axis=1)                       # (n_reps, 81)
    elif n == 82:
        pool81 = _build_empirical_pool_batch(81, n_reps, rng)             # (n_reps, 81)
        outlier = np.full((n_reps, 1), H_AL_OUTLIER, dtype=np.float64)   # (n_reps, 1)
        return np.concatenate([pool81, outlier], axis=1)                  # (n_reps, 82)
    else:
        raise ValueError(
            f"Empirical generator only supports n in {{73, 80, 81, 82}}, got {n}")


def generate_y_empirical_batch(x_batch, rho_s, y_params, rng=None,
                               _calibrated_rho=None):
    """Batch version of generate_y_empirical.

    Builds a fresh pool per rep via _build_empirical_pool_batch (vectorized):
    the 71 digitized values appear exactly once, remainder is resampled.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_reps, n = x_batch.shape
    pool_batch = _build_empirical_pool_batch(n, n_reps, rng)
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks_batch = _rank_rows(x_batch)
    return _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng,
                               marginal="empirical", pool=pool_batch)


# ---------------------------------------------------------------------------
# Y generation – Gaussian copula
# ---------------------------------------------------------------------------

def _spearman_to_pearson(rho_s):
    """Convert Spearman rho to the Pearson correlation of the underlying
    bivariate normal needed by the Gaussian copula.

    Uses the exact identity: rho_s = (6/pi) * arcsin(rho_p / 2).
    """
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def generate_y_copula(x, rho_s, y_params, rng=None):
    """Generate y using a Gaussian copula that targets Spearman *rho_s*.

    .. warning::
       When x has heavy ties the jittering step collapses rank information
       and attenuates the realised Spearman rho.  Prefer
       ``generate_y_nonparametric`` for tied-x scenarios.

    Parameters
    ----------
    x : ndarray
        Observed (possibly tied) x-values.
    rho_s : float
        Target Spearman correlation.
    y_params : dict
        Must contain 'median', 'iqr', and 'range' keys.
    rng : numpy.random.Generator or None
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    rho_p = _spearman_to_pearson(rho_s)

    ranks_x = rankdata(x, method="average")
    u_x = (ranks_x - 0.5) / n
    u_x = np.clip(u_x + rng.normal(0, 1e-6, n), 1e-8, 1 - 1e-8)
    z_x = norm.ppf(u_x)

    z_y = rho_p * z_x + np.sqrt(max(1.0 - rho_p ** 2, 0.0)) * rng.standard_normal(n)

    u_y = norm.cdf(z_y)

    mu_ln, sigma_ln = _fit_lognormal(y_params["median"], y_params["iqr"])
    y = _lognormal_quantile(u_y, mu_ln, sigma_ln)

    return y


@functools.lru_cache(maxsize=32)
def _fit_lognormal(median, iqr):
    """Return (mu, sigma) of a log-normal whose median and IQR match."""
    mu = np.log(median)
    q75_target = median + iqr / 2.0
    sigma = (np.log(q75_target) - mu) / _NORM_PPF_075
    sigma = max(sigma, 0.01)
    return mu, sigma


def _lognormal_quantile(u, mu, sigma):
    return np.exp(mu + sigma * norm.ppf(np.clip(u, 1e-8, 1 - 1e-8)))


# ---------------------------------------------------------------------------
# Y generation – linear model
# ---------------------------------------------------------------------------

def generate_y_linear(x, rho_s, y_params, rng=None):
    """Generate y = a + b*x + noise calibrated to approximate target Spearman rho_s.

    Noise variance is tuned so that the *Pearson* correlation of the
    rank-transformed data approximates *rho_s*.  Because Spearman rho equals
    the Pearson correlation of ranks, this gives a reasonable (though not
    exact) calibration, especially when x has moderate ties.

    Parameters
    ----------
    x : ndarray
        Observed (possibly tied) x-values.
    rho_s : float
        Target Spearman correlation (approximate under linear model).
    y_params : dict
        Must contain 'median' and 'iqr'.
    rng : numpy.random.Generator or None
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    mu_ln, sigma_ln = _fit_lognormal(y_params["median"], y_params["iqr"])

    if abs(rho_s) < 1e-12:
        return np.exp(mu_ln + sigma_ln * rng.standard_normal(n))

    x_std = (x - np.mean(x))
    sx = np.std(x_std, ddof=0)
    if sx < 1e-12:
        return np.exp(mu_ln + sigma_ln * rng.standard_normal(n))
    x_std = x_std / sx

    rho_target = _spearman_to_pearson(rho_s)
    rho_target = np.clip(rho_target, -0.999, 0.999)

    b = rho_target * sigma_ln
    noise_var = sigma_ln ** 2 * (1.0 - rho_target ** 2)
    noise_sd = np.sqrt(max(noise_var, 1e-12))

    log_y = mu_ln + b * x_std + noise_sd * rng.standard_normal(n)
    y = np.exp(log_y)

    return y


# ---------------------------------------------------------------------------
# Y generation – Non-parametric rank-mixing (recommended for tied x)
# ---------------------------------------------------------------------------

def _raw_rank_mix(x_ranks, rho_input, y_params, rng, _ln_params=None,
                  marginal="lognormal", pool=None):
    """Core rank-mixing: returns y with ranks correlated to x_ranks.

    Parameters
    ----------
    _ln_params : tuple or None
        Pre-computed (mu, sigma) from _fit_lognormal.  Avoids repeated
        lookups when called in tight loops.
    marginal : str
        "lognormal" (default) or "empirical".
    pool : ndarray or None
        Required when marginal="empirical".  Values to resample from.
    """
    n = len(x_ranks)
    noise_ranks = rng.permutation(n) + 1.0
    # Jitter breaks integer commensurability that causes a discrete staircase
    # in E[Spearman(rho_in)] for heavily tied x (e.g. k=4 equal groups).
    # |jitter| < 0.5 guarantees adjacent integers can never cross, preserving
    # the uniform random permutation semantics of noise_ranks.  The jitter
    # mean is zero so population-constant standardization below remains valid
    # to within per-sample fluctuations of ~0.49/sqrt(3n) per row (~0.006 for
    # n=80), which is negligible for calibration accuracy.
    noise_ranks = noise_ranks + rng.uniform(-0.49, 0.49, size=n)

    s_x = x_ranks - np.mean(x_ranks)
    sd_x = np.std(x_ranks, ddof=0)
    if sd_x > 0:
        s_x /= sd_x

    s_n = noise_ranks - np.mean(noise_ranks)
    s_n /= np.std(noise_ranks, ddof=0)

    rho_c = np.clip(rho_input, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

    if marginal == "lognormal":
        if _ln_params is not None:
            mu, sigma = _ln_params
        else:
            mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
        y_values = rng.lognormal(mean=mu, sigma=sigma, size=n)
        y_values.sort()
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        if len(pool) != n:
            raise ValueError(
                f"For empirical marginal, pool length must equal n={n}, got {len(pool)}")
        y_values = np.sort(pool)
    else:
        raise ValueError(f"Unknown marginal: {marginal}")

    y_final = np.empty(n)
    y_final[np.argsort(mixed)] = y_values
    return y_final


_CALIBRATION_CACHE = {}


def _fast_spearman(x_ranks, y):
    """Pearson correlation of (x_ranks, rankdata(y)) -- avoids scipy overhead."""
    yr = rankdata(y, method="average")
    xr_m = x_ranks - np.mean(x_ranks)
    yr_m = yr - np.mean(yr)
    num = np.dot(xr_m, yr_m)
    denom = np.sqrt(np.dot(xr_m, xr_m) * np.dot(yr_m, yr_m))
    if denom < 1e-15:
        return 0.0
    return num / denom


def _mean_rho(rho_in, template, y_params, n_cal, seed):
    """Scalar reference implementation; not used in production (see _mean_rho_vec).

    Retained for validation and unit testing against the vectorized path.
    """
    cal_rng = np.random.default_rng(seed)
    mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
    total = 0.0
    for _ in range(n_cal):
        x = template.copy()
        cal_rng.shuffle(x)
        x_ranks = rankdata(x, method="average")
        nn = len(x_ranks)
        noise_ranks = cal_rng.permutation(nn) + 1.0

        s_x = x_ranks - np.mean(x_ranks)
        sd_x = np.std(x_ranks, ddof=0)
        if sd_x > 0:
            s_x /= sd_x
        s_n = noise_ranks - np.mean(noise_ranks)
        s_n /= np.std(noise_ranks, ddof=0)

        rho_c = np.clip(rho_in, -0.999, 0.999)
        mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

        y_vals = cal_rng.lognormal(mean=mu, sigma=sigma, size=nn)
        y_vals.sort()
        # For continuous y, rank(y_final) = rank(mixed), so these draws are
        # redundant for calibration. Retained as reference; see _eval_mean_rho_fast.
        y_final = np.empty(nn)
        y_final[np.argsort(mixed)] = y_vals

        total += _fast_spearman(x_ranks, y_final)
    return total / n_cal


def _precompute_calibration_arrays(template, y_params, n_cal, seed):
    """Precompute the rho-independent arrays for nonparametric calibration.

    Retained as reference implementation for validation and for marginals
    with tied y-values. For lognormal (continuous) y, use
    _precompute_calibration_arrays_fast, which exploits rank(y_final) =
    rank(mixed) to skip y generation entirely. See _eval_mean_rho_fast and
    README § "Skip-y identity for continuous marginals".

    Called once per bisection probe; the returned arrays are reused across
    all ~27 bisection iterations, avoiding redundant RNG work and x-ranking.

    Returns
    -------
    s_x : ndarray, shape (n_cal, n)
        Standardized x ranks.
    s_n : ndarray, shape (n_cal, n)
        Standardized noise ranks.
    y_batch_sorted : ndarray, shape (n_cal, n)
        Lognormal draws, sorted within each row (ready for rank-assignment).
    x_ranks_batch : ndarray, shape (n_cal, n)
        Raw x ranks (needed by _pearson_on_rank_arrays).
    """
    cal_rng = np.random.default_rng(seed)
    n = len(template)
    mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])

    # Batch-shuffle x: (n_cal, n)
    x_batch = np.tile(template, (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        x_batch = cal_rng.permuted(x_batch, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        x_batch = np.take_along_axis(x_batch, perm, axis=1)

    x_ranks_batch = _rank_rows(x_batch)  # (n_cal, n), Numba-parallel

    # Noise ranks: independent permutation per row
    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        noise_batch = cal_rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        noise_batch = np.take_along_axis(noise_base, perm, axis=1)
    # Jitter breaks integer commensurability that causes a discrete staircase
    # in E[Spearman(rho_in)] for heavily tied x (e.g. k=4 equal groups).
    # |jitter| < 0.5 guarantees adjacent integers can never cross, preserving
    # the uniform random permutation semantics.  The jitter mean is zero so
    # population-constant standardization remains valid; per-row mean
    # fluctuation is ~0.49/(sqrt(3n)*noise_std) ~ 0.00137 for n=80, negligible.
    noise_batch = noise_batch + cal_rng.uniform(-0.49, 0.49, size=(n_cal, n))

    # Standardize x_ranks per row
    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)
    s_x /= sd_x

    # Population mean/std for permutations of 1..n remain valid constants
    # after jitter (jitter mean=0, jitter variance 0.49^2/3 << (n^2-1)/12).
    noise_mean = (n + 1) / 2.0
    noise_std = np.sqrt((n * n - 1) / 12.0)
    s_n = (noise_batch - noise_mean) / noise_std

    y_batch_sorted = cal_rng.lognormal(mean=mu, sigma=sigma, size=(n_cal, n))
    y_batch_sorted.sort(axis=1)

    return s_x, s_n, y_batch_sorted, x_ranks_batch


def _precompute_calibration_arrays_fast(template, n_cal, seed):
    """Precompute rho-independent arrays for lognormal nonparametric calibration.

    Exploits rank(y_final) = rank(mixed) for continuous (tie-free) y:
    y_params are not needed and y_batch_sorted is not generated.

    Returns (s_x, s_n, x_ranks_batch) -- note: 3 arrays, not 4.
    """
    cal_rng = np.random.default_rng(seed)
    n = len(template)

    x_batch = np.tile(template, (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        x_batch = cal_rng.permuted(x_batch, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        x_batch = np.take_along_axis(x_batch, perm, axis=1)

    x_ranks_batch = _rank_rows(x_batch)

    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        noise_batch = cal_rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        noise_batch = np.take_along_axis(noise_base, perm, axis=1)
    # Jitter breaks integer commensurability that causes a discrete staircase
    # in E[Spearman(rho_in)] for heavily tied x (e.g. k=4 equal groups).
    # |jitter| < 0.5 guarantees adjacent integers can never cross, preserving
    # the uniform random permutation semantics.  The jitter mean is zero so
    # population-constant standardization remains valid; per-row mean
    # fluctuation is ~0.49/(sqrt(3n)*noise_std) ~ 0.00137 for n=80, negligible.
    noise_batch = noise_batch + cal_rng.uniform(-0.49, 0.49, size=(n_cal, n))

    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)
    s_x /= sd_x

    # Population mean/std for permutations of 1..n remain valid constants
    # after jitter (jitter mean=0, jitter variance 0.49^2/3 << (n^2-1)/12).
    noise_mean = (n + 1) / 2.0
    noise_std = np.sqrt((n * n - 1) / 12.0)
    s_n = (noise_batch - noise_mean) / noise_std

    return s_x, s_n, x_ranks_batch


def _eval_mean_rho(rho_in, s_x, s_n, y_batch_sorted, x_ranks_batch):
    """Evaluate mean realised Spearman rho for a given rho_in.

    Takes the precomputed rho-independent arrays from
    _precompute_calibration_arrays (or _precompute_calibration_arrays_empirical)
    and performs only the rho-dependent steps: mixed-rank construction,
    argsort, y-assignment, ranking, and Pearson-on-ranks averaging.

    Used by both nonparametric and empirical calibration bisection loops.
    """
    n_cal = s_x.shape[0]
    rho_c = np.clip(rho_in, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n  # (n_cal, n)

    order = np.argsort(mixed, axis=1)
    rows = np.arange(n_cal)[:, None]
    y_final = np.empty_like(y_batch_sorted)
    y_final[rows, order] = y_batch_sorted

    ry = _rank_rows(y_final)  # (n_cal, n)
    rhos = _pearson_on_rank_arrays(x_ranks_batch, ry)  # (n_cal,)
    return float(np.mean(rhos))


def _eval_mean_rho_fast(rho_in, s_x, s_n, x_ranks_batch):
    """Fast mean Spearman rho exploiting rank(y_final) = rank(mixed).

    Valid when y has no ties (lognormal marginal). Skips argsort,
    y-assignment, and y-ranking entirely.
    """
    rho_c = np.clip(rho_in, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n
    ry = _rank_rows(mixed)
    rhos = _pearson_on_rank_arrays(x_ranks_batch, ry)
    return float(np.mean(rhos))


def _precompute_calibration_arrays_copula(template, n_cal, seed):
    """Precompute rho-independent arrays for Gaussian copula calibration.

    Exploits rank(y) = rank(z_y) for continuous (tie-free) lognormal y:
    since y = exp(mu + sigma*z_y) is strictly increasing, ordering is
    preserved and y_params are not needed during calibration.

    Returns (z_x_batch, noise_batch, x_ranks_batch).
      z_x_batch   : (n_cal, n) normal quantiles derived from x ranks + jitter
      noise_batch : (n_cal, n) iid standard normal draws (rho-independent)
      x_ranks_batch : (n_cal, n) midranks of x (for Pearson-on-ranks)
    """
    cal_rng = np.random.default_rng(seed)
    n = len(template)

    x_batch = np.tile(template, (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        x_batch = cal_rng.permuted(x_batch, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        x_batch = np.take_along_axis(x_batch, perm, axis=1)

    x_ranks_batch = _rank_rows(x_batch)

    # Copula x-transform: midranks → uniform → normal (with tie-breaking jitter)
    u_x = (x_ranks_batch - 0.5) / n
    u_x = np.clip(u_x + cal_rng.normal(0, 1e-6, (n_cal, n)), 1e-8, 1 - 1e-8)
    z_x_batch = norm.ppf(u_x)

    # Independent standard normal noise (the epsilon term in z_y)
    noise_batch = cal_rng.standard_normal((n_cal, n))

    return z_x_batch, noise_batch, x_ranks_batch


def _eval_mean_rho_copula_fast(rho_in, z_x_batch, noise_batch, x_ranks_batch):
    """Fast mean Spearman rho for the Gaussian copula exploiting rank(y) = rank(z_y).

    Since y = exp(mu + sigma*z_y) is strictly increasing, rank(y) = rank(z_y),
    so Spearman(x, y) = Spearman(x, z_y).  No norm.cdf, lognormal quantile,
    or y_params needed.
    """
    rho_p = _spearman_to_pearson(np.clip(rho_in, -0.999, 0.999))
    z_y = rho_p * z_x_batch + np.sqrt(max(1.0 - rho_p ** 2, 0.0)) * noise_batch
    ry = _rank_rows(z_y)
    rhos = _pearson_on_rank_arrays(x_ranks_batch, ry)
    return float(np.mean(rhos))


def _bisect_for_probe_copula(probe, template, n_cal, seed,
                              n_iter=25, tol=5e-5):
    """Bisect to find rho_in such that E[Spearman_copula(rho_in)] ≈ probe.

    Precomputes rho-independent arrays once (z_x, noise), then calls
    _eval_mean_rho_copula_fast for each bisection step (~27×).
    Returns rho_in (float) or None if the probe is unreachable.
    """
    arrays = _precompute_calibration_arrays_copula(template, n_cal, seed)

    lo, hi = 0.0, min(probe * 2.0, 0.999)
    val_hi = _eval_mean_rho_copula_fast(hi, *arrays)
    if val_hi < probe:
        hi = 0.999
        val_hi = _eval_mean_rho_copula_fast(hi, *arrays)
    if val_hi < probe:
        return None  # unreachable

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        observed = _eval_mean_rho_copula_fast(mid, *arrays)
        if observed < probe:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    # Return the endpoint whose evaluated mean_rho is closer to the probe.
    # Consistent with _bisect_for_probe: avoids midpoint landing on the wrong
    # plateau when step discontinuities are present (heavily tied x).
    val_lo = _eval_mean_rho_copula_fast(lo, *arrays)
    val_hi = _eval_mean_rho_copula_fast(hi, *arrays)
    return hi if abs(val_hi - probe) <= abs(val_lo - probe) else lo


def _mean_rho_vec(rho_in, template, y_params, n_cal, seed):
    """Vectorized mean realised Spearman rho: replaces the n_cal Python loop
    with a single batch of matrix operations.

    Produces an equivalent Monte Carlo estimate as _mean_rho but uses
    _rank_rows (Numba-parallel when available) and BLAS matrix ops,
    giving a ~10-20x speedup for n_cal=300 at n~80.
    Random sequence differs from _mean_rho for the same seed (batch vs
    sequential generation) but the expectation and variance are the same.

    Convenience wrapper around _precompute_calibration_arrays_fast +
    _eval_mean_rho_fast (skip-y path for lognormal).
    """
    s_x, s_n, x_ranks_batch = _precompute_calibration_arrays_fast(
        template, n_cal, seed)
    return _eval_mean_rho_fast(rho_in, s_x, s_n, x_ranks_batch)


def _bisect_for_probe(probe, template, y_params, n_cal, seed,
                      n_iter=25, tol=5e-5):
    """Bisect to find rho_in such that _mean_rho(rho_in, ...) ≈ probe.

    Precomputes rho-independent arrays once (fast path: no y), then calls
    _eval_mean_rho_fast for each bisection step (~27×).
    Returns rho_in (float) or None if the probe is unreachable.
    """
    arrays = _precompute_calibration_arrays_fast(template, n_cal, seed)

    lo, hi = 0.0, min(probe * 2.0, 0.999)
    val_hi = _eval_mean_rho_fast(hi, *arrays)
    if val_hi < probe:
        hi = 0.999
        val_hi = _eval_mean_rho_fast(hi, *arrays)
    if val_hi < probe:
        return None  # unreachable

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        observed = _eval_mean_rho_fast(mid, *arrays)
        if observed < probe:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    # Return the endpoint whose evaluated mean_rho is closer to the probe,
    # rather than the midpoint (lo+hi)/2.  Reason: for highly-tied x
    # distributions, _eval_mean_rho has structural step discontinuities.
    #
    # (a) Cause: Inside _eval_mean_rho we compute mixed = rho_c*s_x + sqrt(1-
    #     rho_c^2)*s_n and assign y via argsort(mixed).  With heavily tied x,
    #     s_x has repeated values; a tiny change in rho_c can flip the
    #     ordering of mixed at tied positions, so rank assignments jump
    #     discretely and mean Spearman rho jumps (e.g. from ~0.28 to ~0.30).
    #     This is not Monte Carlo noise — it persists with n_cal=20,000.
    #
    # (b) Why (lo+hi)/2 was wrong: Bisection converges to the jump, so lo and
    #     hi sit on opposite sides (e.g. f(lo)=0.280, f(hi)=0.301 for
    #     probe=0.30).  The midpoint (lo+hi)/2 can lie on the low plateau and
    #     evaluate to f(mid)=0.280.  Returning that rho_in then caused
    #     simulations to yield mean_rho≈0.280 instead of the target 0.30.
    #
    # (c) For piecewise-constant (step) functions, the only achievable values
    #     are the plateaus.  The rho_in we want is one that gives realised
    #     mean_rho closest to the probe — i.e. whichever of lo or hi has
    #     f(.) closer to the probe.  For continuous calibration (low ties),
    #     lo ≈ hi at convergence so the choice is irrelevant.
    #
    # (d) When |val_hi - probe| == |val_lo - probe| we return hi (<= in the
    #     condition below).  Intentional: the bisection invariant guarantees
    #     f(hi) >= probe at every iteration, so hi is the endpoint we have a
    #     proof about and is the natural default when the two endpoints cannot
    #     be distinguished by distance alone. Note on bias direction: preferring
    #     hi means accepting slight overestimation of the probe (f(hi) >= probe).
    #     For power analysis, overestimating the calibrated rho is the 
    #     anti-conservative direction (simulated correlation slightly above 
    #     target, power slightly inflated). The effect is negligible in the
    #     tiebreaker case but worth noting for auditing purposes.
    val_lo = _eval_mean_rho_fast(lo, *arrays)
    val_hi = _eval_mean_rho_fast(hi, *arrays)
    return hi if abs(val_hi - probe) <= abs(val_lo - probe) else lo


def _interp_with_extrapolation(x, xp, fp):
    """Linear interpolation with linear extrapolation beyond endpoints."""
    if x <= xp[0]:
        if len(xp) >= 2:
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            return fp[0] + slope * (x - xp[0])
        return fp[0]
    if x >= xp[-1]:
        if len(xp) >= 2:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return fp[-1] + slope * (x - xp[-1])
        return fp[-1]
    return float(np.interp(x, xp, fp))


_CALIBRATION_CACHE_MULTIPOINT = {}
# 5 probes at gap 0.10 give max interpolation error < 0.0004 in [0.25, 0.42]
# (the bisection search range).  3 probes at gap 0.20 produced up to 0.0022
# error in that range due to curvature of the calibration curve at higher rho.
_MULTIPOINT_PROBES = [0.10, 0.20, 0.30, 0.40, 0.50]


def _calibrate_rho_multipoint(n, n_distinct, distribution_type, rho_target,
                               y_params, all_distinct=False, n_cal=300,
                               seed=99, freq_dict=None):
    """Multi-point calibration: probe at 3 rho values, interpolate."""
    # Cache key excludes y_params: rank(mixed) = rank(y_final) for continuous y,
    # so the calibration curve depends only on (n, template, seed, n_cal).
    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal,
                 "multipoint")
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_MULTIPOINT:
        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)
        pairs = []
        for probe in _MULTIPOINT_PROBES:
            rho_in = _bisect_for_probe(probe, template, y_params,
                                        n_cal, seed)
            if rho_in is not None:
                pairs.append((probe, rho_in))

        _CALIBRATION_CACHE_MULTIPOINT[cache_key] = pairs

    pairs = sorted(_CALIBRATION_CACHE_MULTIPOINT[cache_key], key=lambda p: p[0])

    if not pairs:
        # No probes succeeded -- return rho_target unmodified
        return float(np.clip(rho_target, -0.999, 0.999))

    abs_target = abs(rho_target)
    sign = 1.0 if rho_target >= 0 else -1.0

    if len(pairs) == 1:
        # Fall back to single-point ratio
        probe, rho_in = pairs[0]
        ratio = rho_in / probe
        result = abs_target * ratio
    else:
        probes = [p for p, _ in pairs]
        rho_ins = [r for _, r in pairs]
        result = _interp_with_extrapolation(abs_target, probes, rho_ins)

    return float(np.clip(sign * result, -0.999, 0.999))


_CALIBRATION_CACHE_MULTIPOINT_COPULA = {}
# Uses the same 5 probes as nonparametric multipoint: max interpolation
# error < 0.0004 in [0.25, 0.42] (the bisection search range).


def _calibrate_rho_multipoint_copula(n, n_distinct, distribution_type, rho_target,
                                      all_distinct=False, n_cal=300,
                                      seed=99, freq_dict=None):
    """Multi-point calibration for the Gaussian copula: probe at 5 rho values, interpolate.

    Cache key excludes y_params: rank(y) = rank(z_y) for continuous lognormal y,
    so the calibration curve depends only on (n, template, seed, n_cal).
    """
    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal,
                 "multipoint")
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_MULTIPOINT_COPULA:
        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)
        pairs = []
        for probe in _MULTIPOINT_PROBES:
            rho_in = _bisect_for_probe_copula(probe, template, n_cal, seed)
            if rho_in is not None:
                pairs.append((probe, rho_in))

        _CALIBRATION_CACHE_MULTIPOINT_COPULA[cache_key] = pairs

    pairs = sorted(_CALIBRATION_CACHE_MULTIPOINT_COPULA[cache_key], key=lambda p: p[0])

    if not pairs:
        return float(np.clip(rho_target, -0.999, 0.999))

    abs_target = abs(rho_target)
    sign = 1.0 if rho_target >= 0 else -1.0

    if len(pairs) == 1:
        probe, rho_in = pairs[0]
        ratio = rho_in / probe
        result = abs_target * ratio
    else:
        probes = [p for p, _ in pairs]
        rho_ins = [r for _, r in pairs]
        result = _interp_with_extrapolation(abs_target, probes, rho_ins)

    return float(np.clip(sign * result, -0.999, 0.999))


def calibrate_rho(n, n_distinct, distribution_type, rho_target, y_params,
                  all_distinct=False, n_cal=300, seed=99, freq_dict=None,
                  calibration_mode="multipoint"):
    """Find the input mixing parameter that produces E[Spearman] = rho_target.

    Uses bisection on a fixed probe rho (0.30) to compute a rho-independent
    attenuation ratio for the given tie structure.  The ratio is cached per
    (n, k, dist_type) so that different rho_target values reuse the same
    calibration -- eliminating the dominant bottleneck where bisection in
    ``min_detectable_rho`` triggered a fresh calibration at every step.

    Parameters
    ----------
    freq_dict : dict or None
        Custom frequency dictionary.  When provided, must contain
        freq_dict[n][n_distinct]["custom"] = list of counts.
        Used instead of FREQ_DICT when distribution_type is "custom".
    calibration_mode : str
        "multipoint" (default) probes at 3 rho values and interpolates;
        "single" uses one probe at 0.30 and a linear ratio.

    Returns
    -------
    float
        Calibrated input rho to feed into _raw_rank_mix.
    """
    if abs(rho_target) < 1e-12:
        return 0.0

    if calibration_mode == "multipoint":
        return _calibrate_rho_multipoint(
            n, n_distinct, distribution_type, rho_target, y_params,
            all_distinct=all_distinct, n_cal=n_cal, seed=seed,
            freq_dict=freq_dict)

    # --- existing single-point code below (unchanged) ---
    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal)
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE:
        probe = 0.30

        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)

        def _mean_rho(rho_in):
            cal_rng = np.random.default_rng(seed)
            mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
            total = 0.0
            for _ in range(n_cal):
                x = template.copy()
                cal_rng.shuffle(x)
                x_ranks = rankdata(x, method="average")
                nn = len(x_ranks)
                noise_ranks = cal_rng.permutation(nn) + 1.0

                s_x = x_ranks - np.mean(x_ranks)
                sd_x = np.std(x_ranks, ddof=0)
                if sd_x > 0:
                    s_x /= sd_x
                s_n = noise_ranks - np.mean(noise_ranks)
                s_n /= np.std(noise_ranks, ddof=0)

                rho_c = np.clip(rho_in, -0.999, 0.999)
                mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

                y_vals = cal_rng.lognormal(mean=mu, sigma=sigma, size=nn)
                y_vals.sort()
                y_final = np.empty(nn)
                y_final[np.argsort(mixed)] = y_vals

                total += _fast_spearman(x_ranks, y_final)
            return total / n_cal

        lo, hi = 0.0, min(probe * 2.0, 0.999)
        if _mean_rho(hi) < probe:
            hi = 0.999

        for _ in range(25):
            mid = (lo + hi) / 2.0
            observed = _mean_rho(mid)
            if observed < probe:
                lo = mid
            else:
                hi = mid
            if hi - lo < 5e-5:
                break

        # Return the endpoint whose _mean_rho is closer to the probe
        # (same fix as _bisect_for_probe: avoids midpoint landing on
        # wrong plateau when step discontinuities are present).
        val_lo = _mean_rho(lo)
        val_hi = _mean_rho(hi)
        calibrated_probe = hi if abs(val_hi - probe) <= abs(val_lo - probe) else lo
        ratio = calibrated_probe / probe if probe > 0 else 1.0
        _CALIBRATION_CACHE[cache_key] = ratio

    ratio = _CALIBRATION_CACHE[cache_key]
    result = rho_target * ratio
    return float(np.clip(result, -0.999, 0.999))


_CALIBRATION_CACHE_COPULA = {}


def calibrate_rho_copula(n, n_distinct, distribution_type, rho_target, y_params,
                         all_distinct=False, n_cal=300, seed=99, freq_dict=None,
                         calibration_mode="multipoint"):
    """Find the input rho_s for the copula that produces E[Spearman] = rho_target.

    Uses the fast-path calibration: precomputes z_x and noise once (rho-
    independent), then bisects using only _eval_mean_rho_copula_fast at each
    step — no generate_y_copula calls in the calibration loop.

    Valid because rank(y) = rank(z_y) for lognormal y: y = exp(mu + sigma*z_y)
    is strictly increasing, so Spearman(x, y) = Spearman(x, z_y).

    Parameters
    ----------
    y_params : dict
        Retained for API compatibility (not used in the fast calibration path).
    freq_dict : dict or None
        Custom frequency dictionary.  When provided, must contain
        freq_dict[n][n_distinct]["custom"] = list of counts.
    calibration_mode : str
        "multipoint" (default) probes at 5 rho values and interpolates;
        "single" bisects at probe rho=0.30 and applies a linear ratio.

    Returns
    -------
    float
        Calibrated input rho_s to feed into generate_y_copula.
    """
    if abs(rho_target) < 1e-12:
        return 0.0

    if calibration_mode == "multipoint":
        return _calibrate_rho_multipoint_copula(
            n, n_distinct, distribution_type, rho_target,
            all_distinct=all_distinct, n_cal=n_cal, seed=seed,
            freq_dict=freq_dict)

    # --- single-point fast path ---
    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal)
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_COPULA:
        probe = 0.30
        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)
        calibrated_probe = _bisect_for_probe_copula(probe, template, n_cal, seed)
        if calibrated_probe is None:
            calibrated_probe = probe
        ratio = calibrated_probe / probe if probe > 0 else 1.0
        _CALIBRATION_CACHE_COPULA[cache_key] = ratio

    ratio = _CALIBRATION_CACHE_COPULA[cache_key]
    result = rho_target * ratio
    return float(np.clip(result, -0.999, 0.999))


# ---------------------------------------------------------------------------
# Empirical calibration (same logic as nonparametric, but resamples from pool)
# ---------------------------------------------------------------------------

_CALIBRATION_CACHE_EMP = {}
_CALIBRATION_CACHE_MULTIPOINT_EMP = {}


def _mean_rho_empirical(rho_in, template, pool, n_cal, seed):
    """Scalar reference implementation; not used in production (see _mean_rho_empirical_vec).

    Retained for validation and unit testing against the vectorized path.
    """
    cal_rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_cal):
        x = template.copy()
        cal_rng.shuffle(x)
        x_ranks = rankdata(x, method="average")
        nn = len(x_ranks)
        noise_ranks = cal_rng.permutation(nn) + 1.0

        s_x = x_ranks - np.mean(x_ranks)
        sd_x = np.std(x_ranks, ddof=0)
        if sd_x > 0:
            s_x /= sd_x
        s_n = noise_ranks - np.mean(noise_ranks)
        s_n /= np.std(noise_ranks, ddof=0)

        rho_c = np.clip(rho_in, -0.999, 0.999)
        mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

        y_vals = np.sort(pool)
        y_final = np.empty(nn)
        y_final[np.argsort(mixed)] = y_vals

        total += _fast_spearman(x_ranks, y_final)
    return total / n_cal


def _precompute_calibration_arrays_empirical(template, pool, n_cal, seed):
    """Precompute the rho-independent arrays for empirical calibration.

    Identical structure to _precompute_calibration_arrays but uses the
    deterministic empirical pool instead of lognormal draws: y_batch_sorted
    is simply the sorted pool tiled n_cal times.

    Returns
    -------
    s_x : ndarray, shape (n_cal, n)
    s_n : ndarray, shape (n_cal, n)
    y_batch_sorted : ndarray, shape (n_cal, n)
        Sorted pool tiled n_cal times.
    x_ranks_batch : ndarray, shape (n_cal, n)
    """
    cal_rng = np.random.default_rng(seed)
    n = len(template)

    # Batch-shuffle x: (n_cal, n)
    x_batch = np.tile(template, (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        x_batch = cal_rng.permuted(x_batch, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        x_batch = np.take_along_axis(x_batch, perm, axis=1)

    x_ranks_batch = _rank_rows(x_batch)  # (n_cal, n)

    # Noise ranks: independent permutation per row
    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_cal, 1))
    if hasattr(cal_rng, 'permuted'):
        noise_batch = cal_rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(cal_rng.random((n_cal, n)), axis=1)
        noise_batch = np.take_along_axis(noise_base, perm, axis=1)

    # Standardize x_ranks per row
    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)
    s_x /= sd_x

    # Noise ranks are always permutations of 1..n
    noise_mean = (n + 1) / 2.0
    noise_std = np.sqrt((n * n - 1) / 12.0)
    s_n = (noise_batch - noise_mean) / noise_std

    # Pool is deterministic; no RNG needed for y
    y_batch_sorted = np.tile(np.sort(pool), (n_cal, 1))  # (n_cal, n)

    return s_x, s_n, y_batch_sorted, x_ranks_batch


def _mean_rho_empirical_vec(rho_in, template, pool, n_cal, seed):
    """Vectorized mean realised Spearman rho using empirical pool.

    Replaces the n_cal Python loop in _mean_rho_empirical with batch ops.
    The pool is fixed (calibration uses get_pool which has a fixed seed),
    so y_batch is simply the sorted pool tiled n_cal times, permuted by
    the mixed-rank ordering.  ~10-20x faster than the scalar version.

    Convenience wrapper around _precompute_calibration_arrays_empirical +
    _eval_mean_rho.
    """
    s_x, s_n, y_batch_sorted, x_ranks_batch = \
        _precompute_calibration_arrays_empirical(template, pool, n_cal, seed)
    return _eval_mean_rho(rho_in, s_x, s_n, y_batch_sorted, x_ranks_batch)


def _bisect_for_probe_empirical(probe, template, pool, n_cal, seed,
                                n_iter=25, tol=5e-5):
    """Bisect to find rho_in such that _mean_rho_empirical(rho_in, ...) ~ probe.

    Precomputes rho-independent arrays once, then calls _eval_mean_rho for
    each bisection step (~27×), avoiding redundant RNG and x-ranking work.
    """
    arrays = _precompute_calibration_arrays_empirical(template, pool, n_cal, seed)

    lo, hi = 0.0, min(probe * 2.0, 0.999)
    val_hi = _eval_mean_rho(hi, *arrays)
    if val_hi < probe:
        hi = 0.999
        val_hi = _eval_mean_rho(hi, *arrays)
    if val_hi < probe:
        return None

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        observed = _eval_mean_rho(mid, *arrays)
        if observed < probe:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    # Same as _bisect_for_probe: return the endpoint closer to the probe (not
    # the midpoint) because _eval_mean_rho has step discontinuities with
    # heavily-tied x (argsort(mixed) flips at rho_c thresholds).  Prefer hi
    # when equidistant (f(hi) >= probe by invariant).
    val_lo = _eval_mean_rho(lo, *arrays)
    val_hi = _eval_mean_rho(hi, *arrays)
    return hi if abs(val_hi - probe) <= abs(val_lo - probe) else lo


def _calibrate_rho_multipoint_empirical(n, n_distinct, distribution_type,
                                        rho_target, pool, all_distinct=False,
                                        n_cal=300, seed=99, freq_dict=None):
    """Multi-point calibration for empirical marginal."""
    cache_key = ("emp", n, n_distinct, distribution_type, all_distinct, n_cal,
                 "multipoint")
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_MULTIPOINT_EMP:
        template = _get_x_template(n, n_distinct, distribution_type,
                                   freq_dict, all_distinct)
        pairs = []
        for probe in _MULTIPOINT_PROBES:
            rho_in = _bisect_for_probe_empirical(probe, template, pool,
                                                 n_cal, seed)
            if rho_in is not None:
                pairs.append((probe, rho_in))

        _CALIBRATION_CACHE_MULTIPOINT_EMP[cache_key] = pairs

    pairs = sorted(_CALIBRATION_CACHE_MULTIPOINT_EMP[cache_key],
                   key=lambda p: p[0])

    if not pairs:
        return float(np.clip(rho_target, -0.999, 0.999))

    abs_target = abs(rho_target)
    sign = 1.0 if rho_target >= 0 else -1.0

    if len(pairs) == 1:
        probe, rho_in = pairs[0]
        ratio = rho_in / probe
        result = abs_target * ratio
    else:
        probes = [p for p, _ in pairs]
        rho_ins = [r for _, r in pairs]
        result = _interp_with_extrapolation(abs_target, probes, rho_ins)

    return float(np.clip(sign * result, -0.999, 0.999))


def calibrate_rho_empirical(n, n_distinct, distribution_type, rho_target, pool,
                            all_distinct=False, n_cal=300, seed=99,
                            freq_dict=None, calibration_mode="multipoint"):
    """Find input mixing parameter for empirical marginal.

    Same structure as calibrate_rho but uses empirical pool instead of
    lognormal y-marginal.  Separate caches (_CALIBRATION_CACHE_EMP,
    _CALIBRATION_CACHE_MULTIPOINT_EMP) avoid collisions with the
    nonparametric calibration.
    """
    if abs(rho_target) < 1e-12:
        return 0.0

    if calibration_mode == "multipoint":
        return _calibrate_rho_multipoint_empirical(
            n, n_distinct, distribution_type, rho_target, pool,
            all_distinct=all_distinct, n_cal=n_cal, seed=seed,
            freq_dict=freq_dict)

    cache_key = ("emp", n, n_distinct, distribution_type, all_distinct, n_cal)
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_EMP:
        probe = 0.30
        template = _get_x_template(n, n_distinct, distribution_type,
                                   freq_dict, all_distinct)

        lo, hi = 0.0, min(probe * 2.0, 0.999)
        if _mean_rho_empirical(hi, template, pool, n_cal, seed) < probe:
            hi = 0.999

        for _ in range(25):
            mid = (lo + hi) / 2.0
            observed = _mean_rho_empirical(mid, template, pool, n_cal, seed)
            if observed < probe:
                lo = mid
            else:
                hi = mid
            if hi - lo < 5e-5:
                break

        # Return the endpoint whose _mean_rho_empirical is closer to probe.
        val_lo = _mean_rho_empirical(lo, template, pool, n_cal, seed)
        val_hi = _mean_rho_empirical(hi, template, pool, n_cal, seed)
        calibrated_probe = hi if abs(val_hi - probe) <= abs(val_lo - probe) else lo
        ratio = calibrated_probe / probe if probe > 0 else 1.0
        _CALIBRATION_CACHE_EMP[cache_key] = ratio

    ratio = _CALIBRATION_CACHE_EMP[cache_key]
    result = rho_target * ratio
    return float(np.clip(result, -0.999, 0.999))


def warm_calibration_cache(generator, y_params, cases=None,
                           n_distinct_values=None, dist_types=None,
                           freq_dict=None, calibration_mode="multipoint",
                           n_cal=300, seed=99):
    """Pre-build and cache calibration curves for all scenarios in the grid.

    Without this, the first call to run_all_scenarios pays calibration cost
    (~0.3s per unique (n, k, dist_type) combo) while subsequent calls reuse
    the cache — making multi-point G estimation unreliable.

    Parameters
    ----------
    generator : str
        'nonparametric', 'copula', or 'empirical'. Linear has no calibration.
    y_params : dict
        Must contain 'median', 'iqr', 'range' keys.
    cases : dict or None
        Case definitions (default: config.CASES).
    n_distinct_values : list of int or None
        Default: config.N_DISTINCT_VALUES.
    dist_types : list of str or None
        Default: config.DISTRIBUTION_TYPES.
    calibration_mode : str
        'multipoint' or 'single'.
    n_cal : int
        Calibration samples per bisection.
    seed : int
        Seed for calibration RNG.

    Returns
    -------
    int
        Number of calibration entries built.
    """
    from config import CASES, N_DISTINCT_VALUES, DISTRIBUTION_TYPES

    if cases is None:
        cases = CASES
    if n_distinct_values is None:
        n_distinct_values = N_DISTINCT_VALUES
    if dist_types is None:
        dist_types = DISTRIBUTION_TYPES

    if generator == "linear":
        return 0

    probe_rho = 0.30
    built = 0

    for case_id, case in cases.items():
        n = case["n"]
        case_y_params = {"median": case["median"], "iqr": case["iqr"],
                         "range": case["range"]}

        for k in n_distinct_values:
            for dt in dist_types:
                if generator == "nonparametric":
                    calibrate_rho(n, k, dt, probe_rho, case_y_params,
                                  all_distinct=False, n_cal=n_cal, seed=seed,
                                  freq_dict=freq_dict,
                                  calibration_mode=calibration_mode)
                elif generator == "copula":
                    calibrate_rho_copula(n, k, dt, probe_rho, case_y_params,
                                        all_distinct=False, n_cal=n_cal,
                                        seed=seed, freq_dict=freq_dict,
                                        calibration_mode=calibration_mode)
                elif generator == "empirical":
                    pool = get_pool(n)
                    calibrate_rho_empirical(n, k, dt, probe_rho, pool,
                                           all_distinct=False, n_cal=n_cal,
                                           seed=seed, freq_dict=freq_dict,
                                           calibration_mode=calibration_mode)
                built += 1

        # All-distinct scenario
        if generator == "nonparametric":
            calibrate_rho(n, n, None, probe_rho, case_y_params,
                          all_distinct=True, n_cal=n_cal, seed=seed,
                          calibration_mode=calibration_mode)
        elif generator == "copula":
            calibrate_rho_copula(n, n, None, probe_rho, case_y_params,
                                all_distinct=True, n_cal=n_cal, seed=seed,
                                calibration_mode=calibration_mode)
        elif generator == "empirical":
            pool = get_pool(n)
            calibrate_rho_empirical(n, n, None, probe_rho, pool,
                                   all_distinct=True, n_cal=n_cal, seed=seed,
                                   calibration_mode=calibration_mode)
        built += 1

    return built


def generate_y_nonparametric(x, rho_s, y_params, rng=None,
                             _calibrated_rho=None, _ln_params=None):
    """Generate y via calibrated rank-mixing to target Spearman *rho_s*.

    Mixes the standardised ranks of x with independent noise ranks, using
    a calibrated mixing weight so that E[Spearman(x, y)] = rho_s despite
    tie-induced attenuation.  The calibration is computed once per scenario
    and cached.

    Parameters
    ----------
    x : ndarray
        Observed (possibly tied) x-values.
    rho_s : float
        Target Spearman correlation (the *output* rho, after calibration).
    y_params : dict
        Must contain 'median' and 'iqr'.
    rng : numpy.random.Generator or None
    _calibrated_rho : float or None
        Pre-computed calibrated input rho.  If None, uses rho_s directly
        (caller is responsible for calibration).
    _ln_params : tuple or None
        Pre-computed (mu, sigma) from _fit_lognormal.
    """
    if rng is None:
        rng = np.random.default_rng()

    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks = rankdata(x, method="average")
    return _raw_rank_mix(x_ranks, rho_input, y_params, rng,
                         _ln_params=_ln_params)


# ---------------------------------------------------------------------------
# Batch data generation (vectorized over n_sims / n_reps)
# ---------------------------------------------------------------------------

def generate_cumulative_aluminum_batch(n_sims, n, n_distinct,
                                       distribution_type=None, freq_dict=None,
                                       all_distinct=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    template = _get_x_template(n, n_distinct, distribution_type, freq_dict, all_distinct)
    x_batch = np.tile(template, (n_sims, 1))        # (n_sims, n)
    # Independent shuffle per row (permuted in NumPy 1.20+; argsort fallback for older)
    if hasattr(rng, 'permuted'):
        return rng.permuted(x_batch, axis=1)
    perm = np.argsort(rng.random((n_sims, n)), axis=1)
    return np.take_along_axis(x_batch, perm, axis=1)


def _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng,
                        _ln_params=None, marginal="lognormal", pool=None):
    n_sims, n = x_ranks_batch.shape

    # Noise ranks: independent permutation per row (permuted in NumPy 1.20+; argsort fallback)
    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_sims, 1))
    if hasattr(rng, 'permuted'):
        noise_ranks = rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(rng.random((n_sims, n)), axis=1)
        noise_ranks = np.take_along_axis(noise_base, perm, axis=1)
    # Jitter breaks integer commensurability that causes a discrete staircase
    # in E[Spearman(rho_in)] for heavily tied x (e.g. k=4 equal groups).
    # |jitter| < 0.5 guarantees adjacent integers can never cross, preserving
    # the uniform random permutation semantics.  Must match the jitter applied
    # in the calibration precompute functions so that calibrated rho_in values
    # produce the correct E[Spearman] in data generation.
    noise_ranks = noise_ranks + rng.uniform(-0.49, 0.49, size=(n_sims, n))

    # Standardize x_ranks per row
    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)   # avoid divide-by-zero (matches 1D guard)
    s_x /= sd_x

    # Population mean/std for permutations of 1..n remain valid constants
    # after jitter (jitter mean=0, jitter variance 0.49^2/3 << (n^2-1)/12).
    noise_mean = (n + 1) / 2.0
    noise_std = np.sqrt((n * n - 1) / 12.0)
    s_n = (noise_ranks - noise_mean) / noise_std

    rho_c = np.clip(rho_input, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n   # (n_sims, n)

    if marginal == "lognormal":
        if _ln_params is not None:
            mu, sigma = _ln_params
        else:
            mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
        y_values = rng.lognormal(mean=mu, sigma=sigma, size=(n_sims, n))
        y_values.sort(axis=1)
    elif marginal == "empirical":
        if pool is None:
            raise ValueError("pool required for empirical marginal")
        if pool.ndim != 2 or pool.shape != (n_sims, n):
            raise ValueError(
                f"For empirical marginal, pool must be 2D with shape ({n_sims}, {n}), "
                f"got shape {pool.shape}")
        y_values = np.sort(pool, axis=1)
    else:
        raise ValueError(f"Unknown marginal: {marginal}")

    # Assign: for each row, place sorted y_values at the argsort of mixed
    order = np.argsort(mixed, axis=1)                       # (n_sims, n)
    rows = np.arange(n_sims)[:, None]
    y_final = np.empty_like(y_values)
    y_final[rows, order] = y_values

    return y_final


def generate_y_nonparametric_batch(x_batch, rho_s, y_params, rng=None,
                                    _calibrated_rho=None, _ln_params=None):
    if rng is None:
        rng = np.random.default_rng()
    rho_input = _calibrated_rho if _calibrated_rho is not None else rho_s
    x_ranks_batch = _rank_rows(x_batch)     # from spearman_helpers — Numba-accelerated
    return _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng,
                               _ln_params=_ln_params)


def generate_y_copula_batch(x_batch, rho_s, y_params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_sims, n = x_batch.shape
    rho_p = _spearman_to_pearson(rho_s)

    ranks_x = _rank_rows(x_batch)                               # (n_sims, n)
    u_x = (ranks_x - 0.5) / n
    u_x = np.clip(u_x + rng.normal(0, 1e-6, (n_sims, n)), 1e-8, 1 - 1e-8)
    z_x = norm.ppf(u_x)

    z_y = rho_p * z_x + np.sqrt(max(1.0 - rho_p ** 2, 0.0)) * rng.standard_normal((n_sims, n))
    u_y = norm.cdf(z_y)

    mu_ln, sigma_ln = _fit_lognormal(y_params["median"], y_params["iqr"])
    y = _lognormal_quantile(u_y, mu_ln, sigma_ln)
    return y


def generate_y_linear_batch(x_batch, rho_s, y_params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_sims, n = x_batch.shape
    mu_ln, sigma_ln = _fit_lognormal(y_params["median"], y_params["iqr"])

    if abs(rho_s) < 1e-12:
        return np.exp(mu_ln + sigma_ln * rng.standard_normal((n_sims, n)))

    x_std = x_batch - x_batch.mean(axis=1, keepdims=True)
    sx = x_std.std(axis=1, keepdims=True, ddof=0)
    # If all x identical in a row (degenerate), fall back to noise-only
    ok = sx > 1e-12
    sx = np.where(ok, sx, 1.0)
    x_std = x_std / sx

    rho_target = _spearman_to_pearson(rho_s)
    rho_target = np.clip(rho_target, -0.999, 0.999)

    b = rho_target * sigma_ln
    noise_var = sigma_ln ** 2 * (1.0 - rho_target ** 2)
    noise_sd = np.sqrt(max(noise_var, 1e-12))

    log_y = mu_ln + b * x_std + noise_sd * rng.standard_normal((n_sims, n))
    y = np.exp(log_y)
    # Where sx was degenerate, replace with pure noise
    if not np.all(ok):
        bad = ~ok.ravel()
        y[bad] = np.exp(mu_ln + sigma_ln * rng.standard_normal((bad.sum(), n)))
    return y
