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
from spearman_helpers import _rank_rows

GENERATORS = {
    "copula": "generate_y_copula",
    "linear": "generate_y_linear",
    "nonparametric": "generate_y_nonparametric",
}

_NORM_PPF_075 = float(norm.ppf(0.75))


def get_generator(name):
    """Return the y-generation function for *name* ('copula', 'linear', or
    'nonparametric')."""
    funcs = {
        "copula": generate_y_copula,
        "linear": generate_y_linear,
        "nonparametric": generate_y_nonparametric,
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

def _raw_rank_mix(x_ranks, rho_input, y_params, rng, _ln_params=None):
    """Core rank-mixing: returns y with ranks correlated to x_ranks.

    Parameters
    ----------
    _ln_params : tuple or None
        Pre-computed (mu, sigma) from _fit_lognormal.  Avoids repeated
        lookups when called in tight loops.
    """
    n = len(x_ranks)
    noise_ranks = rng.permutation(n) + 1.0

    s_x = x_ranks - np.mean(x_ranks)
    sd_x = np.std(x_ranks, ddof=0)
    if sd_x > 0:
        s_x /= sd_x

    s_n = noise_ranks - np.mean(noise_ranks)
    s_n /= np.std(noise_ranks, ddof=0)

    rho_c = np.clip(rho_input, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n

    if _ln_params is not None:
        mu, sigma = _ln_params
    else:
        mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
    y_values = rng.lognormal(mean=mu, sigma=sigma, size=n)
    y_values.sort()

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
    """Mean realised Spearman rho over n_cal samples for given rho_in."""
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


def _bisect_for_probe(probe, template, y_params, n_cal, seed,
                      n_iter=25, tol=5e-5):
    """Bisect to find rho_in such that _mean_rho(rho_in, ...) ≈ probe.

    Returns rho_in (float) or None if the probe is unreachable.
    """
    lo, hi = 0.0, min(probe * 2.0, 0.999)
    if _mean_rho(hi, template, y_params, n_cal, seed) < probe:
        hi = 0.999
    if _mean_rho(hi, template, y_params, n_cal, seed) < probe:
        return None  # unreachable

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        observed = _mean_rho(mid, template, y_params, n_cal, seed)
        if observed < probe:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return (lo + hi) / 2.0


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
_MULTIPOINT_PROBES = [0.10, 0.30, 0.50]


def _calibrate_rho_multipoint(n, n_distinct, distribution_type, rho_target,
                               y_params, all_distinct=False, n_cal=300,
                               seed=99, freq_dict=None):
    """Multi-point calibration: probe at 3 rho values, interpolate."""
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

        calibrated_probe = (lo + hi) / 2.0
        ratio = calibrated_probe / probe if probe > 0 else 1.0
        _CALIBRATION_CACHE[cache_key] = ratio

    ratio = _CALIBRATION_CACHE[cache_key]
    result = rho_target * ratio
    return float(np.clip(result, -0.999, 0.999))


_CALIBRATION_CACHE_COPULA = {}


def calibrate_rho_copula(n, n_distinct, distribution_type, rho_target, y_params,
                         all_distinct=False, n_cal=300, seed=99, freq_dict=None):
    """Find the input rho_s for the copula that produces E[Spearman] = rho_target.

    Uses the same rho-independent attenuation approach as calibrate_rho:
    bisection at probe rho=0.30 to compute a ratio, cached per (n, k, dist_type),
    then applied as rho_in = rho_target * ratio.

    Parameters
    ----------
    freq_dict : dict or None
        Custom frequency dictionary.  When provided, must contain
        freq_dict[n][n_distinct]["custom"] = list of counts.

    Returns
    -------
    float
        Calibrated input rho_s to feed into generate_y_copula.
    """
    if abs(rho_target) < 1e-12:
        return 0.0

    cache_key = (n, n_distinct, distribution_type, all_distinct, n_cal)
    if freq_dict is not None:
        counts = tuple(freq_dict[n][n_distinct]["custom"])
        cache_key = cache_key + (counts,)

    if cache_key not in _CALIBRATION_CACHE_COPULA:
        probe = 0.30

        template = _get_x_template(n, n_distinct, distribution_type,
                                    freq_dict, all_distinct)

        def _mean_rho(rho_in):
            cal_rng = np.random.default_rng(seed)
            total = 0.0
            for _ in range(n_cal):
                x = template.copy()
                cal_rng.shuffle(x)
                x_ranks = rankdata(x, method="average")
                y = generate_y_copula(x, rho_in, y_params, rng=cal_rng)
                total += _fast_spearman(x_ranks, y)
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

        calibrated_probe = (lo + hi) / 2.0
        ratio = calibrated_probe / probe if probe > 0 else 1.0
        _CALIBRATION_CACHE_COPULA[cache_key] = ratio

    ratio = _CALIBRATION_CACHE_COPULA[cache_key]
    result = rho_target * ratio
    return float(np.clip(result, -0.999, 0.999))


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


def _raw_rank_mix_batch(x_ranks_batch, rho_input, y_params, rng, _ln_params=None):
    n_sims, n = x_ranks_batch.shape

    # Noise ranks: independent permutation per row (permuted in NumPy 1.20+; argsort fallback)
    noise_base = np.tile(np.arange(1.0, n + 1.0), (n_sims, 1))
    if hasattr(rng, 'permuted'):
        noise_ranks = rng.permuted(noise_base, axis=1)
    else:
        perm = np.argsort(rng.random((n_sims, n)), axis=1)
        noise_ranks = np.take_along_axis(noise_base, perm, axis=1)

    # Standardize x_ranks per row
    s_x = x_ranks_batch - x_ranks_batch.mean(axis=1, keepdims=True)
    sd_x = x_ranks_batch.std(axis=1, keepdims=True, ddof=0)
    sd_x = np.where(sd_x > 0, sd_x, 1.0)   # avoid divide-by-zero (matches 1D guard)
    s_x /= sd_x

    # Standardize noise_ranks per row
    s_n = noise_ranks - noise_ranks.mean(axis=1, keepdims=True)
    s_n /= noise_ranks.std(axis=1, keepdims=True, ddof=0)

    rho_c = np.clip(rho_input, -0.999, 0.999)
    mixed = rho_c * s_x + np.sqrt(1.0 - rho_c ** 2) * s_n   # (n_sims, n)

    # Lognormal y-values — MUST use mu, sigma (not default 0, 1)
    if _ln_params is not None:
        mu, sigma = _ln_params
    else:
        mu, sigma = _fit_lognormal(y_params["median"], y_params["iqr"])
    y_values = rng.lognormal(mean=mu, sigma=sigma, size=(n_sims, n))
    y_values.sort(axis=1)

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
