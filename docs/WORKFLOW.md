# Development Workflow — KarwowskiSpearmanPowerSims

Two parts:
1. **Pre-run checklist** — sequenced action plan to clear known issues before
   the long-running simulation. Do these in order; each builds on the last.
2. **General workflow principles** — rules for this project and future ones,
   grounded in specific failure modes encountered during the audit.

---

# Part 1 — Pre-Run Checklist

Work through these in order. Each section gives the model, the exact prompt
strategy, and what you verify before moving on.

---

## Step 1 — Fix the silent boundary return [MEDIUM risk, ~15 min]

**Model:** Sonnet
**Why Sonnet:** The fix is mechanical — add a `warnings.warn` after a
bisection loop. No statistical judgment needed.

**Prompt strategy:**
> "In `power_simulation.py`, in the `min_detectable_rho` function, after the
> bisection loop ends and before the return statement, add a `warnings.warn`
> if the result is within `tolerance=1e-4` of the search boundary. The
> relevant bounds are the `lo_bound` and `hi_bound` variables used in the
> bisection. Use `warnings.warn(f'...', UserWarning, stacklevel=2)`. Do not
> change any other logic. Show me the diff only."

**What you verify:** Read the diff. Confirm the warning is after the loop,
not inside it. Confirm no other lines changed.

---

## Step 2 — Wire config bounds or delete dead code [LOW risk, ~10 min]

**Model:** Sonnet
**Why Sonnet:** Two options, both mechanical. Choose one before prompting.

**Decision you make first:** Do you want the search bounds to be configurable
via `config.py` (better long-term), or is hardcoding them with a comment
acceptable (simpler)? If configurable, wire `RHO_SEARCH_POSITIVE` and
`RHO_SEARCH_NEGATIVE` from config into `min_detectable_rho`. If not, delete
the config values and add a comment at the hardcoded bounds explaining why
they're tighter than the full range.

**Prompt strategy (wiring option):**
> "In `power_simulation.py`, `min_detectable_rho` currently hardcodes search
> bounds `(0.25, 0.42)` for positive and `(-0.42, -0.25)` for negative. In
> `config.py`, `RHO_SEARCH_POSITIVE = (0.0, 0.6)` and
> `RHO_SEARCH_NEGATIVE = (-0.6, 0.0)` exist but are unused. Import these
> into `power_simulation.py` and use them as the bisection bounds. Do not
> change any other logic."

**What you verify:** Quick sanity check that the new bounds are wide enough
to contain your expected answer (~0.30-0.33). They are — (0.0, 0.6) is
wider than (0.25, 0.42).

---

## Step 3 — Fix double `.sort()` and unused import [LOW risk, ~10 min]

**Model:** Sonnet
**Why Sonnet:** Two one-line deletions.

**Prompt strategy:**
> "In `data_generator.py`, `_raw_rank_mix` around line 443 has a redundant
> `y_values.sort()` call — the empirical branch already sorted at line 440,
> and the lognormal branch needs sorting (keep that). Remove only the
> redundant second sort. Separately, in `_raw_rank_mix_batch` around line
> 999, same issue: remove the redundant `y_values.sort(axis=1)`. Show me
> the diff only, touching no other lines."
>
> Separately: "In `benchmarks/benchmark_full_grid.py`, remove
> `bootstrap_ci_single` from the import line — it is never used in the
> script."

**What you verify:** Diff shows only the two removed lines in
`data_generator.py` and one modified import line in the benchmark.

---

## Step 4 — Fix the `estimate_interrep_sd` script [HIGH risk for planning]

**Background on the bug:** The script runs `bootstrap_ci_averaged` with
`n_reps=200` or `n_reps=1000` and measures SD across 30 replications. But
`bootstrap_ci_averaged` already averages over n_reps datasets internally.
So the script measures `SD(mean of n_reps endpoints)`, which equals
`SD(single endpoint) / sqrt(n_reps)` by the CLT. At n_reps=1000 this gives
~0.003, at n_reps=200 ~0.009 — both far below the true single-endpoint SD
of ~0.12. If you fed these into `benchmark_precision_params.py` as
`SD_INTER_REP`, you would wildly underestimate the n_reps needed for
target precision.

**Correct behavior:** Each of the 30 replications should call
`bootstrap_ci_averaged` with `n_reps=1` (or equivalently
`bootstrap_ci_simulated`), so each observation is a single CI endpoint
with no internal averaging. Then SD across 30 replications estimates the
true single-endpoint SD.

**Spec checkpoint — do this before prompting (2 minutes):**
Write down: "One observation in this SD calculation is: a single call to
bootstrap_ci_averaged with n_reps=1, producing one lower endpoint and one
upper endpoint. SD of 30 such observations estimates the inter-rep SD of
a single CI endpoint." Read that back and confirm it matches the script's
stated goal.

**Model:** Sonnet, but with explicit spec constraint
**Prompt strategy:**
> "In `estimate_interrep_sd.py`, the script currently calls
> `bootstrap_ci_averaged` with `n_reps=N_REPS` (default 200) inside its
> replication loop, then measures SD across replications. This is wrong:
> it measures SD of a mean, not SD of a single observation. Fix it so each
> replication calls `bootstrap_ci_averaged` with `n_reps=1` instead. The
> `--n-reps` CLI argument should be removed or repurposed to control the
> number of replications (outer loop count), not passed to
> `bootstrap_ci_averaged`. Do not change any other logic. Show diff only."

**Verification — the scaling check (mandatory):**
After the fix, run the script at two settings:
```
python scripts/estimate_interrep_sd.py --case 3 --n-distinct 4 \
    --dist-type heavy_center --generator empirical --replications 30
python scripts/estimate_interrep_sd.py --case 3 --n-distinct 4 \
    --dist-type heavy_center --generator empirical --replications 100
```
The reported SD should be stable across replication counts (sampling noise
decreases, but the quantity being estimated doesn't change). It should also
be in the range 0.10–0.13, consistent with the analytical estimate of 0.1216.
If it's near 0.003, the fix didn't work.

**General rule this illustrates:** For any new estimation script, run it at
two very different parameter values where you know the mathematical
relationship between the outputs and verify they scale correctly. The
original error would have been caught immediately by running at n_reps=200
and n_reps=1000 and noticing the outputs differed by exactly `sqrt(5)`.

---

## Step 5 — Add the six missing tests [~45 min total]

Work through these one at a time. Sonnet for all scaffolding; you supply
the expected values.

**General prompt pattern for each test:**
> "Write a pytest test function for [specific behavior]. The function should
> be in `tests/[filename]`. Expected behavior: [your statement of what the
> output should be, not Sonnet's]. Use standard pytest assertions. Do not
> modify any source files."

**The six tests, with your expected values pre-computed:**

**TEST-1 — Boundary warning fires**
Expected: `pytest.warns(UserWarning)` fires when the answer is forced near
a search boundary. Create a minimal scenario where n is tiny enough that
the answer would be near 0.42.

**TEST-2 — `spearman_var_h0` monotonicity**
Expected value you compute first:
```python
from power_asymptotic import spearman_var_h0, get_x_counts
import numpy as np
x_counts = get_x_counts(73, 4, distribution_type="even")
var_ties = spearman_var_h0(73, x_counts)
var_no_ties = spearman_var_h0(73, None)
print(var_ties, var_no_ties)   # var_ties should be >= var_no_ties
```
Run this yourself first. The test asserts `var_ties >= var_no_ties`. If the
print shows the reverse, that is a real bug — escalate to Opus thinking
before writing the test.

**TEST-3 — Negative rho calibration symmetry**
Expected value you compute first:
```python
from data_generator import calibrate_rho
from config import CASES
case = CASES[3]
y_params = {"median": case["median"], "iqr": case["iqr"]}
pos = calibrate_rho(73, 4, "even", +0.30, y_params)
neg = calibrate_rho(73, 4, "even", -0.30, y_params)
print(pos, neg, pos + neg)   # should be near 0
```
Run this first. The test asserts `abs(pos + neg) < 1e-6`.

**TEST-4 — `_fit_lognormal` median recovery**
Expected value is exact: `np.exp(mu)` must equal `median` to floating point
precision, because `mu = np.log(median)` by definition.

**TEST-5 — `_interp_with_extrapolation` edge cases**
Compute expected values yourself:
```python
from data_generator import _interp_with_extrapolation
# Below first probe
xp, fp = [0.10, 0.30, 0.50], [0.12, 0.34, 0.55]
slope_low = (0.34 - 0.12) / (0.30 - 0.10)   # = 1.1
expected = 0.12 + slope_low * (0.05 - 0.10)  # x=0.05
print(expected, _interp_with_extrapolation(0.05, xp, fp))
```
Run this first. Supply the expected values to Sonnet.

**TEST-6 — Asymptotic formulas across all four cases**
Use `@pytest.mark.parametrize("case_id", [1, 2, 3, 4])`. The assertions
are: CI lower < observed_rho < CI upper, CI width > 0, both endpoints in
[-1, 1].

---

## Step 6 — Validate `benchmark_precision_params.py` inputs

Once `estimate_interrep_sd` is fixed and produces ~0.12, verify it matches
the `SD_INTER_REP = 0.13` in `benchmark_precision_params.py`. The 0.13
includes a FHP tie-correction factor; the fixed script's output (no-tie
case) will be slightly lower. This is expected and documented. No code
change needed — just confirm the relationship makes sense before the long run.

---

# Part 2 — General Workflow Principles

---

## Model selection guide

**Use Sonnet (default) for:**
- Implementing a fix where the spec is already written
- Writing test scaffolding when you supply the expected value
- One-line edits, import changes, renaming
- Any task where "what to do" is already decided and the job is "do it"

**Use Opus thinking for:**
- Verifying a formula against its published source
- Any question of the form "is this approximation error negligible?"
- Statistical design questions with a definite right answer
- Spec verification for complex estimation scripts before implementation

**Upgrade to Opus Max only when:**
Opus thinking explicitly flags uncertainty in its own answer. Do not use
Max as a first attempt — it adds cost without benefit when the question
is within Opus thinking's competence. The upgrade signal is a hedged
statement like "I am not certain whether..." or "this would require
verifying against..." in the Opus thinking response. Add this to every
Opus thinking prompt to ensure you get honest uncertainty signals:

> "If you are uncertain about any step, say so explicitly rather than
> proceeding with a plausible-seeming answer. Flag any assumption you
> are making that you cannot verify from the code alone."

Without that instruction, Opus thinking fills gaps with confident-sounding
reasoning. With it, you get hedged statements on uncertain parts.

**On GPT-style models as alternatives:**
Not recommended for this codebase at this stage. The primary risk in your
work is subtle statistical errors that look correct — the copula attenuation
issue, the estimate_interrep_sd spec mismatch, the FHP correction direction.
Catching those requires a model that both knows the statistical literature
and can hold the full reasoning chain without dropping threads. Switching
models mid-project adds a different risk: a new model has no context on
what has already been verified as correct, and may confidently "fix" things
that are right. The cost savings are not worth the re-verification overhead.
Revisit this question for future projects that don't have this audit history.

**On Auto vs Sonnet:**
Use Sonnet explicitly. Auto will sometimes route to Haiku for simple-seeming
tasks and to Opus for complex-seeming ones, but the routing heuristic doesn't
know that a one-line edit in a statistics file may have hidden complexity.
Explicit model selection keeps costs predictable and avoids Auto
under-investing on something that looks simple but isn't.

---

## The spec checkpoint rule

**When it applies:** Any time you are writing a new script whose job is to
*estimate a statistical quantity* — an SD, a power, a calibration ratio,
a confidence interval, a bisection coefficient.

**What you do (2 minutes, before any prompt):**
Write down in one sentence:
> "This script estimates X. One observation is Y. One observation is
> produced by calling Z with parameters W."

Then read it back and verify:
- Does Z produce a single observation, or an aggregate of many?
- If Z internally averages over n things, is X the average or the
  single-observation quantity?
- Do the units/scale of the output of Z match the units of X?

The estimate_interrep_sd failure: the spec sentence would have been
"This script estimates SD of a single CI endpoint. One observation is one
CI endpoint. One observation is produced by calling bootstrap_ci_averaged
with n_reps=200." Reading that back: bootstrap_ci_averaged with n_reps=200
produces the *mean* of 200 endpoints, not a single endpoint. The mismatch
is visible immediately.

**This is something you catch, not the AI.** The AI will write a
consistent, correct implementation of a wrong spec. The spec checkpoint
only works because you read the sentence and apply your domain knowledge.

---

## The scaling check rule

**When it applies:** After fixing or writing any estimation script.

**What you do:** Run the script at two very different values of the main
parameter (n_reps, n_sims, replications) where you know the mathematical
relationship between outputs.

Examples:
- SD estimate: run at replications=30 and replications=120. SD should be
  similar (it's estimating the same quantity). SE of the SD should halve.
- Power estimate: run at n_sims=500 and n_sims=2000. The point estimate
  should be similar; SE should halve.
- Bisection coefficient: run at n_sims=1000 and n_sims=5000. c should
  converge.

If the outputs change by more than sampling noise predicts, the script is
measuring the wrong thing or has a parameter-coupling bug.

---

## The iteration question

**Iterating within a model (Sonnet):** Yes, worth it. Typical pattern:
1. First prompt: implement the fix
2. Diff looks wrong: second prompt with more constraints ("you changed line
   X which I said not to touch — revert that, keep only the change to line Y")
3. Rarely need a third pass

Budget two iterations per fix as the default. If you're on a third
iteration, stop and read the code yourself — Sonnet is probably confused
about something structural that needs a clearer spec from you, not more
prompting.

**Iterating across models (Sonnet → Opus):** Not an iteration strategy —
a routing decision. If Sonnet produces something statistically wrong on the
second attempt, the issue is that the task needed Opus, not that Sonnet
needed another try. Escalate rather than retry.

**Iterating within Opus thinking:** One well-framed prompt should be
sufficient. If the answer is unsatisfying, the problem is almost always
that the question was too broad. Narrow the question and re-prompt, don't
ask Opus to "try again." Example: "is this approximation negligible?" →
too broad. "By how many standard deviations does the permutation null
shift when 5 out of 73 y values are duplicated?" → answerable.

---

## Cost-effective session structure

**One edit per Cursor session.** Batching multiple edits ("also fix this,
and add this test, and rename this variable") causes Sonnet to make
unintended changes elsewhere to satisfy all constraints simultaneously.
The marginal cost of an extra session is low; the cost of disentangling
unintended edits is high.

**Feed only what's needed.** For a fix in `power_simulation.py`, paste
only the relevant function, not the whole file. For an Opus statistical
audit, paste the specific function and the formula question, not the whole
module. Attention is finite — smaller context means the model focuses on
what matters.

**Reuse verified context across sessions.** Keep `AUDIT.md` open as a
Cursor context file. Point Cursor to it at the start of each session with:
"Refer to AUDIT.md for what has already been verified as correct. Do not
re-audit confirmed-correct code."

---

## OPT-1 — Precomputed null for empirical generator (status: blocked)

This optimization requires answering a statistical question before any code
is written. The question is non-trivial and the answer is not known.

**The optimization:** Route the empirical generator through the precomputed
null (keyed on x tie structure) instead of running per-dataset MC
permutations. This would give the same ~60× speedup the other generators
enjoy.

**Why it might work:** The permutation null depends on x's tie structure,
not on y's marginal distribution. For all-distinct y (lognormal draws), the
precomputed null is essentially exact. Empirical y has 2–10 repeated values
from resampling out of 73–82 — these y ties could shift the null, but the
shift may be small enough to not affect power estimates at the ±0.01
precision target.

**Why it might not work:** Whether the shift is negligible has not been
established. "A handful of ties in 73 values is probably fine" is not
sufficient justification for a method paper. The shift needs to be quantified.

**Before any code is written, resolve this question with Opus thinking:**

Prompt structure:
> "I have a precomputed permutation null built by permuting all-distinct
> y ranks (1..n). I want to use this same null for a generator where y has
> k repeated values (k ≈ 2 for n=73, k ≈ 10 for n=81) due to bootstrap
> resampling. Quantify: how much does the permutation null distribution
> shift when k out of n y values are duplicated, compared to all-distinct y?
> Does this shift affect p-values enough to change power estimates at the
> ±0.01 halfwidth precision target?
> [Paste `_build_precomputed_null`, `build_empirical_pool`, and the
> p-value lookup formula]
> Flag any assumption you cannot verify from the code alone."

**If Opus thinking flags uncertainty** in the quantification step,
escalate to Opus Max for that one question only.

**If Opus thinking says negligible:** Implement with Sonnet, then run a
direct comparison — estimate power at one scenario using the current
per-dataset MC path and the proposed precomputed null path. They should
agree within ±0.005. If they do, the optimization is valid.

**If Opus thinking says non-negligible:** Either keep per-dataset MC for
empirical, or explore building a null that accounts for y ties (more
complex, probably not worth it).

Do not implement OPT-1 speculatively. The optimization is only worth
pursuing if it's statistically valid, and validity is currently unknown.

---

## Workflow for future projects

The principles above generalize. For any new project involving statistical
simulation code:

**At design time (before any code):**
1. Write the spec sentence for every estimation function
2. For any formula from the literature, identify the source before
   implementation, not after
3. For any approximation, identify explicitly what you are approximating
   and what the error bound is

**During implementation:**
4. Use Sonnet for implementation, Opus thinking for formula verification
5. Run the scaling check for every estimation script
6. Keep a living AUDIT.md from the start — add to it as you verify things,
   not in a retrospective pass at the end

**Before a long run:**
7. Work through the pre-run checklist pattern from Part 1
8. Run one quick smoke-test at lightweight parameters (n_sims=50, n_reps=5)
   to confirm nothing crashes before committing to the full run
9. If any output of the long run will be used in a paper, have the
   statistical core (formulas and their derivations) reviewed by Opus
   thinking before the run, not after

**The single most important rule:** The AI implements specs. You write specs.
Any error that is fundamentally a mismatch between what you wanted and what
was built is your responsibility to catch at spec-writing time, not the AI's
responsibility to catch during implementation. The spec checkpoint is the
mechanism for doing that.

---

# Part 3 — Disk Cache Workflow

## When to use the precompute script

Building calibration and null caches in-process takes time that scales with
`n_cal` and `n_pre`.  At the precision tiers that matter most (±0.001,
`n_cal ≈ 96 400`, `n_pre ≈ 500 000`), sequential cache build can dominate
total runtime.  Precomputing once and loading from disk on subsequent runs
eliminates this cost.

**Scenarios where precomputing pays off:**
- Copula or empirical generator (calibration is slower than nonparametric).
- ±0.001 precision tier: `n_cal = 96 400`, `n_pre = 500 000`.
- Repeated runs (CI and power both need the same calibration cache).

---

## Precomputing caches

Run from the project root. The script warms up Numba at the start (one small
run), so no separate warm-up script is needed:

```
python scripts/precompute_caches.py --generators nonparametric --n-cal 96400 \
    --null --n-pre 500000 --n-jobs 4 --output-dir cache/
```

To precompute for multiple generators in one run (e.g. nonparametric and copula):

```
python scripts/precompute_caches.py --generators nonparametric,copula --n-cal 96400 \
    --null --n-pre 500000 --n-jobs 4 --output-dir cache/
```

This writes:
- `cache/calibration_ncal96400.pkl`
- `cache/null_npre500000.pkl`

**Memory note:** at `n_cal = 500 000`, each worker holds ~640 MB–1 GB of
arrays.  With 4 workers: 2.5–4 GB RAM required.  Lower `--n-jobs` if RAM
is limited.

**Numba note:** workers spawn fresh processes on Windows; Numba re-loads its
JIT cache from disk (~1–2 s per worker once compiled).  Running
`warm_up_numba.py` first ensures the cache is populated.

The script prints wall-clock timing and file sizes after each phase.

---

## Using disk caches in a run

Pass `disk_cache_dir` to `run_all_scenarios()` or `run_all_ci_scenarios()`:

```python
from power_simulation import run_all_scenarios

results = run_all_scenarios(
    generator="nonparametric",
    n_cal=96400,
    n_sims=222050,
    n_jobs=4,
    disk_cache_dir="cache/",   # loads calibration_ncal96400.pkl + null_npre*.pkl
)
```

The simulation loads both caches before the `pre_warm` step.  If the load
succeeds, `pre_warm` is instant (88 dict lookups, microseconds).

---

## In-process pre-warm: default behavior without disk caches

**Without `disk_cache_dir`**, nothing about the pre-warm behavior changes.
Both `run_all_scenarios()` and `run_all_ci_scenarios()` still pre-warm
calibration (and null) caches in the main process before running scenarios,
exactly as they did before the disk-cache feature was added.  This applies
to both sequential (`n_jobs=1`) and parallel (`n_jobs > 1`) runs:

- **Sequential:** pre_warm builds all cache entries upfront; subsequent
  scenario calls are instant dict lookups.
- **Parallel:** pre_warm builds all entries in the main process, snapshots
  them, and injects the snapshots into workers via an initializer.  Workers
  receive complete caches and skip all build cost.

Setting `disk_cache_dir` adds an optional *fast path*: the disk load runs
first, populating the module-level dicts, and then the pre_warm calls become
no-ops (cache already full).  On a miss or mismatch, pre_warm runs normally
and builds everything in-process.

**The only way to skip pre-warm entirely is `pre_warm=False`**, which was
the intended opt-out before the disk-cache feature and remains so.  Use it
only when you have pre-warmed caches manually and want to skip even the 88
dict-lookup cost.  With `pre_warm=False` and no successful disk load,
sequential runs build cache entries on demand and parallel workers start
cold (each rebuilds its own entries, results are lost when the worker exits).

---

## Choosing pre_warm and disk_cache_dir (parameter guide)

The interaction of `pre_warm` and `disk_cache_dir` can be confusing.  This
section clarifies what each parameter controls and how to choose them.

**What each parameter controls**

- **`disk_cache_dir`** — When set, the simulation *always* tries to load
  calibration and null caches from disk *before* any pre-warm step.  Load
  is independent of `pre_warm`.  If the load succeeds (matching n_cal/n_pre
  and config hash), the in-memory caches are full; if it fails or is not
  set, caches start empty (or partial).

- **`pre_warm`** — Controls only whether the *in-process warm step* runs
  (building all cache entries for the grid).  It does *not* control disk
  load or whether parallel workers receive a snapshot.  Workers always
  receive a snapshot of whatever is in the main process after (optional)
  disk load and (optional) warm.

**Implications**

- **pre_warm=False + successful disk load:** Caches are full from disk;
  parallel workers get that snapshot.  Runtime behavior is the same as
  pre_warm=True (no lazy builds during the run).  The only differences:
  (1) we never run the warm step (no fill-in of missing keys if the disk
  load was partial); (2) we *never* save caches back to disk when
  pre_warm=False — the save logic is gated on `pre_warm`.

- **pre_warm=False + no disk load (or load failed):** Caches stay empty;
  entries are built on first use (lazy).  Parallel workers start cold and
  may rebuild the same entries redundantly.

**Which parameters to choose**

| Goal | disk_cache_dir | pre_warm |
|------|----------------|----------|
| Normal run, no disk cache | omit or None | True (default) |
| Use pre-built disk cache (e.g. from precompute_caches.py) | path to dir | True (default) — load then optional warm; can save if we built anything |
| Use disk cache, never run warm step, never save back | path to dir | False — load still runs; workers get snapshot; we never save |
| Minimal overhead (you already filled caches manually) | optional | False — if no disk path, workers start cold |

**Possible future change:** The current behavior (disk load independent of
pre_warm; snapshot always sent to workers) may be refined so that
pre_warm=False has a clearer meaning (e.g. “do not populate or inject
caches”) or so that save-to-disk is not gated on pre_warm.  See the backlog
plan for a reminder to revisit.

---

## Read-only by default; opt-in to save

The simulation **never writes** cache files unless you explicitly ask it to.
The default from `config.SAVE_CACHE_TO_DISK` is `False`.

To persist caches built in-process (e.g. no file existed, or load failed):

```python
# Option A — one-off via parameter
results = run_all_scenarios(
    ...,
    disk_cache_dir="cache/",
    save_cache_to_disk=True,
)

# Option B — set in config.py for all future runs
SAVE_CACHE_TO_DISK = True
```

With `save_cache_to_disk=True` and `disk_cache_dir` set, after the run
completes the calibration (and null, for power runs) caches are written to
the named files.

---

## Warning behavior

When a disk cache file exists but its metadata does not match the current
configuration, a `UserWarning` fires and the cache is ignored:

- **n_cal mismatch** — file was built with a different `n_cal`.
- **config hash mismatch** — `CASES`, `FREQ_DICT`, `N_DISTINCT_VALUES`, or
  `DISTRIBUTION_TYPES` changed since the file was built.

In both cases the simulation falls through to normal in-process cache
building.  Re-run `precompute_caches.py` with the correct parameters to
refresh the file.

---

## When to re-precompute after config changes

The right action depends on what changed:

| What changed | Effect on disk cache | Action |
|---|---|---|
| `n_cal` or `n_pre` | Hard mismatch — disk cache is rejected entirely | Full re-precompute with new parameters |
| New case added to `CASES` | Config hash changes; existing entries still valid, new ones missing | Re-precompute (incremental: load-before-warm fills only missing entries) |
| Case removed from `CASES` | Config hash changes; orphan entries accumulate silently | `--cleanup` to prune orphaned entries, or full re-precompute |
| `FREQ_DICT` frequencies changed for an existing `(n, k, dist_type)` | Config hash changes; affected entries are stale (wrong x_counts) | `--cleanup` removes stale entries; then re-precompute to rebuild them |
| `N_DISTINCT_VALUES` or `DISTRIBUTION_TYPES` changed | Config hash changes; added/removed combos | `--cleanup` + re-precompute |

**Key rule:** config hash mismatch is a soft warning — loading proceeds.
Stale entries are not automatically purged on load; they remain on disk
until removed by `--cleanup` or a full re-precompute.

---

## Cleanup: removing stale entries without a full rebuild

The `--cleanup` flag scans all `calibration_ncal*.pkl` and `null_npre*.pkl`
files in `--output-dir` and removes entries whose x_counts no longer match
the current `FREQ_DICT` / grid, without rebuilding them.  Use it after a
`FREQ_DICT` change or a scenario grid change where you want to prune stale
entries before the next precompute run.

**Preview without modifying files:**

```
python scripts/precompute_caches.py --cleanup --dry-run --output-dir cache/
```

**Remove stale entries in place:**

```
python scripts/precompute_caches.py --cleanup --output-dir cache/
```

After cleanup, any pruned entries will be rebuilt on the next
`precompute_caches.py` run (or lazily on first use if you skip recomputing).

**What each cleanup function checks:**

*Calibration (`calibration_ncal*.pkl`):* Every non-all_distinct,
non-custom disk entry carries a `counts_tuple` embedded in its key.  If
that tuple no longer matches `FREQ_DICT[n][k][dist_type]`, the entry is
stale and removed.  All-distinct entries and entries with
`dist_type="custom"` are always kept (cannot validate automatically).

*Null (`null_npre*.pkl`):* Each file records which entries were generated
from the standard config grid at save time (as `standard_keys` in the file
metadata).  An entry is stale if it was in `standard_keys` but its
x_counts_tuple no longer matches any current `(n, k, dist_type)` grid
combination.  Entries not recorded as standard — e.g. built with a custom
`freq_dict` — are never flagged, even if they would not be generated by the
current config.

---

## Precision targets and n_cal / n_pre reference

See `docs/UNCERTAINTY_BUDGET.md` Part 6 for the full analysis linking
precision tier half-widths to the required `n_cal` and `n_pre` values and
expected cache build times.
