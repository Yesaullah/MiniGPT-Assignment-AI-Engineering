"""Coverage of 90% intervals for the mean of n = 7 paired differences (D15e, D24m).

Assumes normally distributed differences, so it does not establish coverage for the
actual 7 hand-picked prompts with bounded author ratings.
Run: python docs/analysis/ci_coverage_sim.py
"""
import numpy as np
from scipy import stats

N, TRIALS, RESAMPLES, LEVEL = 7, 4000, 4000, 0.90

rng = np.random.default_rng(0)
t_crit = stats.t.ppf(1 - (1 - LEVEL) / 2, N - 1)
lo_q, hi_q = 100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2
cover_boot = cover_t = 0
for _ in range(TRIALS):
    x = rng.normal(0.0, 1.0, N)
    means = rng.choice(x, (RESAMPLES, N)).mean(axis=1)
    lo, hi = np.percentile(means, [lo_q, hi_q])
    cover_boot += lo <= 0.0 <= hi
    half = t_crit * x.std(ddof=1) / np.sqrt(N)
    cover_t += x.mean() - half <= 0.0 <= x.mean() + half

print(f"percentile bootstrap: {cover_boot / TRIALS:.3f}")
print(f"t-interval:           {cover_t / TRIALS:.3f}")
