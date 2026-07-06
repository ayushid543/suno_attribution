"""
Incrementality test simulator: answers the JD's "Run marketing experiments"
and "healthy skepticism" requirements directly.

Simulates a geo-holdout test: during the Paid Social spend spike window,
one region keeps seeing Paid Social ads (test) and one region has Paid
Social suppressed (holdout). Compares conversion rate lift to estimate
TRUE incremental impact of Paid Social spend, vs. what platform-reported
last-click numbers would claim.
"""

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(7)

N_PER_REGION = 6000

# Baseline organic conversion propensity (would-have-converted-anyway rate)
BASE_CONV_RATE = 0.045

# TRUE incremental lift Paid Social provides on top of baseline
TRUE_INCREMENTAL_LIFT = 0.35   # +35% relative lift -- this is the "real" answer

# What naive last-click attribution WOULD claim (inflated, from correlated
# high-intent users self-selecting into paid social exposure)
NAIVE_CLAIMED_LIFT = 1.9       # +190% relative lift -- classic last-click inflation

def run_holdout_test():
    # TEST region: exposed to Paid Social during spike window
    test_conversions = np.random.binomial(
        1, BASE_CONV_RATE * (1 + TRUE_INCREMENTAL_LIFT), N_PER_REGION
    )
    # HOLDOUT region: Paid Social suppressed, same baseline propensity
    holdout_conversions = np.random.binomial(
        1, BASE_CONV_RATE, N_PER_REGION
    )

    test_rate = test_conversions.mean()
    holdout_rate = holdout_conversions.mean()
    true_lift = (test_rate - holdout_rate) / holdout_rate

    # Statistical significance (two-proportion z-test)
    count = np.array([test_conversions.sum(), holdout_conversions.sum()])
    nobs = np.array([N_PER_REGION, N_PER_REGION])
    p_pool = count.sum() / nobs.sum()
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
    z = (test_rate - holdout_rate) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    results = {
        "test_region_conv_rate": round(test_rate, 4),
        "holdout_region_conv_rate": round(holdout_rate, 4),
        "measured_incremental_lift": round(true_lift, 3),
        "z_score": round(z, 2),
        "p_value": round(p_value, 5),
        "significant": bool(p_value < 0.05),
        "naive_last_click_claimed_lift": NAIVE_CLAIMED_LIFT,
        "overstatement_factor": round(NAIVE_CLAIMED_LIFT / true_lift, 1) if true_lift > 0 else None,
    }
    return results


if __name__ == "__main__":
    results = run_holdout_test()
    pd.Series(results).to_csv("data/incrementality_test.csv")
    print("=== INCREMENTALITY HOLDOUT TEST RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    print(
        f"\nHeadline: Platform/last-click data implies Paid Social drives "
        f"~{NAIVE_CLAIMED_LIFT:.0%} lift in conversion. The geo-holdout test "
        f"measures TRUE incremental lift at ~{results['measured_incremental_lift']:.0%} "
        f"(p={results['p_value']}). Naive attribution overstates impact by "
        f"~{results['overstatement_factor']}x."
    )
