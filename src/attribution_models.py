"""
Attribution modeling: compares what platforms/last-click would report
vs. a data-driven multi-touch model vs. the TRUE underlying channel
contribution (known here because we generated the data).

This is the core "reconcile platform-reported numbers against reality"
piece the Suno JD asks for.
"""

import pandas as pd
import numpy as np

users = pd.read_csv("data/users.csv")
touchpoints = pd.read_csv("data/touchpoints.csv")
spend = pd.read_csv("data/spend.csv")

CHANNELS = ["Paid Social", "Paid Search", "Influencer", "Organic", "Referral", "Direct"]

converted = users[users["paid_converted"]].copy()

# ---------------------------------------------------------------------------
# 1. LAST-CLICK ATTRIBUTION (what most platforms report by default)
# ---------------------------------------------------------------------------
last_click = converted.groupby("last_touch_channel").agg(
    conversions=("user_id", "count"),
    total_ltv=("ltv", "sum"),
).reindex(CHANNELS).fillna(0)
last_click.index.name = "channel"

# ---------------------------------------------------------------------------
# 2. LINEAR MULTI-TOUCH ATTRIBUTION (equal credit across all touchpoints
#    in a converting user's journey)
# ---------------------------------------------------------------------------
converted_ids = set(converted["user_id"])
conv_touchpoints = touchpoints[touchpoints["user_id"].isin(converted_ids)].copy()

# credit = 1 / n_touches for that user, per touchpoint
touch_counts = conv_touchpoints.groupby("user_id")["channel"].transform("count")
conv_touchpoints["credit"] = 1.0 / touch_counts

ltv_map = converted.set_index("user_id")["ltv"].to_dict()
conv_touchpoints["ltv_credit"] = conv_touchpoints.apply(
    lambda r: r["credit"] * ltv_map.get(r["user_id"], 0), axis=1
)

linear_mta = conv_touchpoints.groupby("channel").agg(
    conversions=("credit", "sum"),
    total_ltv=("ltv_credit", "sum"),
).reindex(CHANNELS).fillna(0)
linear_mta.index.name = "channel"

# ---------------------------------------------------------------------------
# 3. TIME-DECAY MULTI-TOUCH ATTRIBUTION (touches closer to conversion get
#    more credit -- a more realistic model than pure linear)
# ---------------------------------------------------------------------------
conv_touchpoints["touch_order"] = conv_touchpoints["touch_order"].astype(int)
max_touch = conv_touchpoints.groupby("user_id")["touch_order"].transform("max")
decay_weight = 2 ** (conv_touchpoints["touch_order"] - 1)  # more recent = more weight
weight_sum = conv_touchpoints.groupby("user_id")["touch_order"].transform(
    lambda x: sum(2 ** (i) for i in range(len(x)))
)
conv_touchpoints["decay_credit"] = decay_weight / weight_sum
conv_touchpoints["decay_ltv_credit"] = conv_touchpoints.apply(
    lambda r: r["decay_credit"] * ltv_map.get(r["user_id"], 0), axis=1
)

time_decay_mta = conv_touchpoints.groupby("channel").agg(
    conversions=("decay_credit", "sum"),
    total_ltv=("decay_ltv_credit", "sum"),
).reindex(CHANNELS).fillna(0)
time_decay_mta.index.name = "channel"

# ---------------------------------------------------------------------------
# 4. "TRUE" CONTRIBUTION -- known because we control the simulation.
#    Approximated here via first-touch channel, since that's what
#    actually drove the conversion probability in generate_data.py.
# ---------------------------------------------------------------------------
true_contribution = converted.groupby("first_touch_channel").agg(
    conversions=("user_id", "count"),
    total_ltv=("ltv", "sum"),
).reindex(CHANNELS).fillna(0)
true_contribution.index.name = "channel"

# ---------------------------------------------------------------------------
# 5. SPEND + EFFICIENCY (CAC) BY MODEL
# ---------------------------------------------------------------------------
total_spend_by_channel = spend.groupby("channel")["spend"].sum().reindex(CHANNELS).fillna(0)

def add_cac(df, spend_series):
    out = df.copy()
    out["spend"] = spend_series
    out["CAC"] = np.where(out["conversions"] > 0, out["spend"] / out["conversions"], np.nan)
    out["LTV_to_CAC"] = np.where(out["CAC"] > 0, out["total_ltv"] / (out["conversions"] * out["CAC"].replace(0, np.nan)), np.nan)
    return out

last_click = add_cac(last_click, total_spend_by_channel)
linear_mta = add_cac(linear_mta, total_spend_by_channel)
time_decay_mta = add_cac(time_decay_mta, total_spend_by_channel)
true_contribution = add_cac(true_contribution, total_spend_by_channel)

# ---------------------------------------------------------------------------
# SAVE OUTPUTS
# ---------------------------------------------------------------------------
last_click.to_csv("data/model_last_click.csv")
linear_mta.to_csv("data/model_linear_mta.csv")
time_decay_mta.to_csv("data/model_time_decay_mta.csv")
true_contribution.to_csv("data/model_true_contribution.csv")

print("=== LAST-CLICK (what platforms report) ===")
print(last_click[["conversions", "CAC"]].round(1))
print("\n=== TIME-DECAY MULTI-TOUCH ===")
print(time_decay_mta[["conversions", "CAC"]].round(1))
print("\n=== TRUE CONTRIBUTION (ground truth) ===")
print(true_contribution[["conversions", "CAC"]].round(1))

# Quantify the discrepancy that matters most: Paid Social over-crediting
lc_paid_social_share = last_click.loc["Paid Social", "conversions"] / last_click["conversions"].sum()
true_paid_social_share = true_contribution.loc["Paid Social", "conversions"] / true_contribution["conversions"].sum()
print(f"\nPaid Social share of conversions -- Last-click: {lc_paid_social_share:.1%} | True: {true_paid_social_share:.1%}")
print(f"Discrepancy: {(lc_paid_social_share - true_paid_social_share):.1%} points of OVER-credit to Paid Social")
