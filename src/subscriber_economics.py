"""
Subscriber economics: LTV, retention curves, and payback period by
acquisition channel -- directly answers the JD's "Analyze subscriber
economics" requirement.
"""

import pandas as pd
import numpy as np

users = pd.read_csv("data/users.csv")
spend = pd.read_csv("data/spend.csv")

CHANNELS = ["Paid Social", "Paid Search", "Influencer", "Organic", "Referral", "Direct"]
MONTHLY_PRICE = 12.0

paid = users[users["paid_converted"]].copy()

# --- LTV and retention by first-touch (true driving) channel ---
by_channel = paid.groupby("first_touch_channel").agg(
    paid_subs=("user_id", "count"),
    avg_retained_months=("retained_months", "mean"),
    avg_ltv=("ltv", "mean"),
    total_ltv=("ltv", "sum"),
).reindex(CHANNELS).fillna(0)

# CAC per channel using TRUE contribution (first-touch) conversions
total_spend_by_channel = spend.groupby("channel")["spend"].sum().reindex(CHANNELS).fillna(0)
true_conversions_by_channel = users[users["paid_converted"]].groupby(
    "first_touch_channel"
)["user_id"].count().reindex(CHANNELS).fillna(0)

by_channel["spend"] = total_spend_by_channel
by_channel["CAC"] = np.where(
    true_conversions_by_channel > 0,
    total_spend_by_channel / true_conversions_by_channel.replace(0, np.nan),
    np.nan,
)
by_channel["LTV_to_CAC_ratio"] = (by_channel["avg_ltv"] / by_channel["CAC"]).round(2)

# Payback period in months = CAC / monthly price (simplified, no discounting)
by_channel["payback_months"] = (by_channel["CAC"] / MONTHLY_PRICE).round(1)

# --- Retention curve (cohort-style, simplified: % of paid subs retained at month N) ---
retention_curve = {}
for months_mark in [1, 3, 6, 12, 18, 24]:
    retention_curve[months_mark] = (paid["retained_months"] >= months_mark).mean()

retention_df = pd.Series(retention_curve, name="pct_retained")
retention_df.index.name = "month"

by_channel.to_csv("data/subscriber_economics_by_channel.csv")
retention_df.to_csv("data/retention_curve.csv")

print("=== SUBSCRIBER ECONOMICS BY CHANNEL (true/first-touch attribution) ===")
print(by_channel[["paid_subs", "avg_ltv", "CAC", "LTV_to_CAC_ratio", "payback_months"]].round(2))
print("\n=== OVERALL RETENTION CURVE ===")
print(retention_df.round(3))

best_ltv_cac = by_channel["LTV_to_CAC_ratio"].idxmax()
worst_ltv_cac = by_channel[by_channel["paid_subs"] > 5]["LTV_to_CAC_ratio"].idxmin()
print(f"\nBest LTV:CAC channel: {best_ltv_cac} ({by_channel.loc[best_ltv_cac, 'LTV_to_CAC_ratio']}x)")
print(f"Worst LTV:CAC channel (min 5 subs): {worst_ltv_cac} ({by_channel.loc[worst_ltv_cac, 'LTV_to_CAC_ratio']}x)")
