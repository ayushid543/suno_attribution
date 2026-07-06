"""
Signal vs. Noise: Multi-Touch Attribution Audit & Budget Reallocation Engine

A Streamlit dashboard built for the Suno Marketing Data Scientist role.
Ties together:
  1. Attribution model comparison (last-click vs multi-touch vs true contribution)
  2. An incrementality holdout test quantifying how much last-click overstates impact
  3. Subscriber economics (LTV, CAC, payback) by true acquisition channel
  4. A concrete budget reallocation recommendation

Run with: streamlit run src/app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Signal vs. Noise | Attribution Audit", layout="wide")

CHANNELS = ["Paid Social", "Paid Search", "Influencer", "Organic", "Referral", "Direct"]
COLORS = {
    "Paid Social": "#E86A5C", "Paid Search": "#F2B84B", "Influencer": "#5B9BD5",
    "Organic": "#70AD47", "Referral": "#7C5FC4", "Direct": "#A6A6A6",
}

@st.cache_data
def load_data():
    last_click = pd.read_csv("data/model_last_click.csv", index_col="channel")
    time_decay = pd.read_csv("data/model_time_decay_mta.csv", index_col="channel")
    true_contrib = pd.read_csv("data/model_true_contribution.csv", index_col="channel")
    econ = pd.read_csv("data/subscriber_economics_by_channel.csv", index_col="first_touch_channel")
    retention = pd.read_csv("data/retention_curve.csv", index_col="month")
    incr_raw = pd.read_csv("data/incrementality_test.csv", index_col=0).squeeze("columns")
    # CSV round-trip mixes numeric and boolean values, so pandas reads the
    # whole Series as object/strings. Cast the numeric fields back to float
    # explicitly (leave 'significant' as-is since it's just displayed as text).
    incr = incr_raw.copy()
    numeric_fields = [
        "test_region_conv_rate", "holdout_region_conv_rate",
        "measured_incremental_lift", "z_score", "p_value",
        "naive_last_click_claimed_lift", "overstatement_factor",
    ]
    for field in numeric_fields:
        incr[field] = float(incr[field])
    return last_click, time_decay, true_contrib, econ, retention, incr

last_click, time_decay, true_contrib, econ, retention, incr = load_data()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("📡 Signal vs. Noise")
st.caption("Multi-Touch Attribution Audit & Budget Reallocation Engine — built for Suno's Marketing Data Scientist role, using synthetic subscription-app data modeled on a realistic acquisition mix.")

st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 1: THE DISCREPANCY
# ---------------------------------------------------------------------------
st.header("1. What last-click tells you vs. what's actually true")

col1, col2 = st.columns([2, 1])

with col1:
    comparison = pd.DataFrame({
        "Last-Click (platform default)": last_click["conversions"],
        "Time-Decay Multi-Touch": time_decay["conversions"],
        "True Contribution (ground truth)": true_contrib["conversions"],
    }).reindex(CHANNELS)

    fig = go.Figure()
    for col in comparison.columns:
        fig.add_trace(go.Bar(name=col, x=comparison.index, y=comparison[col]))
    fig.update_layout(barmode="group", height=420, legend=dict(orientation="h", y=-0.2),
                       yaxis_title="Attributed conversions", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Influencer — last-click credit", f"{int(last_click.loc['Influencer','conversions'])} conversions",
              delta=f"{int(true_contrib.loc['Influencer','conversions'] - last_click.loc['Influencer','conversions'])} vs. true", delta_color="inverse")
    st.metric("Paid Search — last-click credit", f"{int(last_click.loc['Paid Search','conversions'])} conversions",
              delta=f"{int(true_contrib.loc['Paid Search','conversions'] - last_click.loc['Paid Search','conversions'])} vs. true", delta_color="normal")
    st.info("**Reading this:** Influencer starts far more journeys than it ever gets credit for, because Paid Search usually captures the last click. Last-click makes Paid Search look like the hero — it's often just closing out demand Influencer already created.")

# ---------------------------------------------------------------------------
# SECTION 2: INCREMENTALITY TEST
# ---------------------------------------------------------------------------
st.header("2. Does the spend actually cause the lift? (Incrementality holdout test)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test region conv. rate", f"{incr['test_region_conv_rate']:.2%}")
c2.metric("Holdout region conv. rate", f"{incr['holdout_region_conv_rate']:.2%}")
c3.metric("Measured true lift", f"{incr['measured_incremental_lift']:.0%}", help=f"p-value = {incr['p_value']}")
c4.metric("Naive/last-click claimed lift", f"{incr['naive_last_click_claimed_lift']:.0%}",
          delta=f"{incr['overstatement_factor']}x overstated", delta_color="inverse")

st.warning(
    f"**Healthy skepticism, quantified:** a geo-holdout test (Paid Social suppressed in one region "
    f"during the spend-spike window) measures true incremental lift at "
    f"{incr['measured_incremental_lift']:.0%} (statistically significant, p={incr['p_value']}). "
    f"Naive last-click data would suggest {incr['naive_last_click_claimed_lift']:.0%} lift — "
    f"roughly a **{incr['overstatement_factor']}x overstatement** of true impact."
)

# ---------------------------------------------------------------------------
# SECTION 3: SUBSCRIBER ECONOMICS
# ---------------------------------------------------------------------------
st.header("3. Subscriber economics by true acquisition channel")

econ_display = econ.reindex(CHANNELS).copy()
econ_display["LTV_to_CAC_ratio"] = econ_display["LTV_to_CAC_ratio"].replace([np.inf, -np.inf], np.nan)

col1, col2 = st.columns(2)
with col1:
    fig2 = px.bar(econ_display.reset_index(), x="first_touch_channel", y="payback_months",
                  color="first_touch_channel", color_discrete_map=COLORS,
                  labels={"first_touch_channel": "Channel", "payback_months": "Payback period (months)"})
    fig2.update_layout(showlegend=False, height=380, title="Payback period by channel (lower is better)")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.line(retention.reset_index(), x="month", y="pct_retained", markers=True)
    fig3.update_layout(height=380, title="Overall subscriber retention curve",
                        yaxis_tickformat=".0%", yaxis_title="% retained", xaxis_title="Months since conversion")
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Payback for Paid Social and Paid Search runs 150+ months at current CAC — Influencer and Referral pay back in under 26 months. This is the same story the attribution chart tells, from a different angle.")

# ---------------------------------------------------------------------------
# SECTION 4: BUDGET REALLOCATION RECOMMENDATION
# ---------------------------------------------------------------------------
st.header("4. Recommended budget reallocation")

st.markdown("""
Based on true contribution, the incrementality test, and payback economics together:

- **Cut Paid Social spend during non-promotional periods** — its last-click volume is inflated by
  spend-spike-driven low-intent traffic; the holdout test shows real incremental lift is a fraction
  of what platform reporting implies.
- **Shift freed budget toward Influencer partnerships** — under-credited by last-click, but drives
  the most true conversions and the second-best payback period in the dataset.
- **Protect and grow Referral** — smallest volume channel, but the best LTV:CAC ratio by a wide margin;
  worth testing incentive increases.
- **Keep Paid Search spend, but stop crediting it for demand it didn't create** — reframe it as a
  closing/retargeting channel in planning, not a demand-generation channel.
""")

st.markdown("---")
st.caption("All data in this project is synthetic, generated to mirror realistic attribution bias patterns (spend-spike-driven inflation, first-touch vs. last-touch sequencing bias). Methodology is fully documented in the project README.")
