"""
Synthetic data generator for the "Signal vs. Noise" attribution project.

Simulates a consumer subscription app (Suno-style: AI music generation) with:
- Users arriving via multiple marketing channels
- Multi-touch journeys before converting to a paid subscription
- A deliberate mid-period Paid Social spend spike that inflates last-click credit
- Channel-sequencing bias (Influencer/Organic often initiate awareness,
  Paid Search/Direct often get the last click)
- Subscription behavior: trial start, paid conversion, monthly retention/churn
- Two geographic regions used later for an incrementality holdout test

Output: data/touchpoints.csv, data/users.csv, data/spend.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_USERS = 20000
START_DATE = datetime(2026, 1, 1)
N_DAYS = 90
SPEND_SPIKE_START_DAY = 40   # mid-period Paid Social spend spike
SPEND_SPIKE_END_DAY = 55

CHANNELS = ["Paid Social", "Paid Search", "Influencer", "Organic", "Referral", "Direct"]

# True underlying quality of each channel (used to generate REAL conversion behavior,
# independent of what any attribution model will later claim)
TRUE_CHANNEL_LIFT = {
    "Paid Social": 0.9,   # inflated by last-click due to spend spike + sequencing position
    "Paid Search": 1.0,   # often the last click, gets outsized credit
    "Influencer": 1.6,    # genuinely strong at driving NEW awareness, rarely last click
    "Organic": 1.1,
    "Referral": 1.8,      # genuinely highest quality, but low volume
    "Direct": 0.5,        # mostly brand-aware users who were going to convert anyway
}

REGIONS = ["Region_A", "Region_B"]  # used for incrementality holdout test later

def daily_spend_multiplier(day):
    """Deliberate mid-period Paid Social spend spike."""
    if SPEND_SPIKE_START_DAY <= day <= SPEND_SPIKE_END_DAY:
        return 2.8
    return 1.0

def generate_users():
    users = []
    touchpoints = []
    user_id = 0

    for day in range(N_DAYS):
        date = START_DATE + timedelta(days=day)
        spike_mult = daily_spend_multiplier(day)

        # Number of new users arriving today (Paid Social spike pulls in more traffic,
        # but a good chunk is low-intent — this is the trap the attribution model must catch)
        base_new_users = np.random.poisson(180)
        spike_extra = int(base_new_users * 0.6 * (spike_mult - 1))
        new_users_today = base_new_users + spike_extra

        for _ in range(new_users_today):
            user_id += 1
            region = np.random.choice(REGIONS)

            # Build a realistic multi-touch journey (1-4 touches)
            n_touches = np.random.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.2, 0.1])

            # Awareness-stage channels are more likely to appear FIRST
            awareness_channels = ["Influencer", "Organic", "Paid Social"]
            closing_channels = ["Paid Search", "Direct", "Referral"]

            journey = []
            for t in range(n_touches):
                if t == 0:
                    # first touch: weighted toward awareness channels,
                    # with extra Paid Social weight during the spend spike
                    weights = np.array([0.30, 0.20, 0.35, 0.05, 0.05, 0.05])
                    if spike_mult > 1:
                        weights[0] += 0.15 * (spike_mult - 1)  # Paid Social
                        weights = weights / weights.sum()
                    ch = np.random.choice(CHANNELS, p=weights)
                elif t == n_touches - 1:
                    # last touch: weighted toward closing channels
                    weights = np.array([0.15, 0.35, 0.05, 0.10, 0.15, 0.20])
                    weights = weights / weights.sum()
                    ch = np.random.choice(CHANNELS, p=weights)
                else:
                    ch = np.random.choice(CHANNELS)
                journey.append(ch)
                touchpoints.append({
                    "user_id": user_id,
                    "date": date + timedelta(hours=int(t * 6)),
                    "channel": ch,
                    "touch_order": t + 1,
                    "is_last_touch": (t == n_touches - 1),
                    "region": region,
                })

            # TRUE conversion probability is driven by TRUE_CHANNEL_LIFT of the
            # channels actually in the journey (esp. the first touch = true driver),
            # NOT by whichever channel happens to get last-click credit.
            first_touch_channel = journey[0]
            true_lift = TRUE_CHANNEL_LIFT[first_touch_channel]
            base_conv_rate = 0.11
            conv_prob = min(0.85, base_conv_rate * true_lift)

            # Low-intent traffic from the Paid Social spike converts worse than average —
            # this is what makes last-click attribution look "successful" but be wrong
            if spike_mult > 1 and first_touch_channel == "Paid Social":
                conv_prob *= 0.55

            trial_started = np.random.random() < conv_prob

            paid_converted = False
            ltv = 0.0
            retained_months = 0
            if trial_started:
                # trial -> paid conversion
                paid_converted = np.random.random() < 0.42
                if paid_converted:
                    # monthly retention curve (churn each month)
                    monthly_price = 12.0
                    months = 0
                    retained = True
                    monthly_churn = 0.08
                    while retained and months < 24:
                        months += 1
                        retained = np.random.random() > monthly_churn
                    retained_months = months
                    ltv = monthly_price * months

            users.append({
                "user_id": user_id,
                "signup_date": date,
                "region": region,
                "n_touches": n_touches,
                "first_touch_channel": journey[0],
                "last_touch_channel": journey[-1],
                "trial_started": trial_started,
                "paid_converted": paid_converted,
                "retained_months": retained_months,
                "ltv": round(ltv, 2),
            })

    return pd.DataFrame(users), pd.DataFrame(touchpoints)


def generate_spend():
    rows = []
    base_daily_spend = {
        "Paid Social": 4000,
        "Paid Search": 3500,
        "Influencer": 1500,
        "Organic": 0,
        "Referral": 200,   # referral program cost, not media spend
        "Direct": 0,
    }
    for day in range(N_DAYS):
        date = START_DATE + timedelta(days=day)
        mult = daily_spend_multiplier(day)
        for ch, base in base_daily_spend.items():
            spend = base * (mult if ch == "Paid Social" else 1.0)
            spend = spend * np.random.uniform(0.9, 1.1)
            rows.append({"date": date, "channel": ch, "spend": round(spend, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    users_df, touchpoints_df = generate_users()
    spend_df = generate_spend()

    users_df.to_csv("data/users.csv", index=False)
    touchpoints_df.to_csv("data/touchpoints.csv", index=False)
    spend_df.to_csv("data/spend.csv", index=False)

    print(f"Generated {len(users_df):,} users")
    print(f"Generated {len(touchpoints_df):,} touchpoints")
    print(f"Trial starts: {users_df['trial_started'].sum():,}")
    print(f"Paid conversions: {users_df['paid_converted'].sum():,}")
    print(f"Total spend: ${spend_df['spend'].sum():,.0f}")
