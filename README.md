# Signal vs. Noise — Multi-Touch Attribution Audit & Budget Reallocation Engine

Built for Suno's Marketing Data Scientist role. This project simulates a consumer
subscription business (modeled loosely on an AI creative product) and shows what
happens when you actually challenge platform-reported attribution numbers instead
of trusting them.

## The question this answers

> "Which channels actually drive growth — not just which ones get the last click?"

## What's inside

1. **`src/generate_data.py`** — generates a synthetic dataset of 19K+ users with
   realistic multi-touch journeys, a deliberate mid-period Paid Social spend spike,
   and channel-sequencing bias (awareness channels like Influencer rarely capture
   the last click; closing channels like Paid Search usually do).

2. **`src/attribution_models.py`** — compares three views of the same data:
   - **Last-click** (what most ad platforms report by default)
   - **Time-decay multi-touch** (a more realistic model, more credit to touches
     closer to conversion)
   - **True contribution** (ground truth, since we control the simulation)

   **Finding:** last-click under-credits Influencer (which starts the most
   journeys) and over-credits Paid Search (which usually just closes them out).

3. **`src/incrementality_test.py`** — a simulated geo-holdout experiment: one
   region keeps seeing Paid Social ads during the spend-spike window, one region
   has them suppressed. Statistically significant result (p < 0.001): true
   incremental lift is roughly **6x smaller** than what naive/last-click data
   would imply.

4. **`src/subscriber_economics.py`** — LTV, CAC, payback period, and retention
   curves by true acquisition channel. Paid Social and Paid Search show 150+
   month payback periods at current CAC; Influencer and Referral pay back in
   under 26 months.

5. **`src/app.py`** — a Streamlit dashboard tying all of the above together,
   ending in a concrete budget reallocation recommendation.

## Run it

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/attribution_models.py
python src/incrementality_test.py
python src/subscriber_economics.py
streamlit run src/app.py
```

## Honest methodology notes

- **All data is synthetic.** It's constructed to mirror realistic attribution
  bias patterns I researched (spend-spike-driven inflation, first-touch vs.
  last-touch sequencing bias, awareness vs. closing channel dynamics) — it is
  not real Suno or any real company's data.
- **The incrementality test is simulated, not run on a live product** — the
  purpose is to demonstrate the *method* (geo-holdout design, statistical
  testing of lift, quantifying overstatement), which is directly transferable
  to a real experimentation program.
- **Time-decay weighting uses a simple exponential (2^n) scheme** — a
  production system would likely use a data-driven model (e.g., Shapley value
  or a trained attribution model), which is a natural next iteration.
- Built to demonstrate attribution/incrementality/subscriber-economics
  reasoning end-to-end, not to claim finished, production-grade infrastructure.

## Why this project, for this role

Suno's Marketing Data Scientist JD asks for someone who runs marketing
experiments, owns attribution modeling, challenges platform-reported numbers,
and understands subscriber economics. This project is a compressed, honest
demonstration of that exact workflow, built end-to-end by one person in a
short timeframe — not a claim of professional experience I don't have yet,
but evidence of how I think about the problem.
