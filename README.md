# Market Efficiency Analysis — Time-Series ML Pipeline

A walk-forward machine learning pipeline built to investigate the **efficiency of a real-world predictive market** across multiple seasons. The project tests whether systematic predictive edges persist over time, or whether the market self-corrects as information becomes more widely available.

## Research Question

Can a machine learning committee identify and exploit pricing inefficiencies in a predictive market — and if so, does that edge survive as the market matures?

## Key Finding

The pipeline demonstrates a **monotonic decline in closing-line value (CLV) across four consecutive seasons** — moving from a positive edge in early seasons to a negative edge in the most recent one. The conclusion: it is **rising market efficiency**, not model degradation, that erodes predictive advantage over time. This is a clean empirical result about how information efficiency evolves in a real market.

## Methodology

- **Walk-forward validation** across four seasons to eliminate look-ahead bias and simulate realistic out-of-sample deployment
- **Strict open/close data separation** to prevent target leakage
- **Feature engineering**: pi-ratings (Constantinou & Fenton, 2013), Shin's method for probability de-vigging
- **Model committee**: LDA, Ridge Classifier, Bagged Logistic Regression, Random Forest, XGBoost
- **Ensemble strategies**: equal-vote and Brier-score-inverse weighted committees
- **Benchmark**: closing lines from the most efficient available market reference

## Tech Stack

- **Python** — pandas, scikit-learn, XGBoost, NumPy
- **Validation** — custom walk-forward framework
- **Evaluation** — Brier score, closing-line value (CLV), calibration analysis

## Results Summary

| Season | Predictive Edge (CLV) |
|--------|----------------------|
| Early  | Positive             |
| ...    | Declining            |
| Recent | Negative (~ market-efficient) |

The progressive erosion of edge across seasons is the central, reproducible result of this work.

## What This Project Demonstrates

- End-to-end ML pipeline design with rigorous validation methodology
- Awareness of common pitfalls in time-series ML (leakage, look-ahead bias, overfitting)
- Ensemble modeling and probability calibration
- Translating a noisy real-world dataset into a clear, defensible empirical conclusion

---

*Built as an independent research project exploring information efficiency in predictive markets.*
