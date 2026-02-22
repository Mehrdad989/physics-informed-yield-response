# Quickstart (Synthetic End-to-End)

This repo includes a fully reproducible synthetic pipeline demonstrating:

- physics-informed saturating yield response curve
- per (crop, region) curve fitting
- ML parameter prediction from region features + crop ID
- Bayesian cross-crop hierarchical model with ML-centered priors
- improved performance in data-sparse regions

## Setup

```powershell
git clone <YOUR_REPO_URL>
cd physics-informed-yield-response
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
pip install pymc arviz scikit-learn scipy pandas