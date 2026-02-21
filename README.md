# physics-informed-yield-response
# Physics-Informed Yield Response Modeling (Cross-Crop + Cross-Region)
*A unified response-curve framework for wheat and maize using public U.S. data, with physics-informed ML and Bayesian hierarchical uncertainty.*

## Why this project
Crop yields respond nonlinearly to water availability, temperature stress, nutrient supply, and local growing conditions. Pure ML models often predict yield directly, but may extrapolate poorly and rarely provide interpretable, uncertainty-aware outputs. Classical agronomic response models capture realistic behavior but are hard to scale across regions and crops.

This project builds a **transferable, uncertainty-aware** modeling framework that combines:
- **Agronomic response curves** (biologically plausible structure)
- **Physics-informed ML** (predict curve parameters from context while enforcing constraints)
- **Bayesian hierarchical inference** (partial pooling across geography and crops + uncertainty quantification)
- **Cross-region and cross-crop transfer tests** (to prove generalization)

This framework mirrors the methodology used in nitrogen-response and NUE modeling, but uses only **public, reproducible** datasets.

---

## Core modeling question
**How does crop yield respond to water availability, and how do temperature stress, fertilizer intensity proxies, and regional context modify that response across wheat and maize—especially in regions not seen during training?**

---

## Transferability goals (explicit)
This project treats transferability as a first-class objective.

### Cross-region transfer
We evaluate the model under **leave-one-region-out** settings, e.g.:
- hold out entire states, or
- hold out agro-climatic zones (recommended)

**Success criteria:** performance degrades gracefully on held-out regions and posterior uncertainty widens appropriately.

### Cross-crop transfer
We fit a unified framework across wheat and maize using:
- crop-level priors / embeddings
- shared response-curve family
- hierarchical pooling across crops and regions

---

## Data (public)
The project is designed around a county-year panel:

### Yield (target)
- USDA NASS county-level yield (annual)
  - crop: **wheat** and **maize**
  - unit: typically bushels/acre (converted to kg/ha or t/ha)

### Weather (drivers)
One of:
- PRISM (preferred for U.S. climate gridded data), or
- NOAA station data aggregated to county, or
- ERA5/Daymet (alternative)

Derived features include:
- seasonal precipitation (primary response axis)
- growing degree days (GDD)
- heat stress days
- drought proxies (dry spell length, precipitation anomalies)

### Fertilizer proxy (conditioning variable)
- State-level fertilizer use or sales statistics (USDA/USGS/other public sources)
Used as a **proxy** to condition curve parameters, not as the response axis.

> Note: The model can run without fertilizer proxy (weather-only baseline), then upgrade when added.

---

## Method overview (what we build)

### Stage 0 — Data assembly
Create an analysis-ready panel:
- county × year × crop yield
- joined weather aggregates
- region/zone labels for transfer tests
- fertilizer proxy (optional for v1)

**Output:** `data/processed/panel_county_year.csv`

---

### Stage 1 — Response curve identification (per unit)
For each (crop, county) or (crop, region) grouping, learn a yield–water response curve.

**Backbone curve (example):**
\[
Y(R) = Y_{min} + (Y_{max} - Y_{min}) (1 - e^{-kR})
\]

Where:
- \(R\) = seasonal precipitation / water index (primary axis)
- \(Y_{min}\) = baseline productivity
- \(Y_{max}\) = yield potential (given constraints)
- \(k\) = sensitivity / responsiveness to water

**Output:** fitted parameters per unit-year or unit, depending on design.

---

### Stage 2 — Physics-informed ML (parameter prediction)
Machine learning predicts curve parameters from context:
- climate normals
- temperature stress metrics
- soil proxies (optional)
- fertilizer proxy (optional)
- region labels
- crop identity

**Physics constraints (enforced by design):**
- \(Y_{max} > Y_{min}\)
- \(k > 0\)
- monotonic yield response to water

Implementation trick:
- predict `Ymin`
- predict `log(Ymax - Ymin)` then exponentiate
- predict `log(k)` then exponentiate

**Output:** models that map features → curve parameters.

---

### Stage 3 — Bayesian hierarchical inference (uncertainty + pooling)
Bayesian inference combines:
- ML parameter predictions as priors
- hierarchical pooling across geography and crops
- observed yields (likelihood)

Suggested hierarchy:
- Global
  - Crop (wheat/maize)
    - Agro-climatic zone (or state)
      - County
        - Year residual

**Outputs:**
- posterior distributions for curve parameters
- posterior predictive yield response curves
- calibrated uncertainty intervals

---

### Stage 4 — Scenario & sensitivity analysis
Compute interpretable quantities:
- marginal yield response (dY/dR)
- “flat” vs “responsive” regimes
- stress-conditioned sensitivity (e.g., higher heat reduces water responsiveness)
- model behavior in held-out regions (transfer performance + uncertainty widening)

---

## Evaluation plan (must-have)
We compare three evaluation regimes:

1. **Random split** (sanity baseline)
2. **Leave-one-region-out** (primary transfer test)
   - hold out state(s) or agro-climatic zone(s)
3. **Time-based split** (optional)
   - train early years, test later years

Metrics:
- RMSE / MAE for yield predictions
- calibration: empirical coverage of 50% / 90% credible intervals
- transfer robustness: performance and uncertainty width on held-out regions

---

## Repository structure
