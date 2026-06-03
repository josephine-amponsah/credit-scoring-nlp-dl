# Credit Scorecard API

**LSTM · SHAP · Temperature Calibration · FastAPI**

A production-ready credit scoring system that combines a deep learning classifier with post-hoc explainability and calibrated probability outputs. Given a customer's financial profile, the API returns a risk score, a risk band, and a ranked breakdown of exactly which features drove that score — and by how many points.

---

## Why explainability matters in credit models

Credit decisions sit at the intersection of financial welfare and legal obligation. In most jurisdictions a lender cannot simply decline an application — they must be able to state *why*. In the EU this is codified in GDPR's right to explanation (Article 22); in the US the Equal Credit Opportunity Act requires adverse action notices that name the principal reasons for any unfavourable decision.

Beyond regulation, there is a fairness argument. A model that cannot explain its outputs cannot be audited for discriminatory patterns. A model that can be explained can be challenged, corrected, and trusted. This matters especially for deep learning models, which are capable of fitting complex non-linear patterns that no feature importance table can summarise at training time — explanations must be computed *per prediction*, not once at model build.

Explainability in a credit context means three things in practice:

- **Attribution** — for this specific customer, which features moved the score up or down, and by how much?
- **Direction** — is a feature helping or hurting the score, and is that consistent with domain knowledge?
- **Magnitude** — contributions should be expressed in units a decision-maker understands, not in abstract SHAP units.

This system addresses all three.

---

## Model: LSTM classifier on tabular credit data

### Why LSTM for tabular data?

The majority of credit scoring uses gradient-boosted trees (XGBoost, LightGBM), which are well-suited to tabular data and natively handle mixed feature types. The LSTM used here treats each customer's full feature vector as a single-step sequence passed through a recurrent layer before a feed-forward classification head. In practice this gives the model a richer internal representation than a plain MLP, since the LSTM hidden state can encode interactions across the whole feature vector in a single pass before the dense layers act on it.

The architecture:

```
Input features
    │
    ├─ Categorical columns → Embedding layers (one per column)
    └─ Numerical columns   → StandardScaler
    │
    └─→ Concatenated vector → LSTM (hidden 128) → FC head [64 → 32 → 3]
                                                     BatchNorm + Dropout at each layer
```

Training used:
- **Class-weighted cross-entropy loss** to handle the imbalanced class distribution (Poor / Standard / Good)
- **AdamW** optimiser with gradient clipping (`max_norm=1.0`)
- **ReduceLROnPlateau** scheduler with early stopping on validation AUC

### Probability calibration

Raw softmax outputs from neural networks are not well-calibrated probabilities — a model that outputs 0.9 for "Good" should be right 90% of the time, but in practice is often overconfident. **Temperature scaling** corrects this by learning a single scalar `T` on the validation set:

```
calibrated_probs = softmax(logits / T)
```

`T` is optimised by minimising negative log-likelihood on the held-out validation set using L-BFGS. A `T > 1` softens the distribution (reduces overconfidence); a `T < 1` sharpens it. This is the lightest-weight calibration method that works reliably for multi-class neural networks, and it preserves the model's ranking order while correcting the probability magnitudes.

---

## Explainability: SHAP on an LSTM

### The challenge

Standard SHAP DeepExplainer does not support PyTorch models with `BatchNorm1d` inside a sequential head — the normalisation layer's population statistics break the backpropagation path that DeepLIFT requires. `GradientExplainer` is used instead. It computes expected gradients — averaging the gradient of the output with respect to the input over a set of reference (background) samples — which is mathematically equivalent to Integrated Gradients and works on any differentiable PyTorch model.

### Input wrapping

The LSTM takes two separate tensors (categorical indices, scaled numerics). SHAP expects a single input. An `LSTMWrapper` module concatenates both into one float tensor before the forward pass and casts the categorical slice back to `long` for the embedding lookup internally:

```python
class LSTMWrapper(nn.Module):
    def forward(self, x):
        return self.model(x[:, :n_cat].long(), x[:, n_cat:].float())
```

The explainer is initialised with 200 background samples drawn from the training set:

```python
explainer = shap.GradientExplainer(wrapped_model, background)
shap_values = explainer.shap_values(x)   # → (n_samples, n_features, n_classes)
```

### Loan type feature aggregation

The `Type_of_Loan` column originally contained a comma-separated list of loan types per customer (e.g. `"Auto Loan, Mortgage, Personal Loan"`). TF-IDF vectorisation expanded this into one binary column per loan type (prefixed `loan_type__`). The LSTM trains on all of these individual columns, which preserves the information. For display, however, showing eight separate `loan_type__` rows in a contribution table is noise — a user needs to know whether their loan mix helped or hurt, not the individual TF-IDF weight of each token. The SHAP values of all `loan_type__` columns are therefore summed back into a single `Loan_Type` contribution at serving time.

---

## Risk score design

### Formula

The score is computed from the calibrated class probabilities using an **expected-class formula**:

```
E[class] = 0 × P(Poor) + 1 × P(Standard) + 2 × P(Good)
score    = 300 + 275 × E[class]
```

This gives three clean anchor points:

| Outcome | E\[class\] | Score |
|---------|-----------|-------|
| Certain Poor | 0.0 | 300 |
| Certain Standard | 1.0 | 575 |
| Certain Good | 2.0 | 850 |

The 275 multiplier is `(850 − 300) / 2` — the full score range divided by the number of class steps. It can be tuned without changing the formula's structure.

The formula was chosen over simpler alternatives (e.g. `300 + 550 × P(Good)`) because it uses all three class probabilities. A customer with 80% P(Poor) and 20% P(Standard) scores differently from one with 80% P(Poor) and 20% P(Good), as it should.

### Band thresholds

The band boundaries are not arbitrary — each one corresponds to a specific expected-class value:

| Band | Score range | Interpretation |
|------|-------------|----------------|
| Poor | 300 – 437 | E\[class\] < 0.5 · Poor is the plurality outcome |
| Fair | 438 – 574 | 0.5 ≤ E\[class\] < 1.0 · Mixed Poor/Standard |
| Good | 575 – 712 | 1.0 ≤ E\[class\] < 1.5 · Standard or better is likely |
| Excellent | 713 – 850 | E\[class\] ≥ 1.5 · Good is the dominant outcome |

### SHAP contributions in score points

Raw SHAP values are in probability space. To make contributions interpretable in the same units as the score, each feature's SHAP value is projected into score-point space:

```
score_points[f] = shap[f, Poor] × 0 + shap[f, Standard] × 275 + shap[f, Good] × 550
```

The weights `[0, 275, 550]` are the partial derivatives of the score formula with respect to each class probability. A feature with `score_points = +30` added 30 points to this customer's score; one with `score_points = −45` subtracted 45 points. The top 7 features by absolute contribution are returned in each API response.

---

## Running the API

### Prerequisites

```
Python 3.10+
```

Install dependencies:

```bash
pip install fastapi uvicorn torch shap scikit-learn pandas numpy joblib
```

### 1. Export model artefacts from the notebook

Run the export cell at the end of `deep_learning.ipynb`. This saves the following to `artefacts/`:

```
artefacts/
  lstm_clf.pt           # LSTM model weights
  temp_scaler.pt        # Temperature scaling parameter
  scaler.pkl            # StandardScaler for numerical columns
  le_dict.pkl           # Per-column LabelEncoders for categorical columns
  shap_background.npy   # 200-sample background for GradientExplainer
  meta.json             # Column lists, cat_dims, tfidf_prefix
```

Make sure `meta.json` includes the TF-IDF prefix:

```python
meta["tfidf_prefix"] = "loan_type__"
```

### 2. Add test users

Populate `test_user_ids.json` with customer records in this format:

```json
[
  {
    "user_id": "USR_7241",
    "features": {
      "Age": 34,
      "Occupation": "Engineer",
      "Annual_Income": 72000,
      ...
    }
  }
]
```

### 3. Start the server

```bash
uvicorn api.main:app --reload --port 8000
```

The API will log missing/unexpected state dict keys on startup — unexpected keys from BatchNorm running statistics are expected and safe to ignore.

### 4. Open the scorecard UI

Navigate to [http://localhost:8000](http://localhost:8000) in your browser. Select a test user from the dropdown and click **Run Score**.

The UI returns:

- **Risk score** (300 – 850) with a visual progress bar
- **Risk band** (Poor / Fair / Good / Excellent)
- **Calibrated class probabilities** (Poor / Standard / Good)
- **Top 7 positive drivers** — features that increased the score, in score points
- **Top 7 negative drivers** — features that decreased the score, in score points

### 5. Call the API directly

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_7241",
    "features": {
      "Age": 34,
      "Occupation": "Engineer",
      "Annual_Income": 72000
    }
  }'
```

Example response:

```json
{
  "user_id": "USR_7241",
  "risk_score": 641,
  "risk_band": "Good",
  "probabilities": {
    "Poor": 0.0821,
    "Standard": 0.4914,
    "Good": 0.4265
  },
  "top_positive_factors": [
    { "feature": "Credit_History_Age", "score_points": 38.5 },
    { "feature": "Loan_Type",          "score_points": 21.2 },
    { "feature": "Annual_Income",       "score_points": 14.7 }
  ],
  "top_negative_factors": [
    { "feature": "Outstanding_Debt",           "score_points": -44.1 },
    { "feature": "month3_delay_from_due_date", "score_points": -29.3 },
    { "feature": "Num_of_Delayed_Payment",     "score_points": -11.8 }
  ]
}
```

### Available endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Scorecard UI |
| `POST` | `/score` | Score a single user |
| `GET` | `/users` | List all test users |
| `GET` | `/health` | API health check |

---

## Project structure

```
.
├── api/
│   ├── main.py            # FastAPI app — routes, scoring logic, SHAP aggregation
│   └── index.html         # Scorecard UI
├── artefacts/             # Exported model files (generated by notebook)
├── deep_learning.ipynb    # Model training, SHAP computation, score analysis
├── test_user_ids.json     # Test customer records
└── README.md
```