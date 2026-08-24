---
output:
  html_document: default
  pdf_document: default
---
# RQ2 — Can integrating semantic embeddings with psycholinguistic features improve sarcasm detection?

**Design:** predictive / classification. Where RQ1 *explained* which features differ (inference), RQ2 asks whether those features **improve detection** on top of modern semantic embeddings — evaluated out-of-sample with held-out test performance, cross-validated hyper-parameters, and significance testing (Sessions 4 & 8).

**The comparison (an ablation):** three models, same classifier family where possible, so any difference is attributable to the *inputs*:

| Model | Inputs | Estimator |
|-------|--------|-----------|
| **Features only** (RQ1 baseline) | 11 psycholinguistic features + subreddit | logistic regression |
| **Model 1** | 640-dim contextual **embeddings** only | ridge logistic |
| **Model 2** | **embeddings + 11 features** | ridge logistic |

If Model 2 > Model 1 → features add value on top of embeddings (the RQ2 question). If Model 1 ≫ features-only → embeddings carry most of the signal.

---

## Methodological logic (why this order)

| Step | Question it answers | Why we needed it |
|------|--------------------|------------------|
| 1. Embeddings | *How do we represent meaning?* | Contextual semantics beyond surface features |
| 2. Leakage-safe split | *How do we evaluate honestly?* | Prevents the model seeing test information |
| 3. Feature matrices | *What goes into each model?* | Builds the three ablation inputs |
| 4. Fit models | *Can each represent sarcasm?* | Trains Model 1, Model 2, baseline |
| 5. Evaluate | *Which performs best?* | Out-of-sample F1/AUC on shared test set |
| 6. Robustness | *Does the result hold?* | Across model class **and** data splits |
| 7. Significance | *Is the gap real?* | McNemar on the same test comments |

---

## Step 1 — Generate embeddings

We embed each comment with its parent context using **`microsoft/harrier-oss-v1-270m`** (a 270M-parameter contextual model, 640 dimensions). Input is formatted `"Context: <parent> \n Reply: <comment>"` — a natural-language separator, because this decoder-style model has no trained `[SEP]` token. Embeddings are **computed once and cached** to disk (expensive; deterministic).

## Step 2 — Leakage-safe train/test split

*Why:* a naive row-level split could put the two replies of a matched pair on opposite sides, letting the model "see" thread context it shouldn't. So we split **by `parent_comment`** (80/20) — an entire thread goes wholly to train or test. This mirrors the SARC benchmark design and guarantees no thread leakage. Class balance is checked to hold across the split.

## Step 3 — Prepare feature matrices (no leakage)

The 11 hand-crafted features are standardized using **training-set means/SDs only** (applying test statistics would leak test information into the scaling). Residual `NA`s (a few `sentiment_diff` values from parents that could not be VADER-scored) are **mean-imputed** (0 after scaling), keeping every row. This yields `X_emb` (embeddings) and `X_combined` (embeddings + features).

## Step 4 — Fit the models

- **Model 1 / Model 2:** **ridge logistic regression** (`cv.glmnet`, α = 0) — the appropriate estimator for 640-dim inputs, matching Session 8's `textTrain`. The penalty **λ is tuned by internal cross-validation** on the training set.
- **Features-only baseline:** the RQ1 logistic model refit **on the same split**, with subreddit pooled (top-30 individually, rest as `"Other"`) so it predicts on the **full test set** — the same comments as Model 1/2, making the three-way comparison fair.

## Step 5 — Evaluation

On the held-out test set: **accuracy, precision, recall, F1, and AUC**. AUC (threshold-free) is the primary metric; F1/precision/recall are reported at a 0.5 threshold (appropriate given the balanced classes).

## Step 6 — Robustness (two independent axes)

- **Model-class robustness — XGBoost.** Refit Model 1 and Model 2 as non-linear gradient-boosted trees to confirm the conclusion isn't an artifact of a *linear* model. Early stopping watches a **validation fold carved from train** (never the test set), so the reported metrics are leak-free.
- **Sampling robustness — repeated splits.** Repeat the full split → fit → evaluate over **10 seeds**, reporting F1/AUC **mean ± SD** per model and the **win-rate** (how often Model 2 beats Model 1). These test different things: XGBoost varies the *model* (same split); repeated splits vary the *split* (same model) — the latter is what rules out one-split luck (Session 4: "single splits are noisy — report mean *and* variance").

## Step 7 — Significance

**McNemar's test** — the correct paired test for two classifiers on the **same** test comments — checks whether one model gets significantly more of those specific comments right than another.

---

## Results

**Single-split performance (ranked by F1):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|:--------:|:---------:|:------:|:--:|:---:|
| **Model 2: embeddings + features** | 0.676 | 0.672 | 0.702 | **0.687** | **0.739** |
| XGBoost: embeddings + features | 0.671 | 0.666 | 0.702 | 0.684 | 0.732 |
| Model 1: embeddings only | 0.670 | 0.664 | 0.704 | 0.683 | 0.725 |
| XGBoost: embeddings only | 0.638 | 0.636 | 0.669 | 0.652 | 0.701 |
| Features only (RQ1 baseline) | 0.585 | 0.583 | 0.636 | 0.608 | 0.618 |

**Significance (McNemar):** Model 1 vs Model 2 **p = 0.023**; both embedding models vs features-only **p < 0.001** (p ≈ 1e-40 and 7e-50).

**Cross-split robustness (10 splits):** Model 2 F1 = 0.685 ± 0.003, Model 1 F1 = 0.682 ± 0.003; **Model 2 wins 9/10 splits**, mean ΔF1 = **+0.003 ± 0.003**, ΔAUC = **+0.011**.

**Key findings**

1. **Adding features improves detection — small but robust.** Model 2 beats Model 1 by a modest margin, but the gain is **statistically significant** (McNemar p = 0.023) and **consistent across nearly every split** (9/10 win-rate, low variance) — not one-split noise.
2. **Embeddings carry the bulk of the signal.** Features-only reaches only AUC 0.62 (near chance), embeddings-only 0.72 — so contextual semantics dominate; psycholinguistic features add a small increment on top.
3. **The pattern is model-agnostic.** Both ridge and XGBoost show embeddings + features > embeddings only.
4. **Consistent with RQ1 and SARC:** hand-crafted features are individually weak but real, and semantics/context dominate sarcasm detection.

**One-line answer:** *Integrating psycholinguistic features with semantic embeddings yields a small but consistent and statistically significant improvement in sarcasm detection; embeddings provide most of the predictive power.*

---

## Limitations

- **Small absolute gain:** features add ≈ +0.003 F1 / +0.011 AUC — reliable but modest; embeddings do the heavy lifting.
- **Frozen embeddings + light classifier** (by design, to isolate the feature contribution) — not end-to-end fine-tuning, so absolute performance is not state-of-the-art.
- **Modest ceiling overall** (best AUC ≈ 0.74) — expected; sarcasm is hard (SARC: bag-of-n-grams ≈ 76%, humans ≈ 82%).
- **Baseline estimator differs** (features-only uses plain logistic vs ridge for the embedding models) — a minor confound in that one comparison.
- **Truncation:** embeddings cap input at 256 tokens, so the longest comments are truncated.

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm (SARC).* LREC.
- Kjell, O., Giorgi, S., & Schwartz, H. A. (2023). *The `text` package: analysing language with transformers in R.*
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Dietterich, T. G. (1998). *Approximate statistical tests for comparing classifiers* (McNemar's test).
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). *Regularization paths for GLMs via coordinate descent* (`glmnet`).
