# Final Project — Results: Sarcasm in Marketing-Relevant Reddit Comments

**Summary**: The empirical results (36 plots in `outputs.docx`) of the `06_Exam` group final project. Three findings: (RQ1) hand-crafted linguistic features distinguish sarcasm *statistically* but predict it barely above chance; (RQ2) adding those features on top of transformer embeddings gives a small but significant lift; (RQ3) sarcastic comments score marginally higher, but the effect is negligible and mostly a subreddit artifact.

**Course**: Language Analytics and LLMs in Marketing (Clara Wiebel, SS26)

**Type**: assignment (final project — results)

**Sources**: `06_Exam/outputs.docx` (figures), `06_Exam/rq1.R`, `rq2_embeddings.R`, `rq3_score_analysis.R`, `docs/PROJECT_OVERVIEW.md`

**Related**: [[reading-sarc-sarcasm-corpus]] | [[session4-text-classification]] | [[session7-sentiment-emotion-analysis]] | [[session8-embeddings-transformers]] | [[word-embeddings]] | [[session2-preprocessing-bow-tfidf]] | [[user-generated-data]] | [[text-analytics-method-selection]]

**Last updated**: 2026-08-13

---

## Data & setup

Runs on the SARC balanced export (see [[reading-sarc-sarcasm-corpus]]), filtered to 64 marketing-relevant subreddits → **51,337 rows, 62 subreddits, ~balanced** (24,948 non-sarcastic / 26,387 sarcastic), spanning 2009-09 → 2016-12 (source: outputs.docx; docs/PROJECT_OVERVIEW.md). All models: `set.seed(123)`, 80/20 stratified split, 10-fold CV, Okabe-Ito colours (blue = non-sarcastic, orange = sarcastic).

> **Caveat on the docx**: it is figures only, ordered RQ1 → RQ3 → RQ2, and mixes RStudio *console screenshots* (tables, model summaries) with the actual ggplots. The numbers below are read directly off those images.

---

## RQ1 — What linguistic features distinguish sarcasm? → weakly predictive

Five theory-driven feature families as **LIWC substitutes**, sentiment via VADER (source: rq1.R). See [[session7-sentiment-emotion-analysis]].

**Group means** (non-sarc → sarc) (source: outputs.docx):

| Feature | Non-sarc | Sarc | Direction |
|---|---|---|---|
| Exclamation count | 0.054 | **0.133** | ↑ ~2.5× (strongest) |
| Punctuation total | 0.333 | 0.384 | ↑ |
| Question count | 0.127 | 0.117 | ↓ slightly |
| Caps ratio | 0.0277 | 0.0340 | ↑ |
| VADER compound | 0.0954 | 0.0859 | ↓ slightly |
| Intra-comment incongruity | 0.0344 | 0.0392 | ↑ |
| Parent-reply incongruity | 0.417 | 0.429 | ↑ |

All seven differences are **statistically significant** (Welch t + Wilcoxon), most extreme for exclamation count (t-test p ≈ 1.5×10⁻²⁰¹) (source: outputs.docx). But the gaps are tiny.

**Surface sentiment of sarcastic comments** (VADER): **41.2% neutral, 37.4% positive, 21.4% negative** — sarcasm skews *positive/neutral on the surface*, matching the SARC paper's finding that words like "obviously / clearly / totally" flag sarcasm (source: outputs.docx; cf. [[reading-sarc-sarcasm-corpus]]).

**Logistic regression** (9 predictors) (source: outputs.docx):
- CV accuracy **0.531**, Kappa 0.071.
- Test: accuracy **0.531** (95% CI 0.521–0.540) vs No-Information-Rate 0.514 — significant (p = 3.5×10⁻⁴) but a hair above baseline. **AUC = 0.562**. Sensitivity 0.371 / Specificity 0.699 (predicts "non-sarcastic" by default).
- `varImp`: **exclamation count = 100**, then punct_total 10.7, intra-incongruity 9.1, ellipsis 8.4, question 6.3, parent-incongruity 3.0, caps 1.4, **VADER compound 0.0**.
- One coefficient (`quote_count`) dropped as `NA` for **singularity** — the double-counting collinearity flagged in `PROJECT_OVERVIEW.md §6`.

> **RQ1 verdict**: sarcasm *is* linguistically distinguishable, but hand-crafted features alone barely beat a coin flip (AUC 0.56). **Exclamation marks are essentially the only strong signal**; VADER sentiment adds nothing on its own.

## RQ2 — Do embeddings + psycholinguistics beat embeddings alone? → yes, slightly

Two models, identical classifier (elastic-net logistic, `glmnet`) and rows, differing only in features (source: rq2_embeddings.R). Embeddings from `text::textEmbed()` (a ~270M transformer), mean-pooled, for both comment and parent. See [[session8-embeddings-transformers]], [[word-embeddings]].

| Metric | Model 1 (embeddings) | Model 2 (+ psycholinguistic) |
|---|---|---|
| Accuracy | 0.641 | **0.650** |
| Precision | 0.648 | 0.658 |
| Recall | 0.660 | 0.662 |
| F1 | 0.654 | 0.660 |
| **AUC** | **0.698** | **0.711** |

(source: outputs.docx)

- **DeLong's test**: Z = −7.26, **p = 3.9×10⁻¹³** → the +0.013 AUC gain is statistically significant (at n ≈ 10k test rows) (source: outputs.docx).
- **Model 2 top-20 importance**: `n_exclaim` ranks **#1 (100)**, `punct_density` #3, `n_question` #12 — the hand-crafted features **crack the top of the embedding-dimension ranking**, so they carry signal the embeddings don't fully capture (source: outputs.docx).

> **RQ2 verdict**: embeddings alone (AUC 0.698) hugely outperform RQ1's hand-crafted-only model (0.562); adding psycholinguistic features gives a **small but significant** further lift, driven again by **exclamation/punctuation**. This is the project's cleanest positive result. Connects to the SARC paper's caution that plain summed embeddings weren't automatically best — here a stronger contextual model plus lexical features wins.

## RQ3 — Do sarcastic comments score higher? → statistically yes, practically no

Vote-quality check first, and it removes two of the three vote columns (source: rq3_score_analysis.R § 3, recomputed 18 Aug 2026). Reddit stopped exposing separate up/down tallies around 2014, and this scrape straddles the change, so every row sits in one of two regimes: **85.8%** carry real data (`ups == score`, `downs == 0`, 2009-09 → 2016-09) and **14.2%** carry `-1` sentinels (2016-10 → 2016-12). Three checks confirm these are not vote counts: the identity `score = ups − downs` holds in 100% of real rows but only 4.7% of sentinel rows; `ups` is negative in 20.0% of rows and `downs` in 14.2%, which no tally of votes cast can be; and `max(downs) = 0` across the corpus although 6.4% of comments have a negative score, which by definition requires downvotes. `downs` therefore records no signal at all, and `ups` — an exact copy of `score` in the 86% of rows where it is real (r = 0.897) — adds nothing. **Both are dropped; `score` is the single measure of public approval.**

- **Means**: score 6.59 → **7.32**; median 1 → 2 (source: rq3_score_analysis.R). *Earlier drafts also quoted “ups 5.16 → 6.13” here as corroboration; that pair blends real values with `-1` sentinels, which is why it sits below the score means, and it has been removed.*
- **One-sided Welch t** (H1: sarc > non-sarc): t = −1.699, **p = 0.045** — just significant.
- **Wilcoxon**: p = 2.5×10⁻¹¹ — strongly significant on ranks.
- **Cohen's d = 0.015** — *negligible* (the authors flag that significance is near-guaranteed at this n, so effect size is what matters).
- **Comment length** (control): means 57.2 vs 52.5, Welch p < 2.2×10⁻¹⁶, but **Wilcoxon p = 0.23** and medians equal (45) → the length "difference" is a skew artifact, not a central shift.

**Regressions** (source: outputs.docx):
1. `score ~ label + length`: label coef **0.837, p = 0.048** (R² = 0.0005).
2. `score ~ label + length + subreddit`: label coef **0.652, p = 0.13 → not significant** once subreddit is controlled.
3. `log1p(score) ~ label + length`: label coef 0.083, **p < 2×10⁻¹⁶** (R² = 0.003).

**Heterogeneity across subreddits** (avg score by subreddit × sarcasm): the direction *flips* by community — in `movies`, `television`, `streetwear` non-sarcastic scores higher; in `pcmasterrace`, `cars`, `apple`, `Justrolledintotheshop` sarcastic scores higher. Highest sarcasm rates: `pcmasterrace`, `Justrolledintotheshop`, `apple`, `Android`, `cars` (source: outputs.docx). Over time, scores drift up and comment volume explodes 2014–2016.

> **RQ3 verdict**: sarcastic comments get a *statistically* higher score, but the effect is **negligible (d = 0.015)** and **largely disappears once subreddit is controlled** — so no meaningful engagement uplift; it's mostly which communities are sarcastic, not sarcasm itself. See [[user-generated-data]].

---

## Cross-RQ synthesis

- **Exclamation marks are the single most reliable hand-crafted sarcasm cue** — top feature in both RQ1 and RQ2.
- **Model ladder**: hand-crafted only ≈ 0.56 AUC → transformer embeddings ≈ 0.70 → embeddings + psycholinguistic ≈ 0.71. Context/representation matters far more than lexical counts, but lexical counts still add a sliver (a concrete instance of the method-selection trade-offs in [[text-analytics-method-selection]]).
- **Surface sentiment is a poor sarcasm signal** (VADER importance = 0 in RQ1) — consistent with sarcasm inverting literal sentiment; the useful signal is *incongruity* and punctuation, not polarity. See [[session7-sentiment-emotion-analysis]].
- **Marketing takeaway**: sarcasm doesn't buy engagement (RQ3), and detecting it in UGC needs contextual models, not sentiment lexicons.

### Caveats carried from the code (PROJECT_OVERVIEW §6)
- RQ1 `punct_total` double-counts `?` runs and is collinear with its components (a coefficient drops to `NA`).
- RQ1 vs RQ2 operationalise incongruity differently (VADER vs `sentimentr`; RQ2 adds cosine-based *semantic* incongruity) — not directly comparable.
- No figures/`.rds` are saved to disk; plots were pasted from interactive runs, so `outputs.docx` isn't reproducible from the repo alone.
- Filename vs header RQ-numbering is inconsistent (docx orders RQ1 → RQ3 → RQ2).

## Related pages

- [[reading-sarc-sarcasm-corpus]] — the dataset paper this project runs on
- [[session4-text-classification]] — logistic/elastic-net classification, precision/recall/F1/AUC
- [[session7-sentiment-emotion-analysis]] — VADER, `sentimentr`, why sentiment ≠ sarcasm
- [[session8-embeddings-transformers]] — the `text` package transformer embeddings used in RQ2
- [[word-embeddings]] — embedding representations
- [[user-generated-data]] — Reddit UGC, engagement metrics, subreddit heterogeneity
- [[text-analytics-method-selection]] — the hand-crafted-vs-embeddings trade-off
