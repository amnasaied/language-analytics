---
output:
  html_document: default
  pdf_document: default
---
# RQ3 — Do sarcastic comments receive higher scores (public approval) than non-sarcastic ones?

**Design:** confirmatory / inferential. The question is *directional and comparative*, so we test it with **two complementary methods** — a full-sample rank test and a controlled mixed regression — and check whether they agree. Because n ≈ 51k makes almost anything "significant," the evidence is judged on **effect size and robustness to controls**, not p-values alone.

**Data:** the cleaned SARC corpus already carrying the RQ1 features (comment length, etc.) plus the raw SARC columns `score`, `created_utc`, `author`, `subreddit`, `parent_comment`, `label`. After RQ3-specific cleaning: **51,299 comments** (24,945 non-sarcastic / 26,354 sarcastic), across 62 subreddits. "Public approval" is operationalized as Reddit `score` (net upvotes).

---

## Methodological logic (why this order)

| Step | Question it answers | Why we needed it |
|------|--------------------|------------------|
| 1–3. Clean score / time / controls | *Is each variable usable and leakage-free?* | Removes parse errors, implausible timestamps, and the `[deleted]`-author trap |
| 4. Full-sample test | *Is there a raw score difference at all?* | Uncontrolled rank-based evidence + effect size |
| 5. Mixed regression | *Does it survive statistical controls?* | Adjusts for length, subreddit, time, author simultaneously |
| 6–8. Visualize + diagnose | *What's the verdict?* | Convergence of the methods, plus model diagnostics |

---

## Step 1–3 — Data cleaning (score, time, controls)

| Variable | Checks / actions | Why |
|----------|------------------|-----|
| **score** | coerce numeric; drop NA; keep integer values only; **no outlier trimming** | High scores are genuine virality (part of "approval"), not error; the planned rank-based tests + signed-log transform are already robust to skew |
| **created_utc** | coerce numeric; drop NA; keep 2005–2018 epoch range; standardize (`time_scaled`) | Raw Unix seconds (~1.4e9) destabilize `lmer`; standardizing changes only numerical stability, not significance |
| **comment_len** | re-check (already built in RQ1) | Enters the regression as a covariate |
| **subreddit** | check missing/blank; check casing collisions before pooling | Casing splits (e.g. `PS4`/`ps4`) would fragment a fixed effect |
| **author** | **give every `[deleted]` row a unique pseudo-ID** | Otherwise the `(1 \| author)` random effect pools thousands of different people into one fictitious "super-author", distorting every coefficient |

## Step 4 — Descriptive comparison + full-sample rank test

Median score is **1 (non-sarcastic)** vs **2 (sarcastic)**; distributions overlap heavily. We use the **Wilcoxon rank-sum test** (not a t-test): `score` is heavily right-skewed with negative values, violating normality. Effect size is the **rank-biserial correlation**, because at this n the p-value is near-guaranteed "significant" regardless of practical magnitude.

> **Direction convention (important).** `wilcox.test(score ~ label)` reports the Hodges–Lehmann shift as *non-sarcastic − sarcastic* (R uses the first factor level, `0`, as *x*), so its sign is the **reverse** of the "sarcastic vs. non-sarcastic" framing. On top of that, Reddit scores are heavily **tied integers**, so the HL estimate degenerates toward ~0 and its sign is numerically unreliable. We therefore read the **direction from the rank-biserial effect size** (positive = sarcastic higher), which is well-defined under ties, and cross-check it against the group **medians**. The regression coefficient is already on the sarcastic-vs-non convention and is read directly.

## Step 5 — Full-sample mixed-effects regression

For maximum power, model the **whole sample** with statistical (rather than structural) controls:

`score_transformed ~ label + comment_len + subreddit_pooled + time_scaled + (1 | author)`

- **Outcome:** signed-log score, `sign(score)·log1p(|score|)`, which tames the skew while preserving direction (positive = approval, negative = disapproval).
- **Controls:** comment length, subreddit (top-30 individually, rest pooled to `"Other"`, consistent with RQ1/RQ2), posting time, and an author **random intercept**.
- **Step 5b — linearity check:** quadratic alternatives for `comment_len` and `time_scaled` are compared by likelihood-ratio test + AIC; keep linear unless the data clearly needs more (comment length shows a genuine nonlinear component; the `label` conclusion is unchanged either way).

## Step 6–8 — Visualize, compare, diagnose

Coefficient forest plots and a **comparison table** (does the direction agree across the two methods?). Regression diagnostics (residuals-vs-fitted, Q–Q, author/subreddit effects, predicted-vs-observed) confirm the signed-log transform behaves.

---

## Results

**Descriptive:** median score 1 (non-sarcastic) vs 2 (sarcastic); IQR 2 vs 4.

**Comparison across methods:**

| Method | n | Direction | Effect size | p-value | Significant (.05) |
|--------|:--:|-----------|:-----------:|:-------:|:-----------------:|
| Full-sample Wilcoxon | 51,299 | Sarcastic higher | 0.033 (rank-biserial) | 2.8e-11 | Yes |
| Mixed regression (controlled) | 51,299 | Sarcastic higher | — (signed-log) | 0.115 | No |

Supporting detail: regression `label` coefficient = **+0.016** (SE 0.010, p = 0.12). Controls behave sensibly (`comment_len` and `time_scaled` both p < 0.001; author random-effect SD ≈ 0.24).

**Key findings**

1. **Both methods lean the same way, but the effect is negligible.** Both point toward sarcastic scoring slightly higher, yet the only **statistically significant** result (full-sample) has a **negligible effect size (r = 0.033)** — significant purely because n ≈ 51k. This is exactly the large-N trap the effect-size-first design guards against.
2. **It does not survive controls.** Once subreddit, length, posting time, and author are adjusted for, the sarcasm coefficient is **small, positive, and non-significant** (+0.016, p = 0.12) — a directional hint, not a reliable effect.

**One-line answer:** *No — there is no statistically reliable evidence that sarcastic comments receive meaningfully higher scores. Both methods lean slightly toward "higher", but the only significant result has a negligible effect size and the controlled estimate is positive but not significant.*

---

## Limitations

- **"Approval" ≠ score.** Reddit `score` reflects visibility, timing, and virality as much as genuine approval; it is a proxy, not a direct measure. (The sibling data-quality check also found `downs` uninformative and `ups` largely redundant with `score`.)
- **Tied discrete scores.** Integer, heavily-tied scores make the Hodges–Lehmann location estimate degenerate; direction is therefore judged from effect size / medians, and magnitudes should be read from effect sizes rather than the location shift.
- **Large N inflates significance** — hence effect size and robustness-to-controls, not p-values, drive the verdict.
- **Non-independence** (shared threads, repeat authors) is addressed by the author random effect, but not exhaustively.

---

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm (SARC).* LREC.
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting Linear Mixed-Effects Models Using lme4.* JSS.
- Kuznetsova, A., Brockhoff, P. B., & Christensen, R. H. B. (2017). *lmerTest Package: Tests in Linear Mixed Effects Models.* JSS.
- Kerby, D. S. (2014). *The simple difference formula: an approach to teaching nonparametric correlation* (rank-biserial).
