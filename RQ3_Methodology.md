---
output:
  html_document: default
  pdf_document: default
---
# RQ3 — Do sarcastic comments receive higher scores (public approval) than non-sarcastic ones?

**Design:** confirmatory / inferential. The question is *directional and comparative*, so we test it with **three methods that fail in different ways** and check whether their direction agrees (triangulation). Because n ≈ 41k makes almost anything "significant," the evidence is judged on **effect size and robustness to controls**, not p-values alone.

**Data:** the cleaned SARC corpus already carrying the RQ1 features (comment length, etc.) plus the raw SARC columns `score`, `created_utc`, `author`, `subreddit`, `parent_comment`, `label`. After RQ3-specific cleaning: **41,077 comments** (19,278 non-sarcastic / 21,799 sarcastic), across the subreddits that survived filtering. "Public approval" is operationalized as Reddit `score` (net upvotes).

---

## Methodological logic (why this order)

| Step | Question it answers | Why we needed it |
|------|--------------------|------------------|
| 1–3. Clean score / time / controls | *Is each variable usable and leakage-free?* | Removes parse errors, implausible timestamps, and the `[deleted]`-author trap |
| 4. Unpaired test | *Is there a raw score difference at all?* | First, uncontrolled evidence |
| 5. Paired test | *Does it hold within the same thread?* | Structurally cancels topic/subreddit/thread — the cleanest comparison |
| 6. Bootstrap CI | *How big is the paired difference?* | An interpretable magnitude, not just a p-value |
| 7. Mixed regression | *Does it survive statistical controls?* | Adjusts for length, subreddit, time, author simultaneously |
| 8–10. Visualize + triangulate | *Do all three methods agree?* | The actual verdict, plus model diagnostics |

---

## Step 1–3 — Data cleaning (score, time, controls)

| Variable | Checks / actions | Why |
|----------|------------------|-----|
| **score** | coerce numeric; drop NA; keep integer values only; **no outlier trimming** | High scores are genuine virality (part of "approval"), not error; the planned rank-based tests + signed-log transform are already robust to skew |
| **created_utc** | coerce numeric; drop NA; keep 2005–2018 epoch range; standardize (`time_scaled`) | Raw Unix seconds (~1.4e9) destabilize `lmer`; standardizing changes only numerical stability, not significance |
| **comment_len** | re-check (already built in RQ1) | Enters the regression as a covariate |
| **subreddit** | check missing/blank; check casing collisions before pooling | Casing splits (e.g. `PS4`/`ps4`) would fragment a fixed effect |
| **author** | **give every `[deleted]` row a unique pseudo-ID** | Otherwise the `(1 \| author)` random effect pools thousands of different people into one fictitious "super-author", distorting every coefficient |

## Step 4 — Descriptive comparison + unpaired test

Median score is **1 (non-sarcastic)** vs **2 (sarcastic)**; distributions overlap heavily. We use the **Wilcoxon rank-sum test** (not a t-test): `score` is heavily right-skewed with negative values, violating normality. Effect size is the **rank-biserial correlation**, because at this n the p-value is near-guaranteed "significant" regardless of practical magnitude.

> **Direction convention (important).** `wilcox.test(score ~ label)` reports the Hodges–Lehmann shift as *non-sarcastic − sarcastic* (R uses the first factor level, `0`, as *x*), so its sign is the **reverse** of the "sarcastic vs. non-sarcastic" framing. On top of that, Reddit scores are heavily **tied integers**, so the HL estimate degenerates toward ~0 and its sign is numerically unreliable. We therefore read the unpaired **direction from the rank-biserial effect size** (positive = sarcastic higher), which is well-defined under ties, and cross-check it against the group **medians**. The paired estimate and the regression coefficient are already on the sarcastic-vs-non convention and are read directly.

## Step 5 — Paired Wilcoxon signed-rank (primary evidence)

Restrict to parents with exactly one sarcastic + one non-sarcastic reply (**237 matched pairs**) and compare the two replies' scores with the **signed-rank test**. Both replies share the same parent, so subreddit/thread/topic cancel structurally — no modelling required. This mirrors the SARC benchmark design and is the most credible answer to the question.

## Step 6 — Bootstrap CI on the paired difference

Resample **by pair** (2,000 replicates) and take the median within-pair difference (sarcastic − non-sarcastic), giving an interpretable magnitude alongside Step 5's p-value.

## Step 7 — Full-sample mixed-effects regression

For maximum power, model the **whole sample** with statistical (rather than structural) controls:

`score_transformed ~ label + comment_len + subreddit_pooled + time_scaled + (1 | author)`

- **Outcome:** signed-log score, `sign(score)·log1p(|score|)`, which tames the skew while preserving direction (positive = approval, negative = disapproval).
- **Controls:** comment length, subreddit (top-30 individually, rest pooled to `"Other"`, consistent with RQ1/RQ2), posting time, and an author **random intercept**.
- **Step 7b — linearity check:** quadratic alternatives for `comment_len` and `time_scaled` are compared by likelihood-ratio test + AIC; keep linear unless the data clearly needs more (comment length shows a genuine nonlinear component; the `label` conclusion is unchanged either way).

## Step 8–10 — Visualize, triangulate, diagnose

Paired scatter, within-pair difference histogram, bootstrap distribution, coefficient forest plots, and a **triangulation table** (does direction agree across all three methods?). Regression diagnostics (residuals-vs-fitted, Q–Q, author/subreddit effects, predicted-vs-observed) confirm the signed-log transform behaves.

---

## Results

**Descriptive:** median score 1 (non-sarcastic) vs 2 (sarcastic); IQR 2 vs 3.

**Triangulation across methods:**

| Method | n | Direction | Effect size | p-value | Significant (.05) |
|--------|:--:|-----------|:-----------:|:-------:|:-----------------:|
| Unpaired Wilcoxon (full sample) | 41,077 | Sarcastic higher | 0.017 (rank-biserial) | 0.002 | Yes |
| Paired Wilcoxon (matched pairs) | 237 | Sarcastic higher | 0.102 (rank-biserial) | 0.115 | No |
| Mixed regression (controlled) | 41,077 | Sarcastic lower | — (signed-log) | 0.678 | No |

Supporting detail: paired pseudo-median difference ≈ 0.50 but **bootstrap 95% CI = (0, 0)** and median paired difference = 0; regression `label` coefficient = **−0.005** (SE 0.011, p = 0.68). Controls behave sensibly (`comment_len` and `time_scaled` both p < 0.001; author random-effect SD ≈ 0.25).

**Key findings**

1. **No robust "higher approval" effect.** The only "significant" result (unpaired) has a **negligible effect size (r = 0.017)** — significant purely because n ≈ 41k. This is exactly the large-N trap the effect-size-first design guards against.
2. **The within-thread test does not confirm it.** The paired test points the same (higher) direction but is **not significant** (p = 0.12) and is **underpowered** — only 237 matched pairs, with a median pairwise difference of 0.
3. **It vanishes — even reverses slightly — under controls.** Once subreddit, length, posting time, and author are adjusted for, the sarcasm coefficient is **essentially zero and non-significant** (−0.005, p = 0.68).

**One-line answer:** *No — there is no robust evidence that sarcastic comments receive higher scores. A negligible raw tendency toward higher scores is not confirmed within threads and disappears once subreddit, length, time, and author are controlled for.*

---

## Limitations

- **"Approval" ≠ score.** Reddit `score` reflects visibility, timing, and virality as much as genuine approval; it is a proxy, not a direct measure. (The sibling data-quality check also found `downs` uninformative and `ups` largely redundant with `score`.)
- **Paired test underpowered.** Only 237 parents had exactly one sarcastic + one non-sarcastic reply, so the cleanest comparison has limited power — a non-result here is a power issue, not disconfirmation.
- **Tied discrete scores.** Integer, heavily-tied scores make the Hodges–Lehmann location estimate degenerate; direction is therefore judged from effect size / medians, and magnitudes should be read from effect sizes rather than the location shift.
- **Large N inflates significance** — hence effect size and robustness-to-controls, not p-values, drive the verdict.
- **Non-independence** (shared threads, repeat authors) is addressed by the author random effect and the paired design, but not exhaustively.

---

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm (SARC).* LREC.
- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting Linear Mixed-Effects Models Using lme4.* JSS.
- Kuznetsova, A., Brockhoff, P. B., & Christensen, R. H. B. (2017). *lmerTest Package: Tests in Linear Mixed Effects Models.* JSS.
- Kerby, D. S. (2014). *The simple difference formula: an approach to teaching nonparametric correlation* (rank-biserial).
- Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap.*
