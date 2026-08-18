# Project Overview — Sarcasm in Marketing-Relevant Reddit Comments

Final exam project for *Language Analytics and LLMs in Marketing* (TUM).
An R / RStudio project analysing **sarcasm in user-generated content** on Reddit,
restricted to marketing-relevant communities (tech, gaming, cars, fashion,
retail, entertainment).

---

## 1. What's in the folder

| File | Size | What it is |
|---|---|---|
| [06_Exam.Rproj](06_Exam.Rproj) | 205 B | RStudio project file (2-space indent, UTF-8, `RestoreWorkspace: Default`) |
| [Final Project.R](Final%20Project.R) | 73 lines | Scratch / starting script: downloads the dataset, applies the subreddit filter, sanity checks. Superseded by the three RQ scripts. |
| [rq1.R](rq1.R) | 428 lines | **RQ1** — which linguistic features distinguish sarcastic from non-sarcastic comments |
| [rq2_embeddings.R](rq2_embeddings.R) | 539 lines | **RQ2** — do embeddings + psycholinguistic features beat embeddings alone |
| [rq3_score_analysis.R](rq3_score_analysis.R) | 443 lines | **RQ3** — do sarcastic comments get higher Reddit scores |
| `outputs.docx` | 3.3 MB | Results document: three headings (RQ1, RQ3, RQ2) with **36 embedded PNG plots** pasted from RStudio. No prose — figures only. |
| `.RData` | 5.0 MB | Saved RStudio workspace: `sarcasm` (51,337 × 10), `quality_check` (1 × 3), `url` |
| `.Rhistory` | 5.3 KB | Console history — mirrors the first ~140 lines of `rq3_score_analysis.R` |
| `.Rproj.user/` | — | RStudio internal state (not part of the analysis) |

There is no README, no `renv`/lockfile, no saved figure files, and no cached
intermediate `.rds` files (see [§6 Gaps](#6-gaps-and-things-to-check)).

---

## 2. The data

**Source:** `https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv`
(the well-known SARC / Kaggle "Sarcasm on Reddit" balanced export, self-labelled
via the `/s` convention).

Downloaded fresh with `read_csv(url)` at the top of **every** script — there is no
local copy of the raw CSV.

**Columns (10):** `label`, `comment`, `author`, `subreddit`, `score`, `ups`,
`downs`, `date`, `created_utc`, `parent_comment`

**After the marketing-subreddit filter** (verified from the saved `.RData`):

| | |
|---|---|
| Rows | **51,337** |
| Subreddits | **62** |
| Label balance | 24,948 non-sarcastic (0) / 26,389 sarcastic (1) — roughly balanced |
| Date range | 2009-09 → 2016-12 |

### The subreddit filter
The same hard-coded 64-name `filter(subreddit %in% c(...))` block is repeated
verbatim in all four scripts. Six thematic groups:

- **Tech / mobile** — apple, iphone, Android, GooglePixel, AndroidMasterRace, windowsphone, Surface, GalaxyNote7, galaxynote4, lgv20, pebble
- **Hardware / PC** — nvidia, intel, Amd, razer, hardware, techsupport, buildapc, buildapcsales, pcmasterrace, headphones
- **Gaming** — Steam, playstation, PS4, PS4Pro, xboxone, NintendoSwitch, NintendoNX, wiiu, GameDeals
- **Cars** — askcarsales, cars, Autos, BMW, Volkswagen, SubaruForester, subaru, Miata, FocusST, Datsun, Justrolledintotheshop
- **Fashion / beauty** — streetwear, StreetwearSales, sneakermarket, Sneakers, FashionReps, supremeclothing, bapeheads, goodyearwelt, frugalmalefashion, BeautyBoxes, MakeupAddiction
- **Retail / entertainment** — walmart, starbucks, tacobell, TalesFromRetail, Random_Acts_Of_Amazon, netflix, boxoffice, moviecritic, television, movies, music

The stated rationale (in `rq2_embeddings.R`) is that this is a *substantively
motivated* subset rather than a random sample: it makes transformer embeddings
computationally feasible on a laptop while keeping every remaining comment
relevant to a marketing/UGC context.

> 62 of the 64 listed names actually appear in the data — 2 have no rows.

---

## 3. The three research questions

### RQ1 — What distinguishes sarcastic comments? → [rq1.R](rq1.R)

Five theory-driven features (explicitly used as **LIWC substitutes**, since LIWC
is paid and not R-compatible):

| # | Feature | How it's computed |
|---|---|---|
| 1 | Punctuation | regex counts of `!`, `?`, `...`, `"` |
| 2 | Sentiment | VADER `compound`/`pos`/`neu`/`neg` via `reticulate` → Python `vaderSentiment` |
| 3 | Capitalization | ALL-CAPS words (2+ letters) count and ratio |
| 4 | Intra-comment incongruity | `2 * pmin(VADER_pos, VADER_neg)` — positive *and* negative valence in the same comment (Riloff et al. 2013; Joshi et al. 2015) |
| 5 | Parent-reply incongruity | `abs(VADER_compound(comment) − VADER_compound(parent))` |

**Pipeline:** clean (drop `NA`, empty, and `[deleted]` in either comment or
parent) → engineer features → group means → Welch t-test + Wilcoxon per feature
→ **logistic regression** (`caret`, `glm`, 10-fold CV, 80/20 stratified split,
`set.seed(123)`) → confusion matrix, ROC/AUC, `varImp`.

VADER scores are cached to `vader_scores_rq1.rds` on first run
(`if (file.exists(...))` guard), because it's one Python call per comment *and*
per parent over ~50k rows.

**5 figures:** sentiment distribution of sarcastic comments (bar + histogram,
patchwork) · feature boxplots faceted by label · feature-importance bar ·
confusion-matrix heatmap · ROC curve.

---

### RQ2 — Do embeddings + psycholinguistic features beat embeddings alone? → [rq2_embeddings.R](rq2_embeddings.R)

A controlled two-model comparison:

- **Model 1** — semantic embeddings only
- **Model 2** — semantic embeddings + psycholinguistic features

Both use the **identical** classifier, CV scheme, and train/test rows, so any
gap is attributable to the features, not the modelling.

**Embeddings:** `text::textEmbed()` with model `microsoft/harrier-oss-v1-270m`,
mean-pooled tokens→text, layer `-2`, `keep_token_embeddings = FALSE`. Computed for
both `comment` and `parent_comment`; saved to `embeddings_comment.rds` /
`embeddings_parent.rds`. Requires the `{text}` package's conda/torch backend
(`textrpp_initialize()`).

**Psycholinguistic features (10):** `punct_density`, `n_exclaim`, `n_question`,
`n_ellipsis`, `upper_word_rate`, `n_upper_words`, `intra_comment_incong`,
`parent_reply_incong`, `parent_reply_flip`, `semantic_incong`.

Note the different operationalisation vs RQ1 — this script uses **`sentimentr`**
(not VADER):
- *intra-comment incongruity* = **SD of sentence-level sentiment** within a comment (0 if single sentence)
- *parent-reply incongruity* = `abs(ave_sentiment(comment) − ave_sentiment(parent))`, plus a binary **sign-flip** indicator
- *semantic incongruity* = `1 − cosine_similarity(emb_comment, emb_parent)` — topical rather than sentiment mismatch

**Classifier:** elastic-net logistic regression (`glmnet` via `caret`), 10-fold CV,
`tuneLength = 5`, centered/scaled, `metric = "ROC"`, parallelised with
`doParallel`. Evaluated on Accuracy / Precision / Recall / F1 / AUC, and the AUC
gap is tested with **DeLong's test** (`pROC::roc.test`).

**4 figures:** metric comparison bars · overlaid ROC curves · side-by-side
confusion matrices · Model 2 top-20 feature importance (colour-coded
embedding-dim vs psycholinguistic — i.e. do the hand-crafted features actually
crack the top of the ranking).

---

### RQ3 — Do sarcastic comments score higher? → [rq3_score_analysis.R](rq3_score_analysis.R)

**Vote-data quality check first.** The script validates `ups`/`downs` before
trusting them, and drops both. Recomputed 18 Aug 2026 (51,337 rows, no `NA`s in
any vote column):

- Two regimes, split cleanly by date: **85.8%** real (`ups == score`, `downs == 0`, 2009-09 → 2016-09), **14.2%** `-1` sentinels (2016-10 → 2016-12)
- `score = ups − downs` holds in **100%** of real rows but **4.7%** of sentinel rows
- `ups < 0` in **20.0%** of rows, `downs < 0` in **14.2%** — a vote tally cannot be negative
- `max(downs) = 0` corpus-wide, yet **6.4%** of comments have a negative score
- `ups == score` for **86%** of rows, correlation **0.897**

→ **`downs` carries no signal and `ups` duplicates `score`; both are dropped**
(`select(-ups, -downs)` at the end of § 3, so nothing downstream can reuse them).
`score` is the single measure of public approval. Note the missingness is a time
slice, not noise — restricting to real-vote rows would drop the last three months
of the corpus.

**Tests:**
- One-sided Welch t-test, `score ~ label`, `alternative = "less"` (H1: sarcastic > non-sarcastic)
- Wilcoxon rank-sum (robustness against the heavy score skew)
- **Cohen's d**, explicitly because significance is near-guaranteed at this n
- Same two tests on `comment_length` as a control check

**Regressions (control variables):**
1. `score ~ label + comment_length`
2. `score ~ label + comment_length + subreddit`
3. `log1p(score) ~ label + comment_length` (skew robustness)

**5 figures:** score ridgeline + mean score with Wilcoxon significance bracket
(patchwork) · top-15 subreddits by avg score vs by sarcasm rate, side by side ·
avg score by subreddit × sarcasm · monthly avg score over time with loess +
volume panel · comment length vs score with `lm` smoothers. (The former
ups-vs-score hexbin was cut: it presented convergent validity between two
columns that are one measure.)

---

## 4. Conventions shared across the scripts

- **Style:** every script opens with a header block stating the RQ and a numbered outline, and follows the same skeleton: *Setup → Load → Clean → Features → Descriptives → Tests → Model → Figures*.
- **Packages:** guarded installs — `if (!requireNamespace("x", quietly = TRUE)) install.packages("x")`.
- **Plotting:** shared `theme_report <- theme_minimal(base_size = 13) + bold title` and a colourblind-safe Okabe-Ito palette `pal <- c("Non-sarcastic" = "#0072B2", "Sarcastic" = "#D55E00")`, reused in all three scripts.
- **Reproducibility:** `set.seed(123)` before every split and every `train()`.
- **Label factor:** `label_f <- factor(label, levels = c(0,1), labels = c("Non-sarcastic","Sarcastic"))` — note RQ2 uses `"NonSarcastic"`/`"Sarcastic"` (no hyphen, for `make.names` compatibility with `caret`).

### Package inventory
`tidyverse` · `readr` · `stringr` · `scales` · `caret` · `glmnet` · `pROC` ·
`patchwork` · `reticulate` (→ Python `vaderSentiment`) · `text` (→ conda/torch/
transformers) · `sentimentr` · `doParallel` · `skimr` · `lubridate` · `ggridges` ·
`ggpubr` · `hexbin`

---

## 5. Suggested run order

1. `Final Project.R` — optional; just the download + filter sanity check
2. `rq3_score_analysis.R` — cheapest, pure R, no Python
3. `rq1.R` — needs `reticulate` + `vaderSentiment`; slow first run, then cached
4. `rq2_embeddings.R` — heaviest; needs the `{text}` conda backend and computes transformer embeddings for ~51k comments *and* ~51k parents

---

## 6. Gaps and things to check

These are observations from reading the code, not necessarily errors — flagging
so you can decide.

**Naming / organisation**

1. **The RQ numbers in the filenames don't match the headers.** `rq2_embeddings.R` has no RQ number in its header, and `rq3_score_analysis.R`'s header says **"RQ2"**. `outputs.docx` orders the sections RQ1 → RQ3 → RQ2. Worth settling on one numbering before submission — this overview uses the *filename* numbering.
2. **The 64-subreddit list is duplicated four times.** If it ever changes, it has to change in four places. A shared `subreddits.R` sourced by each script would fix it.
3. ~~**No figures are saved to disk.**~~ **Resolved 18 Aug 2026.** Every script writes 300-dpi PNGs to `figures/` via `ggsave()` and reads its raw data from a local cache instead of re-downloading. RQ3 now runs end to end in ~8 seconds.

**Correctness worth verifying**

4. **`rq2_embeddings.R:190`** — `str_count(comment, "\\.\\.\\.|???")`. The `???` alternative is a quantifier with nothing to quantify; in ICU regex this is invalid or silently meaningless. Compare `rq1.R:127`, which uses the proper `"\\.{3,}|\\?{3,}"`.
5. **`rq1.R:127`** — `ellipsis_count` counts `\\?{3,}` as an "ellipsis", but those same `?` characters are already counted in `quest_count`, so `punct_total` double-counts them.
6. **`rq1.R:278-281`** — the model matrix includes both `punct_total` *and* all four of its components. `punct_total` is an exact linear combination of them, so `glm` will drop one coefficient as `NA` (perfect collinearity). Harmless for prediction, but it makes the coefficient table and `varImp` output confusing.
7. **`microsoft/harrier-oss-v1-270m`** — worth confirming this model ID actually resolves on HuggingFace before the final run; the commented-out `test_embed` block at `rq2_embeddings.R:56-60` suggests this was still being debugged.
8. **`rq3_score_analysis.R:245`** — a comment says "given n > 1,000,000". After the subreddit filter, n is **51,337**. The effect-size argument still holds, just at a different magnitude.
9. **`View()` calls** in `Final Project.R:71` and `rq3_score_analysis.R:112` will fail under `Rscript`; they only work interactively in RStudio.
10. **Leftover debug code** in `rq2_embeddings.R:306-315` — `str()`, `packageVersion()`, `?textEmbed`, and a re-run of `test_embed` sit in the middle of the analysis flow. The `?textEmbed` call opens a help page mid-script.

**Environment**

11. **`.RData` (5 MB) is committed alongside `RestoreWorkspace: Default`**, so RStudio silently reloads a stale `sarcasm` object at startup. That can mask a broken script — it appears to work because the object was already in memory.
12. **The dataset is re-downloaded on every run** of every script. Caching it once with `readr::write_csv()` / `saveRDS()` would make runs faster and reproducible offline.
13. **No cached `.rds` files exist yet** (`vader_scores_rq1.rds`, `embeddings_comment.rds`, `embeddings_parent.rds`) — meaning the expensive steps haven't been run to completion in this folder, or the outputs were deleted.
