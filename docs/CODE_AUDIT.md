# Code Audit — Sarcasm in Marketing-Relevant Reddit Comments

**Audited:** 14 Aug 2026 · **Paper due:** 28 Aug 2026
**Scope:** `rq1.R` · `rq2_embeddings.R` · `rq3_score_analysis.R` · `docs/`

Four fixes are needed before submission — two of them change numbers we report. Most of the
cleanup that looked necessary turned out not to be: measured against the real data, it affects
zero rows. Two questions need a team decision.

| | Count |
|---|---|
| Urgent — fix before submission | **4** |
| Worth doing — cheap, protects us | **3** |
| Verified non-issues — don't spend time here | **4** |
| Open decisions for the team | **2** |

---

## Urgent — before submission

`U1` and `U4` are the two that touch numbers already in the write-up. `U2` blocks the methods
section; `U3` means the script doesn't execute at all. IDs are stable reference keys — use them
in discussion.

### U1 — RQ1's regression is rank-deficient by construction

`rq1.R:129` → `rq1.R:279-281`

`punct_total` is defined as the *exact sum* of its four components, and then all five go into
the model together:

```r
punct_total = excl_count + quest_count + ellipsis_count + quote_count

model_data <- sarcasm_clean %>%
  select(label_f, punct_total, excl_count, quest_count,   # ← punct_total is redundant
         ellipsis_count, quote_count, ...)
```

The design matrix is singular, so `glm` drops a coefficient — exactly the `quote_count` → `NA`
singularity already noted in our results doc. Milder second version of the same problem:
`quest_count` counts every `?` while `ellipsis_count` counts runs of `?{3,}`, so `???` is
counted twice across two predictors that are both in the model.

| Quantity | Effect of the fix |
|---|---|
| Accuracy 0.531 · AUC 0.562 · Kappa 0.071 | **Unchanged** — dropping a redundant column doesn't alter fitted values |
| Coefficient table | Changes — no more `NA`, remaining estimates shift |
| `varImp` ranking | **Changes** — this is what our RQ1 interpretation rests on |

**Fix:** remove `punct_total` from the `select()`, keeping its four components. Re-run rq1,
refresh the RQ1 importance figure and the varImp line in the results doc.

> **Why now:** our RQ1 story — "exclamation count = 100, VADER = 0" — is currently read off a
> degenerate model.

### U2 — Model 1 isn't what the code says it is

`rq2_embeddings.R:344` · `rq2_embeddings.R:468-469` vs `outputs.docx`

Every RQ2 figure labels the models **"Model 1: Merged Embeddings"** and **"Model 2: Merged
Embeddings + Psycholinguistic."** The code on disk builds Model 1 from the comment embeddings
alone, and labels them differently:

```r
model1_data <- bind_cols(emb_comment, label = ...)        # comment only

pal <- c("Model 1: Embeddings only" = ...,                # ≠ figure labels
         "Model 2: Embeddings + Psycholinguistic" = ...)
```

"Merged" most naturally means comment *and* parent embeddings combined. If so, re-running the
committed code produces a materially different Model 1 than the one in our results.

> **Why now:** our methods section describes Model 1. We need to know which version produced the
> numbers before we write that paragraph.

### U3 — `rq2_embeddings.R` cannot run as committed

`rq2_embeddings.R:190` · `rq2_embeddings.R:306-315`

Line 190 throws a hard regex error, verified against our installed R:

```r
n_ellipsis = str_count(comment, "\\.\\.\\.|???")
  # Error: Syntax error in regex pattern (U_REGEX_RULE_SYNTAX)

n_ellipsis = str_count(comment, "\\.{3,}|\\?{3,}")   # rq1's working version
```

Lines 186-194 are a single `mutate()`, so the whole statement fails atomically — six columns
never get created. The failure cascades to the feature assembly at line 337, where
`select(punct_density, n_exclaim, …)` can't find them, so **Model 2 can't be built at all**.

Separately, a scratch block sits mid-pipeline at lines 306-315 — `packageVersion()`,
`?textEmbed` (opens a help pane), and a stray `test_embed <- textEmbed(…)` that re-runs an
embedding call for nothing. Even with line 190 fixed, the file won't source end to end until
this goes.

> **Good news:** this is a regression introduced *after* our results were generated — the docx
> shows Model 2 built cleanly with `n_exclaim` ranked #1. Our RQ2 numbers aren't in doubt; the
> file just no longer reproduces them.

### U4 — Drop `ups` and `downs`; one reported number is wrong

`rq3_score_analysis.R:133-135, 191-192, 292, 311, 328-338` ·
`docs/final-project-sarcasm-results.md:76` · `docs/PROJECT_OVERVIEW.md:143-144, 157-158`

**The one-sentence version:** `downs` never records a single downvote (it's only ever `-1` or
`0`), and `ups` is either missing or an exact copy of `score` — so neither adds any information
that `score` doesn't already carry.

**Why the columns are broken.** Reddit stopped exposing separate up/down tallies around 2014
(vote fuzzing, to frustrate bots), so the API returns only the net score plus filler values.
This dataset's scrape spans that change and the two chunks were concatenated. Every row falls
into one of two regimes, with nothing in between:

| Regime | Share | `ups` | `downs` |
|---|---|---|---|
| Placeholder | ~80% | `-1` | `-1` |
| Real data | ~20% | exactly `score` | `0` |

**Three checks confirm these are not vote counts:**

1. **The identity `score = ups − downs` fails.** It holds in 100% of real-data rows but only
   4.6% of placeholder rows — and those only because `−1 − (−1) = 0` happens to match a score
   of 0.
2. **Counts are negative.** `downs = −1` in 80% of rows, and some rows carry `ups < 0`. A tally
   of votes cast cannot be negative.
3. **No downvote is ever recorded.** `max(downs) = 0` across the sample, yet ~5% of comments
   have a negative score — which by definition requires downvotes.

**The split is perfectly confounded with time.** It is purely chronological, with a hard cutoff
between September and October 2016. The file isn't sorted by date, which is why it looks
interleaved — adjacent rows jump between months. Every real-data row is 2016-09 or earlier;
every `-1` row is 2016-10 or later. Score magnitude and sarcasm label explain nothing (19-20%
flat across every bucket, both labels). Date is the entire explanation.

This makes the case for dropping *stronger*, not weaker: the missingness isn't noise, it's a
time slice. Any model or figure using `ups` silently becomes an analysis of September 2016
alone, while `score` covers the full range.

**What's affected — no statistical result is contaminated.** `rq1.R`, `rq2_embeddings.R`, and
`Final Project.R` never touch `ups`/`downs`. Within RQ3, every inferential step — Welch t-test,
Wilcoxon, Cohen's d, and all three regressions (`rq3_score_analysis.R:218-270`) — runs on
`score` alone. **The headline finding that sarcastic comments score higher stands as-is.
Nothing needs re-running.**

What *is* affected is descriptive and presentational, in severity order:

| # | Where | Problem |
|---|---|---|
| 1 | `docs/final-project-sarcasm-results.md:76` | Reports "ups 5.16 → 6.13" alongside the score means, reading as independent corroboration. Those averages are `-1` sentinels blended with real values — which is exactly why 5.16 sits *below* the score mean of 6.59. **The one place a bad number has reached the write-up.** |
| 2 | `rq3_score_analysis.R:292, 311` | Figure 1's "Ups" facet. The bars compare sarcastic vs non-sarcastic "ups," but the difference is really how many placeholder rows landed in each group. Misleading in a submitted figure. |
| 3 | `rq3_score_analysis.R:328-338` | Figure 2 in full. The hexbin with its equality line and Pearson *r* presents convergent validity between two measures that are one measure — and, given the time confound, a September-2016 slice masquerading as a full-sample validation. |
| 4 | `rq3_score_analysis.R:133-135, 191-192`; `docs/PROJECT_OVERVIEW.md:143-144, 157-158` | The quality check's framing and the `mean_ups`/`mean_downs` descriptive rows. Source of the "86% / r = 0.897 / kept as robustness check" conclusion — right arithmetic, wrong interpretation. |

**Fix:** one number to delete, one figure panel to drop, one figure to cut, and some wording to
correct. Not a re-analysis. Keep the quality check itself and report it in the methods section —
it is what *justifies* using `score` as the single measure of public approval.

> ⚠️ **Verify before this goes in methods.** Two numbers in our own materials disagree: the
> regime split above says ~20% of rows carry real vote data, but `PROJECT_OVERVIEW.md:144`
> reports `ups == score` for **86%** of rows. Those can't both be right unless most placeholder
> rows also have `score = -1`, which the median score of 1 makes implausible. The likely
> explanation is how `NA` is handled — `mean(ups == score, na.rm = TRUE)` drops missing rows —
> but someone should recompute both figures before either is quoted. **The decision to drop is
> robust either way**; only the supporting percentage is in question.

---

## Worth doing

Cheap, and each closes a gap a marker could reasonably poke at. None change our findings.

### W1 — Make the figures reproducible

We already wrote this caveat ourselves: *"plots were pasted from interactive runs, so
outputs.docx isn't reproducible from the repo alone."* Two small changes retire it — cache the
Hugging Face download to an `.rds` instead of re-pulling a 1M-row CSV on every run of every
script, and `ggsave()` the figures to a `figures/` folder instead of copying them out of the
RStudio pane.

Bonus: uniform figure dimensions in the paper, and re-runs drop from minutes to seconds while
we're still iterating.

### W2 — Two claims we can state more strongly, for free

**RQ2's lift is a punctuation story, not a general "psycholinguistic features help" story.**
Of the ten psycholinguistic features, only three appear in Model 2's top 20 — all punctuation:

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | `n_exclaim` | **100** | Psycholinguistic |
| 2 | `Emb_Dim559` | 49 | Embedding dim. |
| 3 | `punct_density` | **42** | Psycholinguistic |
| 4-11 | embedding dimensions | 39 → 26 | Embedding dim. |
| 12 | `n_question` | **25** | Psycholinguistic |
| 13-20 | embedding dimensions | 24 → 21 | Embedding dim. |

Absent entirely: `semantic_incong`, `intra_comment_incong`, `parent_reply_incong`,
`parent_reply_flip`, `n_ellipsis`, `upper_word_rate`, `n_upper_words`.

This also **rules out a confound**: `semantic_incong` is derived from the parent comment's
embedding, not from psycholinguistics. If it had ranked high, part of our RQ2 lift would have
been parent-context information rather than hand-crafted features. It doesn't rank.

**Our two "intra-comment incongruity" measures point in opposite directions.** Worth reporting
as a finding about measurement choice, not hiding:

| Measure | Non-sarc | Sarc | Direction |
|---|---|---|---|
| RQ1 — VADER, `2·min(pos,neg)` | 0.0344 | 0.0392 | sarcastic **higher** |
| RQ2 — `sentimentr`, SD of sentence sentiment | 0.00834 | 0.00710 | sarcastic **lower** |

Not just different scales — contradictory signs. RQ2's version is structurally `0` for any
single-sentence comment, so on Reddit it largely proxies comment length rather than incongruity.
Our results doc currently reports only the RQ1 direction.

*Sanity check:* capitalization validates cleanly across both scripts — RQ1's `caps_ratio` and
RQ2's `upper_word_rate` both give 0.0277 → 0.0340, identical to four digits. Same feature.

### W3 — The headline N is off by two

The results doc reports **51,337** as the analysis N. That's the count after the subreddit
filter but *before* cleaning. The modelled N is **51,335** — confirmed two independent ways:
recomputed from the raw data, and read off the console screenshot in `outputs.docx`
(`NonSarcastic 24948 / Sarcastic 26387`, the label split we already quote).

---

## Verified non-issues — don't spend time here

Each looked like a problem and turned out not to be. Listed with evidence so nobody re-opens them.

### N1 — The three scripts clean data differently, and it affects zero rows

rq1 drops `[deleted]`, rq2 doesn't, rq3 doesn't require a parent comment, and the empty-string
tests differ. This was initially flagged as a correctness bug. **That was wrong.** Measured
against the actual filtered corpus:

| Cleaning variant | Rows |
|---|---|
| rq1 as written | 51,335 |
| rq2 as written | 51,335 |
| rq3 as written | 51,335 |
| Proposed unified cleaning | 51,335 |

`comment == "[deleted]"`: **0 rows**. `[removed]`: **0 rows**. Whitespace-only: **0 rows**.
The SARC balanced export was already stripped of placeholders upstream.

Unifying the cleaning is tidy but buys nothing measurable. It's latent insurance for if we
change the subreddit list or dataset version — not a pre-deadline task.

### N2 — RQ3's one-sided t-test direction is correct

The reported `t = −1.699` with H1 "sarcastic > non-sarcastic" looks like a sign error, but
isn't. `t.test(score ~ label, alternative = "less")` with factor levels `0, 1` tests
*non-sarcastic < sarcastic* — exactly our stated hypothesis. The negative statistic is
consistent with that. Leave it alone.

### N3 — Don't unify VADER vs `sentimentr`

RQ1 computes sentiment and incongruity with VADER; RQ2 uses `sentimentr`. Unifying would make
our "RQ2 adds the RQ1 features" claim literally true, but means re-running the heaviest pipeline
in the project two weeks before the deadline. Already documented as a caveat — better handled as
a limitations paragraph, strengthened with the contradiction evidence in **W2**.

### N4 — The general refactor can wait

Real but cosmetic: the 63-name subreddit vector is copy-pasted four times, the
`pal`/`theme_report` pair three times, and `Final Project.R` is superseded scratch holding a
fourth copy of the list. Consolidating into `R/00_corpus.R`, `R/theme.R`, and
`R/features_surface.R` would remove ~250 duplicated lines and one real drift risk — but changes
no results.

*If we do it later:* `pal` isn't the same object everywhere — rq1/rq3 key it by label, rq2 by
model name. A naive shared `pal` would collide.

---

## Two decisions for the team

Both block writing, not coding. Neither has an obvious default.

### 1. Is Model 1 comment-only, or merged comment + parent embeddings?

The figures say "Merged," the code says comment-only. Whoever ran the RQ2 pipeline should
confirm which produced the numbers in `outputs.docx`. Then we make the code and the methods
section agree.

**Recommendation** — confirm from whoever ran it rather than guessing. If nobody remembers, one
clean re-run settles it, and we should budget for that now rather than in the final week.

### 2. How do we describe the embedding dimensions in the paper?

`Emb_Dim559` and friends are individual coordinates of the mean-pooled transformer vector. They
are **not interpretable** — no dimension corresponds to "sarcasm" or "sentiment," the basis is
arbitrary, and the ranking among dimensions wouldn't survive a different random seed. We should
not name specific dimensions as findings.

The trap: our importance plot shows `n_exclaim` (100) beating every embedding dimension (best:
49), which reads as "punctuation beats embeddings." That's backwards. Embeddings alone reach
**AUC 0.698** versus **0.562** for hand-crafted features alone — they carry far more signal,
just distributed across hundreds of correlated coordinates that split the credit. `varImp`
measures marginal contribution, not total information.

**Suggested wording:**

> Punctuation features rank above any individual embedding dimension in Model 2, with
> exclamation count the strongest single predictor. This does not imply punctuation carries more
> information than the embeddings — the embeddings alone reach AUC 0.698 versus 0.562 for
> hand-crafted features alone — but rather that their contribution is distributed across many
> correlated dimensions, while punctuation contributes a concentrated, non-redundant signal.

---

## Corrections to earlier claims

Two things asserted during the audit before measuring, so nobody acts on the wrong version:

- **"rq2 is training on rows whose comment text is literally `[deleted]`."** False — there are
  zero such rows. See **N1**.
- **"N differs across the three RQs."** False — all three land on exactly 51,335 rows.

Both errors pointed the same way: they made the cleaning inconsistency look like a correctness
problem when it's only a maintainability one. The urgent list survived measurement; the cleaning
work did not.

---

*Line references are to the working tree as of 14 Aug 2026. Row counts recomputed from the
Hugging Face source; feature importances and model metrics read from the figures in
`outputs.docx`. The `ups`/`downs` regime analysis carries over from earlier investigation — see
the flagged discrepancy in **U4** before quoting its percentages.*
