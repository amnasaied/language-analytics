# Plan vs. Code — Alignment Check

Comparison of `Research Questions + Notes` (the group's planning document, with the
literature-backed Q1 feature table) against what `rq1.R`, `rq2_embeddings.R` and
`rq3_score_analysis.R` actually implement.

Companion to [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md),
[CLEANING_CROSSCHECK.md](CLEANING_CROSSCHECK.md) and
[COURSE_COVERAGE_CROSSCHECK.md](COURSE_COVERAGE_CROSSCHECK.md).

---

## Verdict in one line

**The research design is fully implemented; the literature table is not.** All three
RQs match the plan's structure, and RQ3 exceeds it. But of the six cited features in
the Q1 table, three are missing entirely, and none of the plan's proposed R tools were
used.

---

## 1. The three RQs — structural alignment

| Plan | Code | |
|---|---|---|
| RQ1 — features distinguishing sarcastic comments; LIWC substitutes because LIWC is paid & not R-compatible | `rq1.R`, header states the same rationale verbatim | ✔ |
| RQ1 follow-up — "are sarcastic comments more likely positive or negative?" | Figure 1 plots `sentiment_cat` distribution | ⚠ see §4 |
| RQ2 — Model 1 embeddings only vs Model 2 embeddings + psycholinguistic, **identical classifier** | Both elastic-net, same folds, same rows, same split | ✔ |
| RQ2 — evaluate accuracy, precision, recall, F1 | All four reported, **plus AUC and a DeLong test** | ✔ exceeds plan |
| RQ2 — psycholinguistic features "**same as in research question 1**" | **Not the same** — see §3 | ✗ |
| RQ3 — control for other factors, e.g. comment length | Three regressions: `+ comment_length`, `+ subreddit`, `log1p(score)` | ✔ exceeds plan |
| RQ3 — "test whether the difference is significant" | One-sided Welch t-test, Wilcoxon, **plus Cohen's d** | ✔ exceeds plan |
| Visualise sarcastic comments' sentiment distribution | `rq1.R` Figure 1 | ✔ |
| Visualise confusion matrix | `rq1.R` Figure 4; `rq2` side-by-side | ✔ |

The plan's skeleton is faithfully built. Everything below is about the *content* of
the features.

---

## 2. The five "proposed features" — the group's agreed minimum

Listed at the bottom of page 1 of the plan.

| # | Proposed | `rq1.R` | `rq2_embeddings.R` |
|---|---|---|---|
| 1 | Punctuation (regex) | ✔ `!`, `?`, `...`, `"` | ⚠ no quote count; **`???` regex is broken** |
| 2 | Sentiment — **"please use VADER"** | ✔ VADER via `reticulate` | ✗ **`sentimentr`, not VADER** |
| 3 | Capitalization (uppercase word count) | ✔ `caps_count` + `caps_ratio` | ✔ `n_upper_words` + rate |
| 4 | Intra-comment incongruity | ✔ `2 * pmin(VADER_pos, VADER_neg)` | ⚠ different: SD of sentence sentiment |
| 5 | Parent-reply incongruity (abs. sentiment difference) | ✔ | ✔ + sign-flip + `semantic_incong` |

**All five exist in RQ1.** This is the part of the plan that landed cleanly.

---

## 3. The one design-level deviation: RQ2 does not use RQ1's features

The plan is explicit: Model 2 adds "psycholinguistic features (**same as in research
question 1**)". The code silently changes two of them:

| | RQ1 | RQ2 |
|---|---|---|
| Sentiment engine | VADER | **sentimentr** |
| Intra-comment incongruity | co-presence of pos & neg valence | **SD of sentence-level sentiment** |

Why it matters: RQ2's internal comparison is still valid — Models 1 and 2 share rows,
folds and classifier, so the AUC gap is attributable to the added features. But the
project can no longer claim **"RQ2 tests whether RQ1's features add signal"**, which is
what the plan set out to do. If RQ2 shows no improvement, you cannot tell whether
RQ1's features are weak or whether the substituted operationalisations are.

Two ways out, either acceptable:

- **(a)** Re-run RQ2 with the VADER features from `vader_scores_rq1.rds` (already
  cached; cheap) so the two scripts test the same constructs.
- **(b)** Keep the switch and justify it — sentimentr gives sentence-level scores,
  which the SD-based measure requires, and VADER does not. Then state plainly that
  RQ2 tests *the feature family*, not RQ1's exact features.

(a) is truer to the plan. (b) needs one sentence and no code.

---

## 4. The Q1 literature table — feature by feature

This is where the gap is. The plan built a properly sourced table; the code implements
half of it.

### ✔ Implemented

**Hutto & Gilbert (2014) — emphasis-aware sentiment** · tool: `vader`
VADER is used exactly as planned, and capitalization and punctuation are counted.
Two caveats:
- **"intensifiers" and contrastive "but" are not counted.** Both are named in the plan;
  a `but`/intensifier count is one line of regex.
- **Conceptual double-counting.** The plan treats caps + punctuation + sentiment as
  *one* feature ("emphasis-aware sentiment") because VADER already folds caps and `!!!`
  into its compound score. The code treats them as three independent predictors. That,
  plus `punct_total` being an exact sum of its own four components in the same `glm`
  ([rq1.R:279](../rq1.R#L279)), makes the `varImp` ranking hard to read.

**González-Ibáñez et al. (2011) — punctuation, quotation marks, emoticons, laughter**
· tool: `quanteda` + `textfeatures`
Punctuation ✔ and quotation marks ✔ (RQ1 only). But **emoticons and laughter tokens
(`lol`, `haha`) are never counted** — and the paper is cited precisely because these
were its *most reliable* manual cues. `CLEANING_CROSSCHECK.md` verified the corpus is
100% ASCII, so unicode emoji genuinely don't apply — but ASCII emoticons and laughter
tokens do, and are trivial to count:

```r
n_emoticon = str_count(comment, "[:;=8][-o*']?[)\\](\\[dDpP/:}{@|\\\\]"),
n_laughter = str_count(tolower(comment), "\\b(l+o+l+|h[ae]h[ae]+h*|rofl|lmao)\\b")
```

### ⚠ Concept implemented, operationalisation changed

**Riloff et al. (2013) — polarity incongruity** · tool: `tidytext` + Bing/AFINN
The plan specifies a **polarity-flip count** — sequential transitions between positive
and negative spans. The code uses `2 * pmin(VADER_pos, VADER_neg)`: an *aggregate
co-presence* measure that ignores order entirely. Both operationalise "opposite
polarity in one utterance", but Riloff's signal is the **contrast structure**
(positive predicate + negative situation), which the aggregate version cannot see.
Riloff is cited in the `rq1.R` header, so this needs either the flip count or an
explicit note that a simpler proxy was used.

### ✗ Not implemented at all

**Joshi et al. (2016) — explicit vs. implicit incongruity** · tool: `tidytext`
None of the three named features exist: *# opposite-polarity transitions*, *longest
same-polarity run*, *# positive/negative words*. Joshi is cited in the `rq1.R` header
alongside Riloff, but nothing in the code derives from it.

Note the plan's table is internally inconsistent here: the Feature column proposes
countable tidytext features, while the Tool column describes the paper's *actual*
method — cosine similarity between word pairs *within* a sentence (max/min similarity,
distance-weighted). These are two different measures. RQ2's `semantic_incong` is
neither: it is `1 − cos(comment, parent)`, a **document-level, comment-vs-parent**
measure, not within-comment word-pair discordance.

Recommendation: implement the cheap tidytext version (three counts over a lexicon-
tagged token sequence). The embedding version is defensible but expensive at 51k rows.

**Mohammad & Turney (2013) — emotional incongruity (NRC)** · tool: `syuzhet`
**Entirely absent.** No NRC anywhere. This is the most costly omission because it is
missing on three counts at once: it is in the group's plan with a citation ready, it
is core Session 7 course material (the sentiment-vs-emotion distinction and the
8 Plutchik emotions), and it is the one feature that would give the paper a
*marketing-actionable* finding — which emotions sarcastic comments carry. It is also
about five lines:

```r
nrc <- syuzhet::get_nrc_sentiment(sarcasm_clean$comment)
emo_incong <- 2 * pmin(nrc$joy + nrc$trust, nrc$disgust + nrc$anger)
```

**Kreuz & Caucci (2007) — interjections / cue words** · tool: `quanteda` + `udpipe`
**Entirely absent.** No dictionary of "gee / gosh / wow / oh". Cheap (one alternation
regex or a small `quanteda` dictionary), and the plan's own summary makes the strongest
possible case for it: it is the only feature in the table backed by *experimental
evidence on human sarcasm perception*, not just corpus statistics.

**Two placeholder sources** (the Emerald sarcasm-detection paper and `W14-2608`) were
never annotated in the plan and have no counterpart in the code.

### Summary

| Cited source | Status |
|---|---|
| Hutto & Gilbert (2014) — VADER emphasis | ✔ (intensifiers / "but" missing) |
| González-Ibáñez et al. (2011) — typography | ⚠ punctuation & quotes yes; **emoticons & laughter no** |
| Riloff et al. (2013) — polarity incongruity | ⚠ concept yes, **polarity-flip count no** |
| Joshi et al. (2016) — incongruity features | ✗ none of the three features |
| Mohammad & Turney (2013) — NRC emotion | ✗ absent |
| Kreuz & Caucci (2007) — interjections | ✗ absent |

**2 of 6 implemented as planned.**

---

## 5. The tool stack diverged completely

Verified by grep — **zero occurrences** in any script of:

`tidytext` · `syuzhet` · `quanteda` · `textfeatures` · `udpipe` · Bing · AFINN

Every "Free R tool" column entry in the plan except `vader` went unused. The code
instead uses `reticulate`+VADER, `sentimentr`, `text`, and `caret`/`glmnet`.

This is not wrong on its own — the substitutes are course-taught and defensible — but
it means the plan's table cannot be pasted into the term paper as a methods table. It
currently describes an analysis that was not run.

---

## 6. Smaller mismatches

- **The follow-up question is only half-answered.** "Are sarcastic comments *more
  likely* to be positive or negative?" is comparative, but Figure 1 filters to
  `label_f == "Sarcastic"` only. Dropping the filter and filling by label answers the
  question as asked, in one line.
- **The LIWC-substitute argument is half-delivered.** The plan's logic is: LIWC is
  paid, so substitute free lexicons. The code substitutes LIWC's *punctuation and
  capitalization* categories but never its *affect and emotion* categories — which is
  what LIWC is actually known for. NRC (§4) is that substitute, and it's missing.
- **`rq2:190`'s broken `???` regex** silently degrades the ellipsis feature that the
  González-Ibáñez row is meant to support.

---

## 7. What to do, in order

1. **Add NRC emotional incongruity** (`syuzhet`) — ~5 lines, closes a plan citation
   *and* a course gap, and yields the most marketing-relevant finding available.
2. **Add emoticon + laughter counts** — ~2 lines, closes the González-Ibáñez row.
3. **Add an interjection dictionary count** — ~2 lines, closes the Kreuz row.
4. **Decide RQ2's feature set** — re-use the cached VADER features (option a), or write
   the one-sentence justification (option b). Do not leave it unstated.
5. **Add a polarity-flip count** so Riloff is implemented as cited, or soften the
   header comment.
6. **Either implement Joshi's three counts or remove the citation** from the `rq1.R`
   header. Citing a paper whose method you didn't use is the kind of thing an examiner
   checks.
7. **Fix the `???` regex** and the `punct_total` collinearity before reading `varImp`.

Items 1–3 and 5 are together well under an hour and take the literature table from
2/6 to 5/6 implemented.
