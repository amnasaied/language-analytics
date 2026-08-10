# Text Cleaning — Class Material vs. Exam Project

Cross-check of the preprocessing taught in the course wiki + exercise solutions
against what the three exam scripts actually do.
Companion to [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

**Sources checked**
- `wiki /language-analytics/session1-text-data-regex.md` (regex)
- `wiki /language-analytics/session2-preprocessing-bow-tfidf.md` (the 8-step pipeline)
- `wiki /language-analytics/session4-text-classification.md`
- `wiki /language-analytics/session7-sentiment-emotion-analysis.md`
- `wiki /language-analytics/session8-embeddings-transformers.md`
- `wiki /language-analytics/la-exercises.md`
- `02_Exercises/Exercise_02_Solution.Rmd.txt`, `Exercise_04_Solution.Rmd`, `Exercise_07_Solution.Rmd`, `Exercise_08_Solution.Rmd`

All counts below were computed on the actual filtered dataset in `.RData`
(51,337 rows).

---

## The headline

**The exam project does *not* use the Session 2 preprocessing pipeline — and that
is correct, not a gap.**

The course taught **two different cleaning regimes**, and the wiki is explicit
about which belongs where:

| Regime | Taught in | Used when | What it does |
|---|---|---|---|
| **A — Heavy `tm` pipeline** | Session 2, Ex 2, Ex 4 | You are building a **DTM / BoW / TF-IDF** matrix, or doing **topic modelling** | lowercase → stopwords → punctuation → numbers → stem/lemmatize → whitespace |
| **B — Light filter, raw text** | Ex 7 (sentiment), Ex 8 (embeddings) | You are using **VADER / sentimentr / transformer embeddings** | drop `NA` / empty / very short, keep the text otherwise untouched |

The exam builds **no DTM anywhere** — RQ1 uses regex counts + VADER, RQ2 uses
transformer embeddings + sentimentr, RQ3 uses `nchar()`. So Regime A is
genuinely out of scope, and the scripts correctly follow Regime B.

In fact, applying Regime A here would **destroy the analysis**. Session 7 lists
VADER's special capabilities as *"CAPITALIZATION boosts intensity, punctuation
boosts (!!!), degree modifiers, negation"*. The exam's features **are**
capitalization and punctuation. Lowercasing and `removePunctuation` would zero
out three of RQ1's five features and cripple VADER. Session 2's own stopword rule
(*"stop-word removal MUST be before punctuation removal"*) is moot when you
remove neither.

> **If asked to justify this in the exam:** cite Session 7 §7 (VADER uses caps +
> punctuation as signal) and Session 8 §4 (transformers tokenize sub-words and
> need natural text). Cleaning choice follows the *representation*, not habit.

---

## Step-by-step comparison

### Regime A — the 8-step Session 2 pipeline

| # | Session 2 step | In exam scripts? | Verdict |
|---|---|---|---|
| 1 | `VCorpus(VectorSource())` | ✗ | **N/A** — no corpus object needed; all work is on the tibble column |
| 2 | `tolower()` | ✗ | **Correctly skipped.** Caps is Feature 3 in RQ1; transformer models in Ex 8 are cased |
| 3 | Stop-word removal | ✗ | **Correctly skipped.** VADER/sentimentr need negators ("not", "no", "never") — these are *in* the snowball stoplist. Removing them inverts sentiment |
| 4 | Implicit-word removal | ✗ | **N/A** — no vocabulary is being built |
| 5 | `removePunctuation` | ✗ | **Correctly skipped.** Punctuation is Feature 1 |
| 6 | `removeNumbers` | ✗ | **N/A** — no DTM; numbers are harmless to VADER/embeddings |
| 7 | Stem / lemmatize | ✗ | **Correctly skipped.** Would corrupt sub-word tokenization (Session 8: WordPiece already handles morphology, `playing → [play, ##ing]`) |
| 8 | `stripWhitespace` | ✗ | **Minor gap** — see finding 4 below |

### Regime B — what Ex 7 and Ex 8 actually did

Both class exercises open with the exact same line:

```r
filter(!is.na(Content), Content != "", nchar(Content) > 10)
```

| Class step | rq1.R | rq2_embeddings.R | rq3_score_analysis.R |
|---|---|---|---|
| Drop `NA` text | ✔ comment + parent | ✔ comment + parent | ✔ comment only |
| Drop empty `""` | ✔ `comment != ""` | ✔ `nchar(trimws(.)) > 0` | ✗ |
| **`nchar > 10` minimum** | **✗** | **✗** | **✗** |
| Drop `[deleted]` | ✔ | ✗ | ✗ |
| Stable row ID | — | ✔ `row_id` | — |
| Readable label factor | ✔ `label_f` | ✔ `label_f` | ✔ `label_f` |
| Length normalization | **✗** | ✔ `punct_density` | ✔ `comment_length` control |

---

## Findings

### 1. The `nchar > 10` minimum-length filter is missing — the one real gap

This is the only Regime-B step the class used in **both** the sentiment and the
embeddings exercise, and **none** of the three exam scripts apply it.

Measured impact:

| | Count | Share |
|---|---|---|
| Comments with `nchar <= 10` | 3,069 | 6.0% |
| Parent comments with `nchar <= 10` | 1,086 | 2.1% |
| One-word comments | 2,205 | 4.3% |
| Comments with **no alphabetic characters at all** | 256 | 0.5% |

Median comment is only **45 characters** — this corpus is much shorter than the
Amazon reviews the class rule was calibrated on, so the tail matters more here.

Why it matters for *these specific* features:
- VADER returns `compound = 0` on a comment with no lexicon hits → RQ1's Feature 2 is 0, and Feature 4 (`2 * pmin(pos, neg)`) is **structurally 0**
- RQ2's `intra_comment_incong` is *defined* as 0 for single-sentence comments — a one-word comment can never contribute
- A mean-pooled transformer embedding of `"lol"` is close to noise

**But check before applying it — the filter is not label-neutral:**

| Label | `nchar <= 10` | Share of that class |
|---|---|---|
| Non-sarcastic (0) | 2,122 | **8.5%** |
| Sarcastic (1) | 947 | **3.6%** |

Short comments are **2.4× more common in the non-sarcastic class**. Dropping them
would shift the class balance and remove a genuine signal (sarcasm needs room to
set up an incongruity). This directly interacts with RQ3, where `comment_length`
is a *control variable*.

**Recommendation:** don't silently adopt the class rule. Either (a) keep all rows
and state that length is treated as signal, not noise — noting RQ3 shows why — or
(b) apply `nchar > 10` and report it as a robustness check. Option (a) with one
sentence of justification is the stronger exam answer; option (b) as an appendix
is stronger still.

### 2. Session 7's "always normalize by length" is applied inconsistently

Session 7 §10 is unambiguous: *"Longer reviews have higher absolute scores.
Always normalize."* Ex 7 normalizes AFINN by word count.

- `rq2_embeddings.R:193` — ✔ does it: `punct_density = punct_total / comment_length`
- `rq1.R:129` — ✗ does not: `punct_total = excl + quest + ellipsis + quote`, a **raw count**

So RQ1's headline punctuation feature is partly measuring *comment length*, and
RQ1 has no length control anywhere in its logistic regression. Given finding 1
(sarcastic comments are systematically longer), **RQ1's punctuation effect is
confounded with length.** RQ1 does normalize capitalization (`caps_ratio`) — so
the fix is a one-liner and consistent with the script's own style:

```r
punct_density = punct_total / pmax(comment_length, 1)   # as in rq2
```

This is the finding most likely to affect a stated result.

### 3. Session 7's "winnowing — remove duplicates" is not done

Session 7 §1 lists *"Winnowing — remove duplicates, merge overlaps"* as a step.

- **1,221 duplicated comment strings (2.4%)** — Reddit copypasta, bot replies, stock phrases

Roughly balanced across classes (582 non-sarcastic / 639 sarcastic), so it will
not bias the direction of results — but it inflates *n* and violates the
independence assumption behind the t-tests and Wilcoxon tests in RQ1/RQ3, where
p-values are already near-zero from sample size alone. One line fixes it:

```r
distinct(comment, .keep_all = TRUE)
```

### 4. `trimws` / empty-string handling differs between scripts (cosmetic here)

Three different filters for the same idea:

```r
# rq1.R:102   — misses whitespace-only strings
filter(comment != "", parent_comment != "")
# rq2:167     — correct, catches "   "
filter(nchar(trimws(comment)) > 0, nchar(trimws(parent_comment)) > 0)
# rq3:161     — no empty check at all
filter(!is.na(score), !is.na(label))
```

**Verified: this changes nothing on this dataset.** All three filters keep
**exactly 51,335 rows** (0 whitespace-only, 0 empty strings, 2 `NA` comments).
Worth aligning for hygiene and defensibility, but it is not affecting any result.

### 5. The `[deleted]` filter is a no-op — and the comment overstates it

`rq1.R:103` filters `[deleted]` with a justifying comment; `rq2` and `rq3` don't.
I checked: **there are 0 `[deleted]` and 0 `[removed]` rows** in the filtered
data — the balanced SARC export already excludes them.

The filter is good defensive practice and worth keeping, but the RQ1 header
implies it is doing work. Either soften that comment or add the same line to
RQ2/RQ3 so the three scripts are visibly consistent.

### 6. Emoji handling (Session 7 §9) — genuinely not needed here

Session 7 §9 teaches `textclean::replace_emoji()` / `replace_non_ascii()`.
I checked: **0 comments contain any non-ASCII character** (0.00%). This export
is ASCII-only. Nothing to do — and worth one sentence in the writeup, because
"we checked and it didn't apply" is a stronger answer than silence.

Note that even if emoji were present, Session 7 says VADER handles them
natively (so RQ1 would be fine) but **sentimentr does not** — which would have
mattered for RQ2.

### 7. Reddit-specific markup is not stripped — low impact, worth one line

Not taught in class (the course used clean Amazon review text), so this is
domain judgment rather than a deviation:

| Artifact | Count | Effect |
|---|---|---|
| HTML entities (`&amp;` `&gt;` `&lt;`) | 175 | cosmetic; no `&quot;`, so `quote_count` is safe |
| Markdown emphasis (`*` / `_`) | 1,206 | none — not counted by any feature |
| Markdown links `[x](y)` | 8 | negligible |
| URLs (`http`) | 4 | negligible |
| Quote lines (`>`) | 0 | — |
| `/r/` or `/u/` references | 0 | — |

Real but small. Mention it as a limitation rather than fixing it.

I also checked for **label leakage** — the SARC labels come from the `/s`
convention, so a leftover `/s` in the text would be a giveaway feature:
**0 comments contain `/s`.** Clean. Worth stating explicitly in the writeup.

### 8. Session 1 regex was applied — and one instance regressed

The class taught these exact patterns (Session 1 §8, Ex 1):

| Taught | Used in exam | |
|---|---|---|
| `\\b[A-Z]{2,}\\b` (uppercase words) | `rq1.R:138`, `rq2:200` — identical | ✔ |
| `[!?]{2,}` (repeated punctuation) | `rq1.R:127` uses `\\.{3,}\|\\?{3,}` | ✔ equivalent |
| — | `rq2:190` uses `"\\.\\.\\.\|???"` | ✗ **broken** |

`rq2_embeddings.R:190` — `???` is a quantifier with nothing to quantify. The
taught pattern `[!?]{2,}` (or RQ1's `\\.{3,}`) is the correct form and was
available. This silently degrades RQ2's `n_ellipsis` feature.

### 9. Two different sentiment engines across scripts — defensible, but say so

Both are course-taught (Session 7 covers both), but the same construct is
operationalized differently in RQ1 and RQ2:

| Feature | RQ1 | RQ2 |
|---|---|---|
| Sentiment engine | **VADER** | **sentimentr** |
| Intra-comment incongruity | `2 * pmin(VADER_pos, VADER_neg)` | SD of sentence-level sentiment |
| Parent-reply incongruity | `abs` diff of VADER compound | `abs` diff of `ave_sentiment` |

Session 7's own takeaway is *"method choice materially changes conclusions"* and
Ex 7 exists specifically to compare AFINN vs sentimentr vs VADER — so this is
defensible and even shows range. But an examiner will ask why RQ1's finding
doesn't transfer directly to RQ2. **Add one sentence naming the switch and the
reason** (sentimentr gives sentence-level scores, which the SD-based incongruity
measure requires; VADER does not).

Also note: Session 7 says **VADER is "best for social media text, short informal
texts, Twitter/Reddit"** — that's an explicit endorsement of the RQ1 choice, and
a free citation for the writeup.

### 10. One correction to the earlier overview

`microsoft/harrier-oss-v1-270m` is **the model used in class**
(Ex 8 / `la-exercises.md`: `textEmbed(model="microsoft/harrier-oss-v1-270m",
layers=-2, aggregation="mean")`, 640-dim). My earlier note suggesting it be
verified was wrong — RQ2 matches the taught call exactly, including `layers=-2`
and mean aggregation.

The wiki adds one caveat RQ2 should acknowledge: **anisotropy** — modern
contrastively-trained encoders push all sentences into a narrow cone, so even
unrelated texts score ~0.9 cosine. RQ2's `semantic_incong = 1 - cos(comment,
parent)` will therefore sit in a narrow band near 0. *Relative ranking is
meaningful, the absolute value is not* — state this when interpreting that
feature.

---

## Summary scorecard

| Class practice | Status |
|---|---|
| Session 2 heavy `tm` pipeline | **Correctly not applied** (no DTM anywhere) |
| Ex 7/8 light filter: `!is.na`, `!= ""` | ✔ Applied |
| Ex 7/8 light filter: `nchar > 10` | **✗ Missing** — but not label-neutral, see finding 1 |
| Session 7: normalize by length | **⚠ RQ2 yes, RQ1 no** — likely confound |
| Session 7: remove duplicates | **✗ Missing** (2.4%) |
| Session 7: emoji handling | ✔ N/A — verified 0 non-ASCII |
| Session 7: VADER for social media | ✔ Explicitly endorsed by the wiki |
| Session 1: taught regex patterns | ✔ RQ1 · **✗ RQ2 line 190 broken** |
| Session 4: stratified 80/20, 10-fold CV, `set.seed` | ✔ All three scripts |
| Session 8: `textEmbed` args, `saveRDS` after | ✔ Matches Ex 8 exactly |
| Session 8: anisotropy caveat | **✗ Not acknowledged** |

## Highest-value fixes, in order

1. **`rq1.R:129`** — normalize `punct_total` by length (finding 2). Only item likely to change a reported result.
2. **`rq2_embeddings.R:190`** — fix the `???` regex (finding 8).
3. **Add a deduplication line** (finding 3) — cheap, defensible, addresses an explicit class step.
4. **Write one paragraph justifying the cleaning strategy** — that Regime A was deliberately not applied, citing Session 7 §7 and Session 8 §4. This converts the biggest apparent gap into a demonstration of judgment.
5. **Decide and state the `nchar > 10` position** (finding 1), ideally with a robustness check.
6. **Note the anisotropy caveat** when interpreting `semantic_incong` (finding 10).
