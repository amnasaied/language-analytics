# Course Coverage Cross-Check — What the Project Uses, What It Misses

Session-by-session comparison of the course material in `wiki /language-analytics/`
against the three RQ scripts. Companion to [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
(what the project does) and [CLEANING_CROSSCHECK.md](CLEANING_CROSSCHECK.md)
(preprocessing only — this document covers *methods and research design*).

**Timing context:** the term-paper deadline in `session11-course-wrap-up.md` is
**Fri 28.08.2026**. Everything below is prioritised on that basis.

---

## 1. Coverage matrix

| Session | Topic | Used in project? | Verdict |
|---|---|---|---|
| 1 | R basics, regex | **Yes** — `str_count` feature regexes | ✔ taught patterns reused; one regressed (`rq2:190`) |
| 2 | Preprocessing, BoW, DTM, TF-IDF, Zipf, word clouds, `findAssocs` | **No** | ⚠ correctly skipped *as preprocessing*, but the **descriptive/lexical layer is missing entirely** |
| 3 | Distance, hierarchical clustering, k-means | **Partly** — cosine in RQ2's `semantic_incong` | ✔ enough; clustering not needed |
| 4 | Classification: LDA-classifier, KNN, SVM, FastText, NN, CV, metrics | **Partly** — only `glm` and `glmnet` | ⚠ **no course-taught classifier is used; no `resamples()` comparison** |
| 5 | LDA topic modelling, n-grams, PMI | **No** | ✗ **missing** |
| 6 | STM (topics × metadata) | **No** | ✗ **missing — biggest method gap** |
| 7 | Sentiment & emotion | **Partly** — VADER (RQ1), sentimentr (RQ2) | ⚠ **no NRC emotion, no method comparison, no transformer sentiment** |
| 8 | Embeddings & transformers | **Partly** — `textEmbed()` only | ⚠ **no `textProjection()`, no `textTrain()`, no word2vec, anisotropy unacknowledged** |
| 9 | GenAI APIs / LLM-as-annotator | **No** | ✗ **missing — and the course names irony as the flagship use case** |
| 10 | LLM limitations & biases | **No** | ✗ **missing — no limitations/ethics section anywhere** |
| 11 | Method selection, operationalisation chain, paper writing | **No** | ✗ **no prose deliverable exists** |

---

## 2. The gaps that matter, ranked

### Gap 1 — No term paper exists (Session 11)

`outputs.docx` is 36 pasted figures with no prose. Session 11 gives an explicit
structure: introduction (hook → rationale with *findings* foregrounded → hypotheses),
results written first, discussion with **limitations early**, no separate
"future research" section, end by returning to the hook.

It also gives the **operationalisation chain** that the project has never written
down: *Construct → Operationalisation → Method → Expected pattern*, with each
operationalisation justified by a citation. The project has good operationalisations
(Riloff/Joshi incongruity) but they are only defended in code comments.

**This is the highest-priority gap by a wide margin** — it is the graded artefact.

### Gap 2 — Sarcasm/irony is the course's canonical LLM-API use case, and no API is used (Session 9)

`session9-analyzing-text-genai-apis.md` puts **"high-context classification (irony)"**
in the *"Strong fit — APIs win"* column, against *"large labelled data + stable
categories → SVM/FastText"* in the other. The project sits exactly on the boundary
this table was written to adjudicate, and never mentions it.

Two concrete uses, in order of value:

**(a) Label validation.** The SARC labels are self-labelled via the `/s` convention —
noisy, author-dependent ground truth, and never validated in this project. Session 9
step 4 prescribes exactly the fix: hand-annotate 100–150 cases, compute
**inter-iteration agreement** (reliability) and **agreement with human labels**
(validity) via **Krippendorff's α** (>0.667 tentative, ≥0.8 reliable). This turns
"the labels might be wrong" from an unaddressed weakness into a reported measurement.

**(b) A third model arm.** RQ2 currently compares embeddings vs embeddings+features.
Adding a zero/few-shot Gemini annotation arm makes it a three-way comparison
covering the whole course toolkit, and the black-box trade-off (interpretability,
reproducibility, cost) is a ready-made methods paragraph the wiki already argues for.

Cost is low — Gemini Flash Lite on a 1,000-comment sample is cents. Note
`Sys.getenv("GEMINI_API_KEY")`, never a committed key.

### Gap 3 — STM is the obvious fit for this data and is unused (Sessions 5, 6, 11)

Session 11's decision framework, Q4: *"Need to track a construct over time or across
groups? → STM."* RQ3 tracks score over time (2009–2016) with a loess smoother and
compares 62 subreddits — that is the STM question asked without STM.

The dataset has textbook STM metadata: `label`, `subreddit`, `date`.

```r
stm(K = 15, prevalence = ~ label_f + s(date_num) + subreddit, content = ~ label_f, ...)
```

- `prevalence = ~ label_f` → **which topics attract sarcasm** (a marketing finding:
  is sarcasm concentrated in product launches, price complaints, support threads?)
- `content = ~ label_f` → **how the same topic is worded sarcastically vs not** —
  this is a direct, unaddressed answer to RQ1's own question
- `s(date)` → whether sarcasm's topical footprint shifted 2009→2016
- `findThoughts()` gives readable example comments — the qualitative validation the
  project currently has none of

`labelTopics()` with **FREX** for labelling, per Session 6.

### Gap 4 — RQ1 answers "which features" without ever looking at words (Sessions 2, 8)

RQ1's question is *"which linguistic features distinguish sarcastic comments"*, and it
answers with five hand-crafted counts. Two course-taught tools answer it lexically and
are both absent:

- **`textProjection()` / `textProjectionPlot()`** (Session 8 §5) — finds the direction
  in the 640-dim space separating the two groups and projects each word onto it, with
  **permutation p-values + FDR correction**. This is the single most on-target unused
  tool in the course: it produces "these words characterise sarcastic comments" with
  significance testing, from embeddings the project already computes. Requires
  `keep_token_embeddings = TRUE`, which `rq2_embeddings.R` currently sets to `FALSE`.
- **TF-IDF + `comparison.cloud()`** (Session 2 §7, §10) — the comparison cloud is
  defined in the wiki as "words disproportionately frequent in each corpus". Cheap,
  purely descriptive, and gives the paper a figure that shows the corpus rather than
  just model output. This is the one legitimate use of the Session 2 pipeline in this
  project — for *description*, not for the models.

Also missing from Session 5: **n-grams with PMI ≥ 5**. Sarcasm markers are frequently
bigrams ("oh great", "yeah right", "so glad") that unigram features cannot see, and
the wiki is explicit that n-grams must be computed **before** stopword removal.

### Gap 5 — The course's own words: lexicons cannot detect sarcasm (Session 7)

`session7-sentiment-emotion-analysis.md` §6a lists under lexicon limitations:
**"Can't detect sarcasm/irony."** §6d says transformers "handle sarcasm, irony,
implicit sentiment". §13's method table recommends **`sentiment.ai`** for highest
accuracy in any domain.

The project's sentiment layer is entirely lexicon-based (VADER, sentimentr). That is
defensible — VADER is explicitly endorsed for Reddit in §13 — but the limitation is
never stated, and the recommended alternative is never tried. Adding `sentiment.ai`
on a subsample as a robustness check, *or* citing §6a as an acknowledged limitation,
both close this. The first is stronger.

Two further Session 7 items are unused:

- **NRC emotion (8 Plutchik emotions)** via `syuzhet::get_nrc_sentiment()` — the
  session's whole sentiment-vs-emotion distinction, plus the marketing action table
  (anger → fix the failure, disgust → brand concern). "Which emotions do sarcastic
  comments carry?" is a marketing-relevant question this dataset can answer cheaply,
  and it would add a genuinely new feature block to RQ1.
- **Method comparison.** Exercise 7 exists specifically to correlate AFINN vs
  sentimentr vs VADER; Session 7's takeaway is *"method choice materially changes
  conclusions"*. The project uses two engines in two scripts and never compares them
  (also flagged as finding 9 in `CLEANING_CROSSCHECK.md`). A single correlation
  table would convert an inconsistency into a deliberate robustness check.

### Gap 6 — No classifier comparison (Session 4)

Session 4 teaches five classifiers and ranks **SVM-linear** as "the workhorse for most
tasks" and **FastText** as the pick for "large scale, UGC with typos" — both describing
this dataset. The project uses `glm` (RQ1) and `glmnet` (RQ2), neither of which is in
the session.

Also unused: **`resamples()`** for comparing models on identical CV folds, which
Session 4 §11 presents as the correct comparison protocol and which RQ2's two-model
design would benefit from directly.

Minor: [rq1.R:303](../rq1.R#L303) tunes on `metric = "Accuracy"`. Session 4 §3 says
accuracy is the weakest choice and **F1** is "the most informative single metric for
marketing classification". RQ2 already uses `metric = "ROC"` — RQ1 should match.
(F1 *is* printed via `confusionMatrix`, so this is about the tuning objective, not
the reporting.)

### Gap 7 — RQ3 is a causal question run as OLS (Session 9, Saljoughian et al.)

Session 9's paper spends its methods section on exactly the problem RQ3 has:
regressing an outcome on a text feature when assignment is not random.

- **Endogeneity** — an unobserved factor (topic heat, thread position, author
  reputation) drives both *whether a comment is sarcastic* and *how it scores*.
- **Selection** — the corpus is comments that survived and got votes; scores are
  time- and thread-dependent. The paper's fixes were a control function / IV and
  two Heckman corrections.

Nobody expects IV in a term paper. But RQ3 currently reports `lm(score ~ label + ...)`
without naming the identification problem, and the course taught it in the session
directly upstream. **One paragraph naming endogeneity and selection, and reframing
the result as associational, is required, not optional.**

Two cheap improvements: `author` is in the data and unused — repeat authors violate
the independence assumption behind the t-tests, and author-level clustering or a
random effect would address it. And `score` is not comparable across a 2009→2016
window with growing Reddit traffic; RQ3's own time-trend figure shows this.

### Gap 8 — No limitations, bias, or ethics discussion (Session 10)

Session 10 is entirely about critically evaluating this kind of work, and Session 11
puts **limitations early** in the discussion. The project has none. The material
supplies the content:

- **The Salminen et al. critique is a template — and the project shares one of its
  flaws.** The class criticised that paper for "88 t-test/χ² comparisons with no
  type-I error control". RQ1 runs a Welch t-test **and** a Wilcoxon test per feature
  across ~10 features with **no `p.adjust()` anywhere in the repo**. The course taught
  FDR correction twice (Session 5 `topicsTest`, Session 8 `textProjection`). This is
  a one-line fix and a direct hit if an examiner reads Session 10 alongside the paper.
- **Artificial ground truth** — the same criticism levelled at Salminen's Prolific
  fake reviews applies to `/s` self-labelling: it captures *marked* sarcasm, and
  systematically misses unmarked sarcasm, which is most of it. This belongs in
  limitations, and Gap 2(a) is how to quantify it.
- **Bias in, bias out** (Session 8 §7, Session 10) — the embeddings carry the biases
  of their training corpus; sarcasm detection is register- and dialect-sensitive.
- **GDPR / UGC ethics** — Session 10 flags that pasting user data into an LLM API may
  violate GDPR. Relevant the moment Gap 2 is implemented, and worth a sentence about
  Reddit data regardless.
- **Anisotropy** (Session 8 §5) — `semantic_incong = 1 − cos(...)` will sit in a narrow
  band near 0 because modern encoders push everything into a cone. Relative ranking is
  meaningful, absolute values are not. Already noted in `CLEANING_CROSSCHECK.md` and
  still unaddressed.

---

## 3. Correctly out of scope

Not every unused method is a gap. These are defensible omissions worth one sentence
each in the paper rather than any code:

- **The Session 2 `tm` preprocessing pipeline as preprocessing** — the project builds
  no DTM for modelling, and lowercasing/`removePunctuation` would destroy RQ1's
  capitalisation and punctuation features and cripple VADER. Fully argued in
  `CLEANING_CROSSCHECK.md`. (Note this does *not* excuse Gap 4, which uses the same
  tools descriptively.)
- **k-means / hierarchical clustering (Session 3)** — the labels are known, so this is
  a supervised problem; Session 11's Q2 routes labelled data to classification.
- **KNN (Session 4)** — the session itself calls it "rarely worth it for text".
- **Seeded LDA, sentence-LDA (Session 6)** — subsumed by the STM recommendation.
- **Emoji handling (Session 7 §9)** — verified 0 non-ASCII characters in the corpus.

---

## 4. Suggested order of work

Ranked by value per hour, given 17 days to the deadline.

| # | Action | Effort | Closes |
|---|---|---|---|
| 1 | **Write the paper** — Session 11 structure, operationalisation chain per feature | High | Gap 1 |
| 2 | **Limitations + ethics section**, incl. `/s` ground-truth caveat and anisotropy | Low | Gap 8 |
| 3 | **`p.adjust(method = "fdr")`** on all RQ1 group tests | Trivial | Gap 8 |
| 4 | **One paragraph on endogeneity/selection in RQ3**, reframe as associational | Low | Gap 7 |
| 5 | Apply the three pending fixes from `CLEANING_CROSSCHECK.md` (`punct` normalisation, `???` regex, `distinct()`) | Trivial | — |
| 6 | **TF-IDF + comparison cloud** of sarcastic vs non-sarcastic | Low | Gap 4 |
| 7 | **NRC emotion profile** by label | Low | Gap 5 |
| 8 | **RQ1 tuning metric → ROC/F1**; add an SVM-linear baseline with `resamples()` | Low | Gap 6 |
| 9 | **`textProjection()`** on the existing embeddings (needs `keep_token_embeddings = TRUE`) | Medium | Gap 4 |
| 10 | **STM** with `prevalence = ~ label + s(date)`, `content = ~ label` | Medium | Gap 3 |
| 11 | **LLM validation arm** — 150 hand-coded cases, Gemini, Krippendorff's α | Medium–High | Gap 2 |

Items 2–5 are a single afternoon and remove the objections an examiner reaches for
first. Items 10 and 11 are what turn this from a competent feature-engineering
exercise into a project that covers the course.
