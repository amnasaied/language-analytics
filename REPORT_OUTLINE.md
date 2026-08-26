# Report Outline (v2) — Sarcasm Detection in Marketing-Relevant Reddit Communities

Corrected outline, remapped onto the chair's **binding 6-section skeleton**
(`PAPER_TEMPLATE.md`), aligned to the canonical code (`Final_Project_report.R`)
and the three `RQ*_Methodology.md` files. Word budgets are at 2.0 spacing
(~250–300 words/page). Target 10–15 pages excl. title page + references.

**What changed from the team's first outline (and why):**
1. "Related Work" → **Theory**, built on the **construct-validity-threat spine** and
   grounded in the course sources (`cta-dev/language-analytics/Research Questions +
   Notes.md` feature table + the SARC and Salminen readings), plus a small set of
   verified marketing anchors (Timoshenko & Hauser 2019; Salminen et al. 2025; Berger
   et al. 2023 optional). Recovers the 10% marketing-relevance + 10% theory points and
   avoids the "list, don't synthesise" trap. **Not** from `THEORY_PLAN.md`.
2. "Limitations + Future Direction" (§7) and "Conclusion" (§8) **deleted as
   standalone sections** — the chair forbids a separate future-work section and
   wants limitations *early inside* the Discussion, which ends on the opening hook.
3. "Dataset" folded into **Methods §3.1**.
4. RQ1 feature ranking corrected to **one method** (standardised OR + FDR); the old
   "3-method (AUC + random forest)" ranking was deliberately removed from the code.
5. RQ3 regression labelled as **mixed-effects** (author random intercept is in the
   final model, not optional); RQ2 **10-split win-rate** robustness added.
6. Unread references (Rajadesingan, Barbieri, Reyes) flagged for drop/verify.

---

## Title Page (not counted)
Working title: **"When the Instrument Misreads: Sarcasm as a Validity Threat to
Marketing Text Analytics in Consumer Reddit Communities"**
(alt, plainer: *"Detecting and Understanding Sarcasm in Consumer-Focused Reddit
Communities"*). Names, course, date. Team of 5 — **confirm the chair allows >4.**

## Abstract (~120 words, write last) — *optional, not in the chair's spec*
Dataset (SARC, filtered to 62 marketing/consumer subreddits), the three RQs, the
three methods (psycholinguistic features / semantic embeddings / score comparison),
and one-sentence result per RQ. Not counted in the page budget.

---

## 1. Introduction — 1–1.5 pg (~350–450 words) · *Wiktoria* · **write last**

- **¶1 The punch.** A concrete sarcastic product comment whose surface sentiment
  inverts its meaning ("Oh great, another flagship that bends in my pocket — love
  it"), and the brand dashboard that scores it *positive*. No definition, no "In
  recent years…".
- **¶2 Rationale (marketing).** Sentiment tooling drives brand tracking and product
  feedback; sarcasm corrupts it *systematically*, biased toward the product
  failures monitoring exists to catch. ≤2 citations per step (Timoshenko & Hauser
  2019; Berger et al. 2023).
- **¶3 Purpose + predictions.** Restate purpose; 2–3 sentences on the logic of the
  three methods (no detail); close with one sentence predicting the pattern —
  *surface features weak, embeddings dominant, no engagement premium.* Past tense.

---

## 2. Theory — 2.5–3 pg (~700–900 words) · *Amna + Tuan Anh* · **drafting core**

**Spine (one thesis for all three RQs):** firms "listen" to consumers by mining UGC —
measuring brand sentiment, detecting product problems, gauging engagement — and every
such instrument assumes text means what it literally says. *Sarcasm is the systematic
violation of that assumption, so it is not a classification curiosity but a
**construct-validity threat to the measurement instrument marketing analytics runs
on** — and how costly it is depends on whether sarcasm leaves recoverable surface
traces.* This spine makes the paper *marketing* research, turns RQ1's weak result into
a theory test, and gives RQ3 a managerial meaning.
Rule: **synthesise, don't list; ≤2 citations per step.** Every added marketing ref
must be opened/verified before submission (zero-tolerance).

- **2.1 Sarcasm as a construct in UGC** (~120 w). Define: an **incongruity between the
  literal and the intended meaning** of an utterance, produced with mocking intent.
  Separate from **negation** (explicit, lexically visible) and broad **irony**
  (situational). Why it's hard: sarcasm is **intent-dependent**, and even humans
  disagree (SARC inter-annotator Fleiss κ ≈ 0.5) — no text feature observes intent
  directly, which §3.2's label-validity discussion cashes in. *Riloff et al. 2013
  (contrast-based definition); Khodak et al. 2018 (intent-dependence). Optionally Grice
  1975 for the pragmatic anchor — verify.*
- **2.2 Sarcasm as a validity threat to marketing analytics** (~180 w, 2 ¶). ¶1 the
  instrument: marketing reads UGC at scale to surface customer needs and track brand
  perception, resting on the literal-meaning assumption (**Timoshenko & Hauser 2019** —
  UGC as marketing instrument). ¶2 the breach: sentiment/dictionary measures invert on
  sarcasm, and the miss is **systematic** — biased toward the product-failure and
  launch moments monitoring exists to catch. Scope it: base rate ~0.25% overall but
  **concentrated in consumer-product communities** (Khodak et al. 2018), so exposure is
  uneven across exactly the categories marketers watch.
- **2.3 How sarcasm becomes detectable** (~300 w, 3 ¶) — **critical-engagement points
  are won here.** Staged disagreement (not a survey); each account implies a different
  feature family (all sourced from the group's feature table):
  | Account | Claim | Implies |
  |---|---|---|
  | Pragmatic marker | speakers signal ironic intent (interjections, caps, heavy punctuation, quotes); genuine emoticons/laughter mark *sincerity* | typographic markers carry signal |
  | Context incongruity | a positive expression about a negative situation; the clash itself is the signal | within-comment + comment-vs-parent sentiment distance carries signal |
  | Context dependence | neither surface family suffices; recovering intent needs conversational context, which distributed representations encode | embeddings dominate hand-crafted features |
  ¶1 marker (Kreuz & Caucci 2007; González-Ibáñez et al. 2011) — reinforced by SARC's
  `:)`/`lmao` weighting *toward sincerity*. ¶2 incongruity + the tension: positive
  predicate on a negative situation is the dominant form (Riloff et al. 2013); semantic
  discordance detects it even without sentiment words (Joshi et al. 2016) — marker
  account says sarcasm is written to be *obvious*, incongruity account says *resolved*.
  ¶3 context + the resolution RQ2 tests: annotators need conversational context
  (Wallace et al. 2014); SARC baselines show bag-of-bigrams (75.8%) *beat* summed
  embeddings (71.0%) yet none nears the 92.0% human ceiling. The marketing precedent is
  decisive and **untested at the seam**: Salminen et al. (2025) ran psycholinguistics
  (67.5%) and a transformer (BERT 82.6%) as **separate** experiments and never combined
  them — that *substitutes-vs-complements* question is RQ2's contribution.
- **2.4 Sarcasm and engagement** (~140 w) — RQ3's parent, stated as a prior worth
  testing: in SARC, comment **length and score were *not* informative** sarcasm
  features (Khodak et al. 2018). Two opposed predictions: *uplift* (sarcasm is humour /
  in-group signalling that rewards the reader — **Berger et al. 2023**, optional
  engagement anchor, verify) vs *no uplift / community-level* (sarcasm imposes a
  processing cost and fails without shared context — Wallace et al. 2014 — so any
  reward is a property of sarcastic *communities*, not of sarcasm itself). **Predict in
  advance:** the effect attenuates once subreddit is controlled — which converts our
  RQ3 null into a confirmed prediction.
- **2.5 Gap + research questions** (~110 w). Gap in 3 sentences: sarcasm detection is a
  CS literature optimised for accuracy on general corpora; marketing analytics assumes
  literal meaning and has not measured what figurative language costs its instruments;
  no study has tested whether psycholinguistic features add *over* contextual
  embeddings, or whether sarcasm earns an engagement premium in consumer communities.
  Then the three RQs **verbatim from the group's notes**, one directional expectation
  each:
  - **RQ1** — what features distinguish sarcastic from non-sarcastic comments?
    *(follow-up: are sarcastic comments more positive / negative / neutral?)* Expect
    markers present in predicted directions but weak.
  - **RQ2** — can integrating semantic embeddings with psycholinguistic features
    improve sarcasm detection in UGC? Expect a small positive increment over
    embeddings alone.
  - **RQ3** — do sarcastic comments generally receive higher public approval (scores)
    than non-sarcastic ones? Expect a weak effect that disappears under community
    controls.

---

## 3. Methods — 3–4 pg (~850–1,200 words) · *Tuan Anh (RQ1) + Emin (RQ2) + RQ3*
**Describe how each method was used, not what it is. No results here.**

- **3.1 Data.** SARC "balanced" export — cite **Khodak et al. (2018)** *and* the
  **HuggingFace distribution**. State plainly it is a **public dataset we downloaded
  — never scraped.** Marketing-subreddit filter: 64 names / 6 thematic groups (tech,
  hardware, gaming, cars, fashion, retail/entertainment) → **51,337 comments, 62
  subreddits, ~50/50, 2009-09→2016-12.** Justify the filter (feasibility + every
  remaining comment marketing-relevant).
- **3.2 Label validity.** `/s` is **author self-annotation**, not independent coding
  — say so, and acknowledge it as a limitation. *(Team decision: no manual
  hand-coding check is being run, so §3.2 states the caveat rather than reporting
  agreement numbers.)*
- **3.3 Preprocessing** (worth 10% on its own). Order + rationale: drop NA/empty →
  **strip `/s` (critical anti-leakage — the marker that generated the labels)** →
  remove Reddit markup/HTML/URLs → dedupe (`distinct(comment, parent)`). Balance
  re-checked (~50/50). After cleaning: **51,299 comments.**
- **3.4 Operationalisation — the hypothesis-translation chain (Session 11 slide 19).**
  Show the chain **for every RQ** (so all three hypotheses visibly go through the
  process), then expand RQ1 to feature level.
  *RQ-level chain:*
  | RQ | Construct | Operationalisation | Method | Expected pattern |
  |---|---|---|---|---|
  | RQ1 | pragmatic / incongruity markers of ironic intent | 11 regex + VADER features (expanded below) | Wilcoxon/χ² + FDR, logistic + subreddit FE | markers in predicted directions, mostly weak |
  | RQ2 | incremental value of psycholinguistics over semantics | 640-dim embeddings vs embeddings + 11 features | ridge-logistic ablation, McNemar, 10 splits | Model 2 > Model 1 (small) |
  | RQ3 | public approval | Reddit `score` (net upvotes) | full-sample Wilcoxon + mixed-effects regression | sarcastic higher, attenuates under controls |
  *RQ1 feature-level expansion* (one row per feature family, Construct → Operationalisation
  → Method → Expected, each measure justified + cited): 11 features (regex counts + VADER
  compound). Sentiment incongruity is operationalised as VADER `|comment−parent|`.
- **3.5 Analysis, per RQ** (model, tuning, metric, split, seed = **1**):
  - **RQ1** — univariate: Wilcoxon + rank-biserial (density features), chi-square +
    Cramér's V (sparse), **BH-FDR** across tests. Multivariate: logistic regression,
    standardised predictors, all features + comment length + **subreddit fixed
    effects** (top-30 individually, rest pooled "Other"); multicollinearity checked
    (max ρ = 0.31, VIF ≈ 1). Ranking: **standardised OR + FDR only** (predictive ranking
    is RQ2's job). Supplementary separability: LDA/PCA.
  - **RQ2** — embeddings: `microsoft/harrier-oss-v1-270m` *(⚠ verify the ID resolves
    on HuggingFace before final)*, 640-dim, `"Context: <parent>\nReply: <comment>"`
    (decoder-only, no trained `[SEP]`), cached. **Leakage-safe 80/20 split by
    parent.** Three models: features-only (RQ1 refit) / **Model 1 embeddings** /
    **Model 2 embeddings+features**, ridge logistic (`cv.glmnet`, α=0, λ by internal
    CV). Metrics: acc/prec/rec/F1/**AUC**; **McNemar** for paired significance.
    Robustness: **XGBoost** (model-class) + **10 repeated splits** (sampling).
  - **RQ3** — score cleaning (drop NA/non-integer; **no outlier trimming** — virality
    is signal); `[deleted]` authors given unique pseudo-IDs. Full-sample **Wilcoxon +
    rank-biserial**; **mixed-effects regression**: `sign(score)·log1p(|score|) ~ label +
    length + subreddit_pooled + time_scaled + (1|author)`; linearity (poly) check.

---

## 4. Results — 3–4 pg (~850–1,200 words) · *Emin* · **write this section first**
Photo-essay: 4–6 figures carry the findings; rest to appendix. State each finding
as behaviour, attach the statistic as documentation, link to its RQ. Do not narrate
tables.

- **4.1 RQ1 — which features distinguish sarcasm.**
  - Verdict table (9 judged features): **Exclamation** V=0.133, OR **1.33/SD**,
    Confirmed–moderate (the only meaningful one); **Emoticons** V=0.083, OR 0.83 and
    **Laughter** OR 0.95 — *reversed sincerity cues* (sarcastic authors avoid them);
    Interjections V=0.052, OR 1.11; everything else confirmed-direction but
    negligible; **Quotation marks Mixed** (flips sign once controlled); **Ellipsis
    Not supported.** 7 of 9 confirmed in direction.
  - Follow-up sentiment sub-question (χ²=18.61, p<0.001): surface polarity **barely
    differs** — sarcastic comments are, if anything, **marginally more negative** (21.8%
    vs 20.2% negative; ~38% positive both), a shift too small to be useful. Read as a
    tendency, not a rule; surface polarity does not discriminate sarcasm.
  - Supplementary: LDA/PCA show heavy class overlap (consistent with weak features).
  - *Figures: feature effect-size / verdict plot; sentiment-composition bar.*
- **4.2 RQ2 — do features add over embeddings.**
  - 3-way table (F1 / AUC): **Model 2 (emb+feat) 0.687 / 0.739** > **Model 1 (emb)
    0.683 / 0.725** ≫ **features-only 0.608 / 0.618.** XGBoost mirrors the ordering.
  - Significance: **McNemar M1 vs M2 p = 0.023**; both embedding models vs
    features-only **p < 0.001**.
  - Robustness: across **10 splits, Model 2 wins 9/10**, ΔF1 +0.003±0.003, ΔAUC +0.011
    — small but consistent, not one-split noise.
  - *Figures: ROC curves; 3-way metric bars (confusion matrix → appendix).*
- **4.3 RQ3 — do sarcastic comments score higher.**
  - Descriptive: median **1 (non) vs 2 (sarc)**, IQR 2 vs 4.
  - Comparison: full-sample Wilcoxon (n=51,299) sarcastic higher, r=0.033,
    p=2.8e-11 (significant but **negligible** effect — the large-N trap);
    mixed regression **label β = +0.016, p=0.12, ns** (controls behave: length &
    time p<0.001; author SD≈0.24).
  - One sentence: both methods agree in direction, magnitude negligible, no
    reliable effect once controlled.
  - *Figure: subreddit heterogeneity in score — 1 only.*

---

## 5. Discussion — 2–3 pg (~550–850 words) · *Amna* · ends the paper
Chair's order. Avoid the six mistakes (too long, disorganised, summarising-not-
discussing, off-topic, wrong tone, self-praise). **No separate Conclusion / Future
Work section.**

1. **Orienting ¶** (1–3 sentences): surface features weak, embeddings dominant, no
   engagement premium.
2. **Limitations early** (brief — the honest four): self-annotated `/s` labels; one
   platform; 2009–2016 vintage; subreddit filter proxies marketing relevance rather
   than measuring it. Add the three biases: self-selected sarcasm markers, subreddit
   demographics, embedding anisotropy.
3. **1–3 study-specific issues tied back to §2 theory:**
   - RQ1's weakness *confirms* the context-dependence account, not a failed model
     (the theory test) — and the reversed emoticon/laughter cues.
   - RQ2 — features add a *real but small* increment (McNemar-significant, 9/10
     splits): embeddings implicitly capture most of what the markers measure →
     substitutes more than complements. **Fold the "trained implicit-incongruity
     classifier" follow-up into this paragraph** (not a future-work section).
   - RQ3 — the null-under-controls result was *predicted from theory in advance* (§2.4:
     a community-level, not comment-level, effect — consistent with SARC's finding that
     score is not an informative sarcasm feature): sarcasm doesn't buy engagement;
     which communities are sarcastic does.
4. **Implications** (practical/managerial first): brand-monitoring on consumer
   platforms needs contextual models, not sentiment lexicons; sarcasm concentrates in
   exactly the product-failure moments monitoring targets. Be humble (no "we are the
   first…").
5. **End strong** — return to the §1 dashboard hook, updated: the dashboard still
   misreads the comment, and the fix is contextual representation. State the take-home.

---

## 6. References — not counted · APA 7th, verified from DOIs (Zotero export required)
Grounded in the group's `Research Questions + Notes.md` feature table + the two course
readings + the R package citations in the code.
- **Sarcasm / linguistics:** Khodak, Saunshi & Vodrahalli (2018, SARC) + the HuggingFace
  distribution · Riloff et al. (2013) · Joshi et al. (2016) · González-Ibáñez et al.
  (2011) · Kreuz & Caucci (2007) · Wallace et al. (2014, `aclanthology.org/W14-2608`) ·
  Hutto & Gilbert (2014, VADER).
- **Marketing (Theory anchors — verify each before submitting):** Timoshenko & Hauser
  (2019, *Marketing Science*) · Salminen et al. (2025, *J. Marketing Analytics*) ·
  Berger, Moe & Schweidel (2023) — *optional, RQ3 engagement only*.
- **Stats / packages:** Benjamini & Hochberg (1995) · Dietterich (1998, McNemar) ·
  Friedman, Hastie & Tibshirani (2010, glmnet) · Bates et al. (2015, lme4) · Kuznetsova
  et al. (2017, lmerTest) · Kerby (2014, rank-biserial) · Efron & Tibshirani (1993).
- **Decide before citing:** *Grice (1975)* — optional pragmatic anchor for §2.1.
  *Mohammad & Turney (2013, NRC)* — in the group's feature table but NRC incongruity was
  **not implemented**; cite for the §2.3 theoretical claim only, or drop — never imply
  it was measured. **Drop** Rajadesingan, Barbieri, Reyes, Saljoughian, Humphreys & Wang
  (not in the course sources). Read-or-drop the unlabelled **Emerald** sarcasm-detection
  link in the notes. Verify the **`microsoft/harrier-oss-v1-270m`** model page resolves.
  One hallucinated/unread source = automatic fail.

---

## Open items to settle before drafting prose
1. **harrier-oss-v1-270m** — confirm the HuggingFace ID resolves.
2. **Team size >4** — confirm the chair permits 5 authors.
3. **Abstract** — include or not (not in the chair's spec).
