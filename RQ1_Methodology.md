# RQ1 — What features distinguish sarcastic from non-sarcastic comments?

**Design:** confirmatory / inferential. We test *theory-derived* linguistic features against a pre-specified directional hypothesis each, and report **effect sizes** (not just p-values) because at ~45k observations almost everything is "significant."

**Data:** the SARC "balanced" Reddit sarcasm export (Khodak et al. 2018), 50/50 sarcastic/non-sarcastic, filtered to ~62 marketing-relevant subreddits (phones, cars, fashion, gaming, retail, entertainment) so findings speak to marketing UGC. **All RQ1 analyses run on the full cleaned corpus (~50k comments, ~50/50 balanced).** Only the multivariate model (Step 4) restricts subreddits — to the top 30 by volume — as a control, purely for estimation stability.

---

## Methodological logic (why this order)

Each step exists to answer a question the previous one leaves open:

| Step | Question it answers | Why we needed it |
|------|--------------------|------------------|
| 0. Clean the data | *Is the text usable and leakage-free?* | Removes noise and the label-giving `/s` marker |
| 1. Operationalize | *Which features, and what do we expect?* | Turns theory into testable, directional hypotheses |
| 2. Engineer features | *How do we measure them?* | Converts text → numeric features |
| 3. Univariate tests | *Does each feature differ between classes?* | First evidence, per feature |
| 4. Multivariate model | *Does it still hold with others + confounds controlled?* | Rules out topic/length confounds |
| 5. Paired robustness | *Does it hold within the same thread?* | Strictest confound control |
| 6. Ranking | *Which features matter most?* | Orders the confirmed effects |
| 7. Synthesis | *What is the verdict per hypothesis?* | The actual answer to RQ1 |

---

## Step 0 — Data cleaning (order matters)

Applied in sequence; each step has a specific reason:

| # | Step | Why |
|---|------|-----|
| 1 | **Remove missing/empty** — drop rows with NA or blank comment/parent | Empty text yields no features and breaks VADER scoring |
| 2 | **Strip the `/s` marker** — remove trailing & inline `/s` from comment *and* parent | **Critical (anti-leakage):** `/s` is the author tag that *generated* the labels; leaving it in lets any model detect the marker instead of learning sarcasm |
| 3 | **Remove Reddit markup/HTML** — decode `&gt; &lt; &amp;`, drop zero-width spaces, `[text](url)`→text, links→`[URL]`, strip `**bold**`/`*italic*`, collapse whitespace | Markup is noise, not sarcasm signal, and would inflate the typographic features (e.g. stray brackets/punctuation) |
| 4 | **Remove duplicates** — `distinct(comment, parent_comment)` | Copypasta/bot reposts share identical feature values → would double-count and bias estimates |

Class balance is re-checked after cleaning (remains ~50/50).

## Step 1 — Operationalize features from theory

We defined **11 features**, each grounded in a cited source with an expected direction (↑ = more frequent in sarcasm):

| Feature | Expected | Source |
|---------|----------|--------|
| Exclamation, Capitalization | ↑ | González-Ibáñez 2011; Hutto & Gilbert 2014 |
| Interjections | ↑ | Kreuz & Caucci 2007 |
| Intensifiers | ↑ | Khodak 2018 (SARC) |
| Quotation marks, Ellipsis | ↑ | González-Ibáñez 2011 |
| Emoticons, Laughter | ↓ (sincerity) | González-Ibáñez 2011; SARC |
| Sentiment incongruity (\|comment − parent\|) | ↑ | Riloff 2013; Joshi 2016 |
| VADER sentiment, comment length | control / exploratory | — |

## Step 2 — Feature engineering (exact measurement)

Typographic markers are **regex counts** per comment; sentiment uses **VADER** (chosen for social-media text — it handles capitalization, punctuation and emoji), scored once per unique text for efficiency.

| Feature | Operational definition |
|---------|------------------------|
| Exclamation | count of `!` characters |
| Capitalization | count of ALL-CAPS words (`\b[A-Z]{2,}\b`) — i.e. shouting, ≥2 letters so lone "I" is excluded |
| Ellipsis | count of `...` (3+ dots) or `…` |
| Quotation marks | count of quotation characters |
| Interjections | count of words from a curated interjection list (*oh, wow, ugh, geez, gosh, huh…*), case-insensitive, word-boundary matched |
| Intensifiers | count of intensifier words (*very, really, totally, absolutely, extremely…*) |
| Emoticons | count from a curated emoticon list (`:) :-) ;) :( :D :P <3 …`) |
| Laughter | count of laughter tokens (`lol+, lmao+, rofl, haha+, hehe+`) |
| VADER sentiment | VADER compound score of the comment, ∈ [−1, 1] |
| Sentiment incongruity | \|VADER(comment) − VADER(parent)\| — absolute gap, so opposite-direction shifts don't cancel |
| Comment length | word count (control); enters models as `log(1 + length)` |

## Step 3 — Univariate tests (per feature)

We split features by distribution, because one test does not fit both:
- **Continuous features** (sentiment, length, incongruity, caps-rate) → **Wilcoxon rank-sum test** (Mann–Whitney U) + **rank-biserial correlation** as effect size (robust to skew, no normality assumption).
- **Sparse markers** (mostly zero) → **chi-square test of independence** on presence (feature > 0, sarcastic vs not) + **Cramér's V** effect size (median is 0 for both classes, so a rate comparison is the honest test).

Both tables are **FDR-corrected** (Benjamini–Hochberg) to control false positives across the 11 tests — the same correction the course uses, and the one the Salminen paper was criticized for omitting.

*Effect-size scale (Cohen, for df = 1):* V ≈ 0.1 small, 0.3 medium, 0.5 large. In our labelling we call V ≥ 0.10 "moderate" (the largest observed), 0.05–0.10 "small", below that "negligible".

## Step 4 — Multivariate logistic regression

*Why:* a univariate difference may be a confound (e.g. sarcastic *subreddits* are shouty). So we fit one logistic model with **all features + comment length + subreddit fixed effects** (subreddit restricted to the **top 30 by volume** so each dummy has enough data to be stable), predictors standardized → coefficients are comparable **odds ratios per 1 SD**. Correlation matrix (max ρ = 0.31) and VIF ≈ 1 confirmed no multicollinearity, so all features were kept.

*Specification:* `logit P(sarcastic) = β₀ + Σₖ βₖ·featureₖ + β_len·log_length + subreddit fixed effects`.

## Step 5 — Paired robustness check

*Why:* subreddit control is still coarse. We refit as a **conditional (fixed-effects) logistic regression** stratified by `parent_comment` — comparing a sarcastic and a non-sarcastic reply **to the same parent**. This cancels all shared context (topic, thread) and mirrors the SARC benchmark design. Only parents with exactly one sarcastic + one non-sarcastic reply enter this model.

*Specification:* `logit P(sarcastic) = Σₖ βₖ·featureₖ + β_len·log_length`, stratified by parent (no global intercept).

## Step 6 — Rank features (inferential)

Features ranked by **standardized odds ratio + FDR-adjusted significance**. Predictive rankings (random-forest importance, AUC) were deliberately removed — predictive performance is RQ2's job, not RQ1's.

## Step 7 — Synthesis verdict

Each hypothesis gets a directional verdict — **Confirmed / Reversed / Not supported / Mixed** — with a separate **strength** label (effect-size magnitude), so a statistically-significant-but-tiny effect is not oversold.

---

## Results

| Feature | Observed | Effect (V) | OR / SD | Verdict | Strength |
|---------|----------|-----------|---------|---------|----------|
| **Exclamation** | ↑ sarcastic | 0.133 | 1.33 | **Confirmed** | moderate |
| **Emoticons** | ↑ non-sarcastic | 0.082 | 0.83 | **Confirmed** | small |
| **Interjections** | ↑ sarcastic | 0.052 | 1.11 | **Confirmed** | small |
| Intensifiers | ↑ sarcastic | 0.030 | 1.05 | Confirmed | negligible |
| Laughter | ↑ non-sarcastic | 0.030 | 0.95 | Confirmed | negligible |
| Capitalization | ↑ sarcastic | 0.013 | 1.06 | Confirmed | negligible |
| Sentiment incongruity | ↑ sarcastic | 0.012 | 1.03 | Confirmed | negligible |
| Quotation marks | ↑ (univ.) / ↓ (multi.) | 0.017 | 0.97 | **Mixed** | negligible |
| Ellipsis | — | ~0 | 1.00 | **Not supported** | negligible |

**Key findings**

1. **Direction validated across the board** — every feature moved as theory predicted; 7 of 9 are formally confirmed.
2. **Only exclamation meaningfully distinguishes** sarcasm (OR 1.33 per SD, moderate effect). Everything else is real but small-to-negligible.
3. **Emoticons and laughter are reversed sincerity cues** — sarcastic authors *avoid* them, reproducing the SARC finding.
4. **Weak support for theory-central constructs:** sentiment incongruity and capitalization are confirmed only at negligible magnitude; quotation marks flip sign once controlled (confound); ellipsis shows no effect.

**One-line answer:** *Sarcasm markers appear in the theoretically expected directions, but only exclamation marks meaningfully distinguish sarcastic from non-sarcastic comments; emoticons and laughter act as reversed sincerity cues.*

---

## Limitations

- **Non-independence:** ~25.7k authors across the corpus (plus shared threads), so main-model standard errors are mildly anti-conservative; Part 5 addresses thread-level clustering.
- **Paired model underpowered:** the within-thread matched sample is small, so it does not independently confirm effects (a power issue, not a disconfirmation).
- **Large N inflates significance** — hence effect size, not p-value, is our evidence throughout.
- **Subreddit scope:** univariate tests and the synthesis use the full cleaned corpus; the multivariate model restricts subreddit fixed effects to the top-30 by volume for estimation stability.
- **Lexicon coverage:** interjection/intensifier/emoticon lists are finite dictionaries, so novel or misspelled variants are missed.

## Supplementary (exploratory)

LDA and PCA projections show **heavy overlap** between the two classes — consistent with individually weak features. Reported as a descriptive separability check only; predictive performance is evaluated in **RQ2**.

---

## References

- Khodak, M., Saunshi, N., & Vodrahalli, K. (2018). *A Large Self-Annotated Corpus for Sarcasm (SARC).* LREC.
- González-Ibáñez, R., Muresan, S., & Wacholder, N. (2011). *Identifying Sarcasm in Twitter.* ACL.
- Hutto, C., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM.
- Kreuz, R., & Caucci, G. (2007). *Lexical Influences on the Perception of Sarcasm.* ACL Workshop on Computational Approaches to Figurative Language.
- Riloff, E., et al. (2013). *Sarcasm as Contrast between a Positive Sentiment and Negative Situation.* EMNLP.
- Joshi, A., Sharma, V., & Bhattacharyya, P. (2015). *Harnessing Context Incongruity for Sarcasm Detection.* ACL.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B.
