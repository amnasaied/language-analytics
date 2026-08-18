# Plan — Section 2 (Theory)

Blueprint for the term paper's theoretical framework: the organising argument, the
subsection structure with word budgets, which source goes where, and the risks that
would cost points. Companion to [PAPER_TEMPLATE.md](PAPER_TEMPLATE.md) (the format and
rubric spec) and [final-project-sarcasm-results.md](final-project-sarcasm-results.md)
(the findings the theory has to set up).

**Budget:** 2.5–3 pages at 2.0 spacing = **700–850 words**, ~8 paragraphs.
**Rubric weight:** 10% directly ("Theoretical foundation and use of literature"), but
it also carries the 10% RQ-clarity criterion (§2.5) and gates the 15% Discussion
criterion, which is graded on *reconciling findings with the theoretical framework*.

---

## 1. The two rules that should drive every choice below

Both come from what the chair actually said, not from generic advice.

**Rule 1 — Synthesise, do not list.** Her written critique of Salminen et al. (2025),
Session 10 slide 6: *"In the theoretical background, results of previous studies are
simply listed, instead of summarized and synthesized to see the bigger picture."* The
rubric's best outcome is "critical engagement", and Session 11 slide 15 caps it at
**two citations per argumentative step**. A theory section that surveys ten sarcasm-
detection papers scores worse than one that stages a disagreement between two camps
and resolves it with our data.

**Rule 2 — Theory must precede features.** Her sharpest criticism of the same paper,
Session 10 slide 5: *"their test was not theory-driven at all. They just used all 88
categories and checked what sticks."* Our feature families were chosen a priori from
the literature — the group's `Research Questions + Notes` table predates the analysis
and cites a source per feature. §2.3 has to make that visible, so that §3.4's
operationalisation table reads as a derivation rather than a fishing expedition.

---

## 2. The organising argument (the spine)

One thesis carries all three RQs:

> Automated text analysis reads user-generated content by assuming that measured
> surface features track intended meaning. Sarcasm is the systematic violation of that
> assumption. So sarcasm is not merely a classification problem — it is a
> **construct-validity threat to the instrument marketing analytics runs on**, and how
> much it costs depends on whether sarcasm leaves recoverable surface traces.

This spine does three jobs at once:

1. It makes the paper *marketing* research rather than NLP research, which is what the
   RQ-relevance criterion (10%) is checking.
2. It turns RQ1's weak result into a **theory test**, not a failed model. If sarcasm
   were surface-recoverable, lexical features would work; they barely do (AUC 0.562),
   which is evidence *for* the context-dependence account and directly motivates RQ2.
3. It gives the Discussion something to reconcile against, which is where 15% lives.

---

## 3. Subsection structure

### 2.1 Sarcasm as a construct in user-generated content — ~120 words, 1 paragraph

Define sarcasm as an **incongruity between the literal and the intended meaning of an
utterance**, produced with critical or mocking intent. Two boundary moves, one sentence
each: separate it from **negation** (which inverts meaning explicitly and is
lexically visible) and from **irony** broadly (situational, not necessarily
speaker-directed). One sentence on why the construct is hard: sarcasm is defined by
*speaker intent*, which no text feature observes directly — the point §3.2's label-
validity discussion will cash in.

- Sources: Grice (1975) on flouting the maxim of quality — the canonical pragmatic
  definition; Riloff et al. (2013) for the contrast-based operational definition.
- **This definition is a contract.** Everything §3.4 measures must be defensible as a
  measure of *this* construct. Do not define sarcasm as intent and then measure
  exclamation marks without a bridging argument (§2.3 supplies it).

### 2.2 Sarcasm as a validity threat to marketing text analytics — ~180 words, 2 paragraphs

**¶1 — The instrument.** Marketing research reads UGC at scale to extract customer
needs, track brand perception, and explain engagement. Foreground the findings, not the
authors: machine-filtered UGC surfaces customer needs as efficiently as focus groups
(Timoshenko & Hauser, 2019); linguistic features of a post predict how long audiences
attend to it (Berger et al., 2023). Both rest on the literal-meaning assumption.

**¶2 — The breach.** Where sarcasm is common, dictionary- and sentiment-based measures
invert. A sarcastic complaint scored positive is not noise that averages out — it is
**systematic**, biased toward exactly the moments (product failures, launches, service
breakdowns) that brand monitoring exists to catch. Name the scope: sarcasm's true base
rate on Reddit is ~0.25% overall (Khodak et al., 2018), but it concentrates in
consumer-product communities, so the exposure is uneven across the very categories
marketers monitor.

- Sources: Timoshenko & Hauser (2019); Berger et al. (2023); Humphreys & Wang (2018)
  for the automated-text-analysis method taxonomy that our approach sits inside;
  Khodak et al. (2018) for base rates.
- Course-paper coverage lands here — the rubric says "if applicable: also use course
  papers", and this is where three of the five fit naturally.

### 2.3 How sarcasm becomes detectable: markers, incongruity, and context — ~300 words, 3 paragraphs

**The heart of the section.** Structure it as Session 11 slide 14's option (B),
contrasting alternatives, not as a survey. Three accounts, each with a *testable*
implication for our feature set:

| Account | Claim | Implies |
|---|---|---|
| **Pragmatic marker** | Speakers must signal ironic intent for hearers to recover it, so they add conventional markers: interjections, intensifiers, capitalisation, heavy punctuation, emoticons | Typographic and emphasis features carry the signal |
| **Context incongruity** | Sarcasm is a positive expression about a negative situation; the clash itself is the signal | Within-comment polarity contrast and comment-vs-parent sentiment distance carry the signal |
| **Context dependence** | Neither surface family suffices; recovering intent needs conversational and world context, which only distributed representations encode | Transformer embeddings dominate hand-crafted features |

**¶1 — Marker account.** Interjections and punctuation predict readers' sarcasm
judgements (Kreuz & Caucci, 2007), and emoticons, laughter tokens, and heavy
punctuation are the cues human annotators rely on most (González-Ibáñez et al., 2011).

**¶2 — Incongruity account, and the tension.** A positive-sentiment predicate attached
to a negative situation is the dominant sarcastic form (Riloff et al., 2013), and
semantic discordance detects sarcasm even where no sentiment word appears (Joshi et
al., 2016). **State the disagreement explicitly:** the marker account predicts sarcasm
is written to be *obvious*, the incongruity account predicts it is written to be
*resolved*. They imply different feature sets and different failure modes.

**¶3 — Context account and the resolution to be tested.** Human annotators need
conversational context to infer ironic intent (Wallace et al., 2014), and self-annotated
sarcasm resists strong context-free baselines — bag-of-bigrams reaching 75.8% against a
92.0% human ceiling, with summed word embeddings *below* n-grams (Khodak et al., 2018).
The parallel case in marketing is decisive: psycholinguistic dictionaries detect
deceptive reviews at 67.5% accuracy while a fine-tuned transformer reaches 82.6%
(Salminen et al., 2025). Close the paragraph by naming the open question that RQ2
answers — whether the two feature families are **substitutes or complements**. Salminen
et al. ran them as separate experiments and never combined them; that unexamined
seam is our contribution.

- **This paragraph is where the 10% is won.** It contrasts, it foregrounds findings, it
  names an unresolved question, and it hands RQ2 a genuine motivation.

### 2.4 Sarcasm and engagement — ~140 words, 1–2 paragraphs

RQ3 needs its own theoretical parent; do not smuggle it in as an afterthought.

Two opposed predictions, stated as such:
- **Uplift.** Sarcasm is humour and in-group signalling; language that rewards the
  reader with an interpretive payoff holds attention, and linguistic features
  measurably drive engagement (Berger et al., 2023).
- **No uplift.** Sarcasm imposes a processing cost and fails without shared context
  (Wallace et al., 2014), so it should reward only readers who already share the
  community's frame — predicting a **community-level** effect rather than a
  comment-level one. Comment score was not an informative sarcasm feature in the source
  corpus (Khodak et al., 2018).

Close with the resolution our data will test: if the effect is norm-driven rather than
message-driven, it should attenuate once community is controlled. Saljoughian et al.
(2025) supplies the mechanism — conversational norms are community-level properties
that firms steer, not attributes of individual messages.

- **This paragraph is doing real work:** it predicts the actual RQ3 result (β drops to
  non-significance once subreddit is controlled) *from theory, in advance*. That
  converts a null finding into a confirmed prediction.

### 2.5 Research gap and research questions — ~110 words

**¶ — The gap, in three sentences.** (1) Sarcasm detection is a computer-science
literature optimised for accuracy on general-purpose corpora. (2) Marketing text
analytics assumes literal meaning and has not measured what figurative language costs
its instruments. (3) No study has tested whether the psycholinguistic feature families
marketing analytics relies on add anything **on top of** contextual representations, or
whether sarcasm carries an engagement premium in the consumer communities where brands
actually listen.

Then the three RQs, one sentence each, each visibly answering the gap:

- **RQ1.** Which linguistic features distinguish sarcastic from non-sarcastic comments
  in marketing-relevant communities?
- **RQ2.** Do psycholinguistic features improve sarcasm detection over semantic
  embeddings alone?
- **RQ3.** Do sarcastic comments receive higher public approval than non-sarcastic ones?

**Attach one directional expectation per RQ** (one clause, not a paragraph). Session 11
slide 15 asks the Introduction to close with a predicted pattern of results, and the
rubric asks for research questions — a stated expectation per RQ satisfies both without
committing to formal hypothesis testing the design does not support.

---

## 4. Source budget and placement

Target **14–16 distinct sources** in §2. More than that and the two-citations-per-step
rule is broken by definition.

**Course papers — verified in the lecture slides, cite with confidence**

| Source | Where | Why it earns its place |
|---|---|---|
| Timoshenko & Hauser (2019), *Marketing Science* 38(1), 1–20 | 2.2 ¶1 | UGC as a marketing instrument |
| Berger, Moe & Schweidel (2023), *Journal of Marketing* 87(5), 793–809 | 2.2 ¶1, 2.4 | Linguistic features drive engagement — RQ3's parent |
| Salminen et al. (2025), *Journal of Marketing Analytics* | 2.3 ¶3 | Direct precedent: psycholinguistics vs transformers |
| Saljoughian et al. (2025), *JMR* 62(6), 1003–1025 | 2.4 | Community-level conversational norms |
| Humphreys & Wang (2018), *JCR* 44(6), 1274–1306 | 2.2 ¶2 | Method taxonomy + the validity standard |

Humphreys, Isaac & Wang (2021) is the one course paper with no natural home here —
force it in and it shows. Leave it out of §2 rather than padding.

**Sarcasm literature — already sourced in the group's notes with working links**

Khodak, Saunshi & Vodrahalli (2018) · Riloff et al. (2013) · Joshi et al. (2016) ·
González-Ibáñez et al. (2011) · Kreuz & Caucci (2007) · Wallace et al. (2014,
`aclanthology.org/W14-2608`) · Hutto & Gilbert (2014) · Grice (1975)

Wallace et al. (2014) is currently an unlabelled link at the bottom of the notes table.
It is the strongest citation available for §2.3 ¶3 and §2.4 — promote it.

**Verify before citing** (open the PDF, confirm authors/year/venue/DOI — one
hallucinated source is an automatic fail)

- Attardo's work on irony markers, if the marker account needs a theoretical rather
  than empirical anchor.
- Sperber & Wilson's echoic-mention account, if §2.1 needs a second definitional pole.
- The Emerald *Information Discovery and Delivery* 52(2) sarcasm-detection paper linked
  in the group's notes — unread and unlabelled; either read it or drop it.
- Tausczik & Pennebaker (2010) — only if §3.4 justifies the LIWC-substitute decision by
  reference to what LIWC categories measure.

---

## 5. Risks that would cost points

**Promising features the Methods section does not deliver.** Per
[PLAN_ALIGNMENT.md](PLAN_ALIGNMENT.md) §4, three of the six literature-sourced features
were never implemented: Joshi's within-comment word-pair discordance, NRC emotional
incongruity, and González-Ibáñez's emoticon/laughter tokens. §2.3 may **motivate the
account** those papers support, but must not imply we measured what they measured. Two
clean exits: implement the emoticon and laughter counts (a two-line regex, cheap, and
it strengthens the marker account), or cite those papers for the *theoretical claim*
only and let §3.4 state the operationalisation actually used.

**The incongruity mismatch between RQ1 and RQ2.** RQ1 measures within-comment polarity
co-presence via VADER; RQ2 measures sentence-level sentiment SD via sentimentr plus a
comment-vs-parent cosine. If §2.3 presents incongruity as one construct, §3.4 must show
two operationalisations of it and say why — otherwise the RQ2 comparison reads as
testing a different thing than RQ1 found.

**Defining sarcasm by intent, then never addressing whether the labels capture intent.**
§2.1 sets up the problem; §3.2 must pay it off with the `/s` self-annotation discussion
and the manual validity check. Theory that raises a validity question the Methods
ignores is worse than theory that never raised it.

**Writing the theory as three disconnected mini-reviews, one per RQ.** The spine in §2
above is what prevents this. Every subsection should be traceable to the one thesis:
sarcasm breaks the literal-meaning assumption, and the cost depends on what survives at
the surface.

---

## 6. Drafting order

1. Write **2.3** first — it is the argumentative core and the hardest to get right;
   everything else calibrates to it.
2. Then **2.5**, so the gap and RQs are fixed before the surrounding prose is tuned.
3. Then **2.2** and **2.4**, which are framing and can be sized to whatever budget
   remains.
4. Write **2.1** last and keep it short. Definitions are the easiest place to overspend.
5. Re-read against §3.4's operationalisation table and cut any construct §3.4 does not
   measure.
