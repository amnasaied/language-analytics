# Term Paper Template, Style Guide, and Checklists

Everything the chair has stated about the term paper, assembled into a single
working document: the binding format spec, a section-by-section skeleton mapped
onto this project, the writing rules from Session 11, and the checklists to run
before submission.

**Sources.** Format, sections, rubric, and zero-tolerance rules: `01_Lectures/Session2.pdf`
slides 5–8. Writing style and structure: `01_Lectures/Session11_Course wrap up.pdf`
slides 13–19 (*Scientific Writing for Psychology*; the book plus an AI summary is
on Moodle). Assessment weighting and deadline: `01_Lectures/Session1.pdf` slides 11–12.

**Deadline:** Friday, 28.08.2026, 23:59. Reflection and peer evaluation on
Moodle within 2 weeks after.

> This encodes every criterion the chair has published — it is the checklist she
> grades against, not a guarantee of a grade. What it cannot supply is the
> substance: a defensible research question, honest validity checks, and real
> engagement with the literature.

---

## 1. Hard format spec — non-negotiable

| Item | Requirement |
|---|---|
| Font | Times New Roman, 12 pt |
| Line spacing | 2.0 (double) |
| Margins | 1 inch all sides |
| Headings | **Bold, numbered** (1., 1.1, 1.2, …) |
| Citations | APA 7th |
| Figures & tables | Numbered, captioned |
| Length | **15 pages maximum**, excluding title page and references (Session 1 frames the target as 10–15) |
| Team | 3–4 students |

**Submit three things:**
1. The term paper.
2. The code for scraping (if applicable), cleaning, and analysis — or add the
   lecturer to the GitHub project instead.
3. An export from the reference manager (e.g. Zotero) covering every cited paper.

**Weighting context:** the term paper is 50 of 90 total course points. Reflection
and peer evaluation add 20, live quizzes 20.

---

## 2. Section skeleton with page budget

The chair's prescribed structure, with the per-section budget and what belongs in
each for *this* project. Budgets are at 2.0 spacing, so a "page" is roughly
250–300 words.

### 1. Introduction — 1 to 1.5 pages (~350–450 words)

**Purpose:** establish background. **Must contain:** a marketing-relevant,
researchable question, and the scope of the problem.

Three paragraphs, following her Chapter 5 framing:

- **¶1 — The punch.** Open with an example, people and their behaviour, a
  rhetorical question, a striking statistic, or an anecdote. Not a definition,
  not "In recent years…". For this project: a concrete sarcastic product comment
  whose surface sentiment inverts its meaning, and the brand-monitoring dashboard
  that would score it positive.
- **¶2 — The rationale.** Why this matters for marketing, documented with
  findings. Sentiment tooling drives brand tracking and product feedback loops;
  sarcasm systematically corrupts it. Two citations per argumentative step,
  no more.
- **¶3 — Purpose and predictions.** Restate the purpose, give 2–3 sentences on
  the logic of the method with no details, close with one sentence predicting the
  pattern of results. Past tense.

### 2. Theory — 2.5 to 3 pages (~700–900 words)

**Purpose:** define the constructs, show what prior research found, name the gap.

- **2.1 Sarcasm as a construct in user-generated content.** Define it — an
  incongruity between literal and intended meaning — and separate it from irony
  and negation. This definition is what your operationalization must later match.
- **2.2 Why sarcasm matters for marketing analytics.** Sentiment misclassification,
  brand monitoring, review helpfulness, engagement.
- **2.3 What prior work has found.** Anchor on the course papers where they fit —
  Salminen et al. (2025) is the direct methodological precedent for combining
  psycholinguistic features with transformer representations, and Timoshenko &
  Hauser (2019) for extracting marketing-relevant content from UGC. Foreground
  findings, background the authors.
- **2.4 Research gap and questions.** State RQ1–RQ3 explicitly here, each in one
  sentence, each connected to the gap you just named.

**Trap:** this section is graded on *critical engagement* (10%), not coverage.
Contrasting two findings that disagree is worth more than listing six that agree.

### 3. Methods — 3 to 4 pages (~850–1,200 words)

**Purpose:** data sourcing and the analytical pipeline. **Describe how you used
each method, not what the method is.** No results here.

- **3.1 Data.** The Self-Annotated Reddit Corpus (SARC), balanced export —
  cite Khodak, Saunshi, and Vodrahalli (2018) *and* the Hugging Face
  distribution. State plainly that it is a public dataset you downloaded; never
  imply you scraped it. Report the marketing-subreddit filter (64 names, six
  thematic groups), the resulting **51,337 comments across 62 subreddits**,
  label balance, and the 2009-09 to 2016-12 span. Justify the filter: why these
  communities are marketing-relevant.
- **3.2 Label validity.** The `/s` convention is author self-annotation, not
  independent coding. Say so here, and report your manual check (§5 below).
- **3.3 Preprocessing.** The cleaning and preprocessing pipeline, each decision
  justified — this is a full 10% of the grade on its own. Cover domain-specific
  noise: URLs, markdown, deleted comments, emoji.
- **3.4 Feature operationalization.** One table running the chain from Session 11
  slide 19: **construct → operationalization → method → expected pattern**, one
  row per feature family. Justify each measure and cite the source that used or
  validated it.
- **3.5 Analysis.** Per RQ: the model, the tuning procedure, the evaluation
  metric, the train/test split, the seed. Parameters tuned by data-driven
  approaches is explicitly named in the rubric — say how you tuned, not just that
  you did.

### 4. Results — 3 to 4 pages (~850–1,200 words)

**Purpose:** findings and visual insight. **Write this section first.** No method
description whatsoever — that belongs in §3.

Structure it as a "photo essay": choose a small number of figures that carry the
essential findings and walk the reader through each one. You have 36 plots in
`outputs.docx`; four to six belong in the paper, the rest in an appendix or
nowhere.

- **4.1 (RQ1)** Which linguistic features distinguish sarcastic comments.
- **4.2 (RQ2)** Whether psycholinguistic features add over embeddings alone.
- **4.3 (RQ3)** Whether sarcastic comments earn higher scores.

Each subsection: state the finding as behaviour, attach the statistic as
documentation, link explicitly to the question it answers.

> Bad: "The main effect of sarcasm on score was significant, β = …"
> Good: "As predicted, sarcastic comments earned more upvotes than literal ones,
> β = …"

Do not narrate a table in prose. Report robustness and ancillary checks in a
sentence each, with detail in the appendix.

### 5. Discussion — 2 to 3 pages (~550–850 words)

**Purpose:** interpret and give marketing impact. Her prescribed order:

1. **Orienting paragraph**, 1–3 sentences summarizing the key findings.
2. **Limiting conditions, early.** Sample, measures, design. Brief — "not a
   confession of every blemish". Getting them out of the way early lets the
   discussion build momentum. For this project the honest four are: self-annotated
   labels, one platform, 2009–2016 vintage, and a subreddit filter that proxies
   marketing relevance rather than measuring it.
3. **One to three study-specific issues** connecting the findings back to the
   theory in §2. Unexpected but provocative results belong here.
4. **Implications** — practical and managerial first, then optionally the broader
   ones. Be humble; the rubric says groundbreaking insight is not expected.
5. **End strong.** Return to the hook from §1, updated by what you found, and
   state the take-home message.

**No separate "future research" section.** Put a specific follow-up study inside
the paragraph discussing the question it would answer. Never end with "more
research is needed".

### 6. References — not counted

APA 7th. Every entry verified against the real paper.

---

## 3. Writing style rules

### Sentences

- Replace nominalizations with verbs and adjectives: "the overestimation of…" →
  "xy is overestimated".
- Subject, verb, object — get to the subject fast. Avoid long introductory clauses.
  Bad: "Some consumers, due to prior negative experiences, are skeptical of…"
- Tell a story about people: actors are subjects, actions are verbs.
  Bad: "Greater emotionality of reviews is observed following service failures."
  Good: "Consumers write more emotional reviews after service failures."
- Turn negatives into affirmatives: "not different" → "similar".
- Delete what readers can infer.
- Replace phrases with single words: "due to the fact that" → "because".

### Paragraphs

- One paragraph, one idea. New idea, new paragraph.
- 4–8 sentences.
- Open with a topic sentence carrying the main idea.
- Develop it over 3–6 sentences by (a) supporting evidence, (b) contrasting
  alternatives — "on the one hand… on the other…", or (c) describing a process —
  "first… then… finally".
- Optionally close with "thus" or "in summary".
- Flow: start each sentence with familiar information, then introduce the new.
- Hold one viewpoint and one abstraction level throughout. Do not mix people
  ("consumers complain more when…") with abstractions ("complaint frequency is
  associated with…") inside one paragraph.

### Citations

- Findings in the foreground, studies and authors in the background.
  Good: "Negative reviews hurt sales more than positive reviews help them
  (Smith, 2020; Lee, 2021)."
  Bad: "Smith (2020) studied 500 reviews and found… Lee (2021) analyzed…"
- Describing a study explicitly is the exception, not the norm.
- **Two citations per argumentative step is adequate.** Do not cite all or most
  relevant studies.

### Discussion — the six mistakes to avoid

Too long (discussing every result) · poorly organized (stream of consciousness) ·
summarizing without discussing · going off topic · wrong tone (over-hedged or
arrogant) · self-praise ("we are the first to…", "this important finding…" —
make that case in the Introduction).

---

## 4. Rubric map — where each point is earned

| Criterion | Weight | Earned in |
|---|---|---|
| Clarity of research question(s) and marketing relevance | 10% | §1 ¶2–3, §2.4 |
| Theoretical foundation and use of literature | 10% | §2.1–2.3 |
| Data quality and novelty | 10% | §3.1 |
| Data cleaning and preprocessing | 10% | §3.3 |
| **Methodology** | **20%** | §3.4–3.5 — correct application of ≥1 course method, aligned to the RQ, with data-driven parameter tuning |
| Visual communication of insights | 10% | §4 figures |
| **Discussion** | **15%** | §5 — findings in context of literature, plus critical evaluation of data, validity, and bias |
| Expected contributions | 5% | §5 implications |
| Structure, style, and writing | 10% | throughout, **plus well-documented, readable code with piped workflows** |

Two things to notice. Methodology and Discussion together are 35% — more than
the introduction, theory, and data sections combined. And code quality is graded
inside the writing criterion, so the scripts need cleaning before they are handed
over, not just the prose.

---

## 5. Project-specific obligations

Items the rubric demands that this project does not yet satisfy. Each one is a
paragraph you owe the reader.

- **Manual validity check on automated annotations.** Session 11 slide 19
  requires it: sample ~100 comments, hand-code them, and report agreement with
  the `/s` labels and with the VADER/sentimentr scores. Report the numbers in
  §3.2. Without this, the validity criterion inside Discussion (15%) has nothing
  to stand on.
- **Data-driven parameter tuning, stated explicitly.** The rubric names it. Say
  which parameters were tuned, over what grid, by what resampling scheme.
- **Bias and limitations.** Session 10 covers LLM limitations and biases; the
  Discussion criterion asks for "potential biases". Self-selected sarcasm markers,
  subreddit demographics, and the anisotropy of raw embedding spaces are the
  three live ones here.
- **Dataset honesty.** SARC is public and downloaded. Say exactly that. Claiming
  otherwise is a zero-tolerance fail.
- **One numbering scheme.** `rq3_score_analysis.R` currently carries an "RQ2"
  header and `outputs.docx` orders sections RQ1 → RQ3 → RQ2. Settle the numbering
  before writing, and make the paper, scripts, and figures agree.

---

## 6. Zero tolerance — automatic fail

- One hallucinated source in the references.
- Fake or AI-generated data.
- Lying about the data source (claiming you scraped what you downloaded).
- Plagiarizing someone else's work or analysis.
- Being unable to send the preprocessing, analysis, or scraping code.

**Mitigation:** before submitting, open every reference and confirm the title,
authors, year, venue, and DOI resolve to a real paper. Build the bibliography in
Zotero from DOIs rather than by hand — the export is a submission requirement
anyway.

---

## 7. Pre-submission checklist

**Format**
- [ ] Times New Roman 12pt, 2.0 spacing, 1-inch margins throughout
- [ ] Headings bold and numbered
- [ ] ≤15 pages excluding title page and references
- [ ] Every figure and table numbered, captioned, and referred to in the text
- [ ] APA 7th in text and in the reference list

**Content**
- [ ] Results section was written first
- [ ] No method description in §4; no results in §3
- [ ] Every result linked explicitly to the question it answers
- [ ] Operationalization chain table present and each measure justified
- [ ] Manual validity check reported with numbers
- [ ] Limitations appear early in the Discussion, not at the end
- [ ] No separate "future research" section
- [ ] Discussion ends by returning to the opening hook
- [ ] No self-praise anywhere in §5

**Integrity and submission**
- [ ] Every reference opened and verified against the real paper
- [ ] Zotero library exported
- [ ] Dataset described as public and downloaded, cited correctly
- [ ] Code cleaned, commented, piped, and pushed — lecturer added to the repo
- [ ] Exam registered in TUMonline (deregistration closed 21.08.2026)
- [ ] Submitted by Friday 28.08.2026, 23:59
- [ ] Reflection and peer evaluation on Moodle within 2 weeks after