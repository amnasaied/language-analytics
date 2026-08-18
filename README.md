# Sarcasm in Marketing-Relevant Reddit Comments

Final exam project for *Language Analytics and LLMs in Marketing* (TUM).
An R analysis of **sarcasm in user-generated content** on Reddit, restricted to
marketing-relevant communities (tech, gaming, cars, fashion, retail,
entertainment).

## Research questions

| | Question | Script |
|---|---|---|
| **RQ1** | Which linguistic features distinguish sarcastic from non-sarcastic comments? | [rq1.R](rq1.R) |
| **RQ2** | Do embeddings + psycholinguistic features beat embeddings alone? | [rq2_embeddings.R](rq2_embeddings.R) |
| **RQ3** | Do sarcastic comments receive higher Reddit scores? | [rq3_score_analysis.R](rq3_score_analysis.R) |

[Final Project.R](Final%20Project.R) is the scratch script that downloads the
dataset and applies the subreddit filter — superseded by the three RQ scripts.

## Data

[`marcbishara/sarcasm-on-reddit`](https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit)
on Hugging Face (the SARC / Kaggle "Sarcasm on Reddit" balanced export,
self-labelled via the `/s` convention). Downloaded at runtime by each script —
no local copy is committed.

After the 64-name marketing-subreddit filter: **51,337 rows**, 62 subreddits,
roughly balanced labels (24,948 non-sarcastic / 26,389 sarcastic), spanning
2009-09 → 2016-12.

## Running it

Open `06_Exam.Rproj` in RStudio. Suggested order, cheapest first:

1. `rq3_score_analysis.R` — pure R, no Python
2. `rq1.R` — needs `reticulate` + Python `vaderSentiment`; slow first run, then cached to `vader_scores_rq1.rds`
3. `rq2_embeddings.R` — heaviest; needs the `{text}` package's conda/torch backend (`textrpp_initialize()`) and embeds ~51k comments *and* ~51k parents

Packages are installed on demand via guarded
`if (!requireNamespace(...)) install.packages(...)` calls at the top of each
script. Core stack: `tidyverse`, `caret`, `glmnet`, `pROC`, `patchwork`,
`reticulate`, `text`, `sentimentr`, `doParallel`, `ggridges`, `ggpubr`.

Every split and `train()` call is preceded by `set.seed(123)`.

## Repository contents

- [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) — full walkthrough of the data, methods, figures, and known gaps
- [docs/CLEANING_CROSSCHECK.md](docs/CLEANING_CROSSCHECK.md) — preprocessing decisions cross-checked against the course sessions
- [docs/COURSE_COVERAGE_CROSSCHECK.md](docs/COURSE_COVERAGE_CROSSCHECK.md) — methods and research design cross-checked against all 11 course sessions
- [docs/PLAN_ALIGNMENT.md](docs/PLAN_ALIGNMENT.md) — code cross-checked against the group's *Research Questions & Notes* plan and its cited feature table
- `outputs.docx` — results document with the 36 generated plots

Cached intermediates (`*.rds`), the RStudio workspace (`.RData`), and session
state are gitignored — see [.gitignore](.gitignore).
