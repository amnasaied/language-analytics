## ==================================================================
## SHARED PSYCHOLINGUISTIC FEATURE MODULE
##
## Sourced by BOTH rq1.R and rq2_embeddings.R so the two research
## questions are guaranteed to use an identical feature set. RQ1 asks
## which of these features distinguish sarcasm; RQ2 asks whether this
## same block adds anything on top of semantic embeddings. That second
## question is only meaningful if the block is literally the same
## object -- hence one definition, one file.
##
## Provides:
##   build_feature_dataset()  -> the full cleaned + featurised tibble
##   PSYCH_FEATURES           -> chr vector of MODEL predictor names
##   PSYCH_FEATURE_LABELS     -> pretty names for plots/tables
##
## Feature set (12 predictors), with the literature each is drawn from:
##   Punctuation / typography
##     1. excl_count      exclamation marks     Gonzalez-Ibanez+ 2011
##     2. quest_count     question marks        Gonzalez-Ibanez+ 2011
##     3. ellipsis_count  trailing-off "..."    Gonzalez-Ibanez+ 2011
##     4. quote_count     quotation marks       Gonzalez-Ibanez+ 2011
##     5. caps_ratio      ALL-CAPS word rate    Hutto & Gilbert 2014
##   Lexical cue markers
##     6. interj_count    "wow", "gee", "gosh"  Kreuz & Caucci 2007
##     7. laughter_count  "lol", "haha"         Gonzalez-Ibanez+ 2011
##   Sentiment & incongruity
##     8. vader_compound        literal polarity      Hutto & Gilbert 2014
##     9. intra_incongruity     pos+neg in one text   Riloff+ 2013
##    10. parent_incongruity    |reply - parent|      Joshi+ 2015
##    11. parent_flip           polarity SIGN flip    Riloff+ 2013
##   Control
##    12. log_word_count  length confounds every count feature above
##
## DELIBERATELY NOT MODEL PREDICTORS (kept for descriptive tables only):
##   punct_total  -- exact sum of features 1-4, makes the design matrix
##                   singular; glm would silently drop a coefficient.
##   caps_count   -- collinear with caps_ratio; the ratio is the
##                   length-normalised version we actually want.
##   word_count   -- enters the model as log_word_count instead.
##
## MEASURED, THEN DROPPED -- report this, it is a result:
##   Emphatic repeated punctuation ("??", "?!", "!!") is one of the most
##   cited typographic sarcasm cues (Gonzalez-Ibanez+ 2011), but it is
##   effectively ABSENT from this corpus: across all 51,335 comments
##   there are 0 instances of "??", 1 of "?!", and 13 of "!!". Repeated
##   punctuation was evidently normalised away when SARC was built. A
##   constant-zero predictor cannot be estimated, so the feature is not
##   in the set -- not because the theory is wrong, but because the
##   corpus cannot measure it.
##
## NOT IMPLEMENTED (candidate extension): NRC EmoLex emotional
## incongruity (co-present opposing emotions, Mohammad & Turney 2013).
## It is the one row of the literature table with no feature here;
## note it as a limitation if it stays out.
## ==================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
})

## ------------------------------------------------------------------
## Predictor names. Both scripts build their model matrix from THIS
## vector, so the two feature sets cannot drift apart.
## ------------------------------------------------------------------
PSYCH_FEATURES <- c(
  "excl_count", "quest_count", "ellipsis_count",
  "quote_count", "caps_ratio", "interj_count", "laughter_count",
  "vader_compound", "intra_incongruity", "parent_incongruity",
  "parent_flip", "log_word_count"
)

PSYCH_FEATURE_LABELS <- c(
  excl_count         = "Exclamation marks",
  quest_count        = "Question marks",
  ellipsis_count     = "Ellipses (...)",
  quote_count        = "Quotation marks",
  caps_ratio         = "ALL-CAPS word ratio",
  interj_count       = "Interjections",
  laughter_count     = "Laughter tokens",
  vader_compound     = "VADER sentiment",
  intra_incongruity  = "Intra-comment incongruity",
  parent_incongruity = "Parent-reply incongruity",
  parent_flip        = "Parent-reply polarity flip",
  log_word_count     = "Comment length (log words)"
)

## Marketing-relevant subreddits: tech, gaming, cars, fashion, retail,
## entertainment. Defined once so RQ1 and RQ2 cannot drift here either.
MARKETING_SUBREDDITS <- c(
  "apple", "iphone", "Android", "GooglePixel", "AndroidMasterRace",
  "windowsphone", "Surface", "GalaxyNote7", "galaxynote4", "lgv20",
  "pebble", "nvidia", "intel", "Amd", "razer", "hardware", "techsupport",
  "Steam", "playstation", "PS4", "PS4Pro", "xboxone", "NintendoSwitch",
  "NintendoNX", "wiiu", "askcarsales", "cars", "Autos", "BMW",
  "Volkswagen", "SubaruForester", "subaru", "Miata", "FocusST", "Datsun",
  "streetwear", "StreetwearSales", "sneakermarket", "Sneakers",
  "FashionReps", "supremeclothing", "bapeheads", "goodyearwelt",
  "frugalmalefashion", "BeautyBoxes", "MakeupAddiction", "walmart",
  "starbucks", "tacobell", "TalesFromRetail", "Justrolledintotheshop",
  "Random_Acts_Of_Amazon", "netflix", "boxoffice", "moviecritic",
  "television", "movies", "music", "headphones", "GameDeals", "buildapc",
  "buildapcsales", "pcmasterrace"
)

## Kreuz & Caucci (2007) showed interjections significantly predict
## whether readers RATE an utterance as sarcastic. Interjections proper
## only -- not stance adverbs like "obviously"/"clearly", which are a
## different construct and would muddy the citation.
INTERJECTION_RE <- regex(
  "\\b(wow|gee|gosh|geez|jeez|sheesh|whoa|woah|yikes|ugh|meh|huh|oh|ah|aw|yeah)\\b",
  ignore_case = TRUE
)

## Gonzalez-Ibanez et al. (2011): laughter onomatopoeia was among the
## most reliable manual cues separating sarcastic from sincere tweets.
## NOTE: the "/s" tag itself must NEVER become a feature -- it is the
## mechanism that generated the labels. It was stripped when SARC was
## built (verified: 0 occurrences in the raw file), but do not add it.
LAUGHTER_RE <- regex(
  "\\b(lol+|lmao+|rofl|lulz|ha(ha)+|he(he)+)\\b",
  ignore_case = TRUE
)


## ------------------------------------------------------------------
## load_sarcasm_raw(): download-once, read-from-disk corpus loader.
## ------------------------------------------------------------------
load_sarcasm_raw <- function(
    csv_cache_file = "train-balanced-sarcasm.csv",
    url = "https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv") {

  if (!file.exists(csv_cache_file)) {
    cat("Downloading raw dataset (~255 MB, one time only -- please wait)...\n")

    # R's default download timeout is 60s, far too short for a 255 MB
    # file, and it truncates silently.
    old_timeout <- getOption("timeout")
    options(timeout = 3600)
    on.exit(options(timeout = old_timeout), add = TRUE)

    # Download to a .part file and rename only on success, so a dropped
    # connection cannot leave a truncated file that the next run treats
    # as a valid cache.
    tmp_file <- paste0(csv_cache_file, ".part")
    download.file(url, tmp_file, mode = "wb")
    file.rename(tmp_file, csv_cache_file)
    cat("Saved raw dataset to", csv_cache_file, "\n")
  } else {
    cat("Using locally cached raw dataset:", csv_cache_file, "\n")
  }

  read_csv(csv_cache_file, show_col_types = FALSE)
}


## ------------------------------------------------------------------
## clean_sarcasm(): the SHARED cleaning contract.
##
## Both RQs need a usable parent_comment, because the parent-reply
## incongruity features depend on it. "[deleted]" is Reddit's
## placeholder for removed content, not real text, so it is dropped
## like a missing value.
##
## The filter ORDER here is load-bearing: vader_scores_rq1.rds is a
## positional cache built against exactly this sequence. Change the
## order or the predicates and you MUST delete that cache and re-score.
## ------------------------------------------------------------------
clean_sarcasm <- function(sarcasm_raw) {
  sarcasm_raw %>%
    filter(subreddit %in% MARKETING_SUBREDDITS) %>%
    filter(!is.na(label), !is.na(comment), !is.na(parent_comment)) %>%
    filter(comment != "", parent_comment != "") %>%
    filter(comment != "[deleted]", parent_comment != "[deleted]") %>%
    mutate(label_f = factor(label, levels = c(0, 1),
                            labels = c("NonSarcastic", "Sarcastic")))
}


## ------------------------------------------------------------------
## add_surface_features(): everything computable by regex, no Python.
## ------------------------------------------------------------------
add_surface_features <- function(df) {
  df %>%
    mutate(
      # --- Punctuation -------------------------------------------------
      excl_count     = str_count(comment, "!"),
      quest_count    = str_count(comment, "\\?"),
      # Trailing-off ellipsis only. RQ1 previously folded repeated "???"
      # in here, which is a different cue (emphatic disbelief, below).
      ellipsis_count = str_count(comment, "\\.{3,}"),
      quote_count    = str_count(comment, '"'),
      # Descriptive only -- NOT a model predictor (exact sum => singular).
      punct_total    = excl_count + quest_count + ellipsis_count +
                       quote_count,

      # --- Capitalisation ----------------------------------------------
      word_count = str_count(comment, "\\S+"),
      # length >= 2 so the standalone pronoun "I" is not read as shouting
      caps_count = str_count(comment, "\\b[A-Z]{2,}\\b"),
      caps_ratio = ifelse(word_count > 0, caps_count / word_count, 0),

      # --- Lexical cue markers -----------------------------------------
      interj_count   = str_count(comment, INTERJECTION_RE),
      laughter_count = str_count(comment, LAUGHTER_RE),

      # --- Length control ----------------------------------------------
      # Raw counts above scale with length, so length enters the model
      # explicitly rather than being hidden inside density ratios. This
      # keeps every count coefficient interpretable as "per extra mark,
      # holding comment length constant".
      log_word_count = log1p(word_count)
    )
}


## ------------------------------------------------------------------
## add_vader_features(): sentiment + both incongruity channels.
##
## VADER is the single sentiment engine for the whole project. It is
## the lexicon cited in the proposal, and its emphasis-amplification
## rules (caps, "!!!", degree modifiers) ARE the "emphasis-aware
## sentiment" construct from the literature table. RQ2 previously used
## sentimentr, which measured a different quantity under the same name.
##
## Scoring calls Python once per comment AND once per parent over the
## full cleaned corpus, so the result is cached. Delete the cache file
## to force a re-score.
## ------------------------------------------------------------------
add_vader_features <- function(df, vader_cache_file = "vader_scores_rq1.rds") {

  if (file.exists(vader_cache_file)) {
    cat("Loading cached VADER scores from", vader_cache_file, "...\n")
    vader_cache <- readRDS(vader_cache_file)
    vader_comment_df <- vader_cache$comment
    vader_parent_df  <- vader_cache$parent

    # The cache is positional (bind_cols, no key), so a row-count
    # mismatch means the cleaning contract changed underneath it.
    # Fail loudly rather than silently misalign sentiment with text.
    if (nrow(vader_comment_df) != nrow(df)) {
      stop("VADER cache has ", nrow(vader_comment_df), " rows but the ",
           "cleaned dataset has ", nrow(df), ". The cleaning contract ",
           "changed. Delete '", vader_cache_file, "' and re-run to re-score.")
    }
  } else {
    if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
    library(reticulate)
    py_require("vaderSentiment")
    vader <- import("vaderSentiment.vaderSentiment")
    vader_analyzer <- vader$SentimentIntensityAnalyzer()

    cat("Scoring comment sentiment with VADER on", nrow(df),
        "rows -- this takes a while...\n")
    vc <- lapply(df$comment, vader_analyzer$polarity_scores)
    vader_comment_df <- do.call(rbind, lapply(vc, as.data.frame)) %>%
      rename(VADER_neg = neg, VADER_neu = neu, VADER_pos = pos,
             VADER_compound = compound)

    cat("Scoring parent-comment sentiment with VADER...\n")
    vp <- lapply(df$parent_comment, vader_analyzer$polarity_scores)
    vader_parent_df <- do.call(rbind, lapply(vp, as.data.frame)) %>%
      select(compound) %>%
      rename(VADER_parent_compound = compound)

    saveRDS(list(comment = vader_comment_df, parent = vader_parent_df),
            vader_cache_file)
    cat("Cached VADER scores to", vader_cache_file, "\n")
  }

  bind_cols(df, vader_comment_df, vader_parent_df) %>%
    mutate(
      vader_compound = VADER_compound,

      # Intra-comment incongruity. 2*min(pos, neg) peaks when a comment
      # carries a substantial amount of BOTH positive- and negative-
      # valenced language at once ("I absolutely LOVE waiting an hour
      # for nothing") and is 0 when it is purely one-sided -- the
      # "clashing valence in one utterance" pattern of Riloff+ (2013).
      #
      # Note this replaces RQ2's old sentence-level SD measure, which
      # was structurally 0 for every single-sentence comment and so
      # mostly encoded "is this comment multi-sentence".
      intra_incongruity = 2 * pmin(VADER_pos, VADER_neg),

      # Parent-reply incongruity: how far the reply's polarity sits
      # from the polarity of what it replies to.
      parent_incongruity = abs(VADER_compound - VADER_parent_compound),

      # Polarity SIGN flip -- a distinct construct from the magnitude
      # gap above. Going -0.9 -> -0.1 is a large gap with no flip;
      # +0.4 -> -0.4 is a smaller gap that IS a contrast. Riloff+ (2013)
      # treat the contrast, not the distance, as the sarcasm marker.
      parent_flip = as.integer(
        sign(VADER_compound) != sign(VADER_parent_compound) &
          VADER_compound != 0 & VADER_parent_compound != 0
      ),

      # 3-level category for descriptive plots, using VADER's own
      # documented compound thresholds.
      sentiment_cat = factor(
        case_when(
          VADER_compound >=  0.05 ~ "Positive",
          VADER_compound <= -0.05 ~ "Negative",
          TRUE                    ~ "Neutral"
        ),
        levels = c("Negative", "Neutral", "Positive")
      )
    )
}


## ------------------------------------------------------------------
## build_feature_dataset(): the one entry point both scripts call.
##
## Returns the FULL cleaned + featurised corpus. RQ1 uses all of it;
## RQ2 draws a stratified subsample from the returned frame. Because
## the subsample is a random draw from this same object, RQ1's
## descriptive statistics still describe RQ2's data.
## ------------------------------------------------------------------
build_feature_dataset <- function(cache_file = "features_full.rds",
                                  refresh = FALSE) {
  if (!refresh && file.exists(cache_file)) {
    cat("Loading cached feature dataset from", cache_file, "\n")
    return(readRDS(cache_file))
  }

  d <- load_sarcasm_raw() %>%
    clean_sarcasm() %>%
    add_surface_features() %>%
    add_vader_features()

  cat("Feature dataset built:", nrow(d), "rows,",
      length(PSYCH_FEATURES), "model predictors\n")
  saveRDS(d, cache_file)
  d
}
