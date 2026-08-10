## ==================================================================
## RQ1: What features distinguish sarcastic comments from
##      non-sarcastic ones?
##
## Features (drawn from sarcasm/irony theory + literature, LIWC
## substitutes since LIWC is paid & not R-compatible):
##   1. Punctuation        (regex counts: !, ?, ..., quotes)
##   2. Sentiment          (VADER polarity, via reticulate)
##   3. Capitalization     (ALL-CAPS word count/ratio)
##   4. Intra-comment incongruity
##        (co-occurrence of positive AND negative valence within
##         the SAME comment -- the classic "positive sentiment,
##         negative situation" sarcasm marker, Riloff et al. 2013;
##         Joshi et al. 2015)
##   5. Parent-reply incongruity
##        (|VADER compound(comment) - VADER compound(parent_comment)|
##         -- how much the reply's sentiment clashes with the
##         sentiment of the comment it is replying to)
##
## Structure of this script:
##   1. Setup (packages)
##   2. Load data
##   3. Data cleaning
##   4. Feature engineering
##   5. Descriptive statistics
##   6. Statistical tests (group comparisons per feature)
##   7. Classifier (logistic regression, CV) using the 5 features
##   8. Evaluation (confusion matrix, ROC/AUC)
##   9. Figures
## ==================================================================


# ==================================================================
# 1. SETUP
# ==================================================================
library(tidyverse)
library(stringr)
library(scales)

if (!requireNamespace("caret", quietly = TRUE)) install.packages("caret")
library(caret)          # train/test split, CV training, confusionMatrix

if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")
library(pROC)           # ROC curve / AUC

if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")
library(patchwork)      # combine ggplot panels

if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
library(reticulate)     # bridge to Python's vaderSentiment
py_require("vaderSentiment")   # uncomment on first run if needed

vader <- import("vaderSentiment.vaderSentiment")
VADER_analyzer <- vader$SentimentIntensityAnalyzer()

# quick sanity check
VADER_analyzer$polarity_scores("Oh great, ANOTHER Monday. Just what I needed!!!")


# ==================================================================
# 2. LOAD DATA
# ==================================================================
# Same source + same marketing-related subreddit filter as the RQ3
# script, so RQ1 and RQ3 stay comparable/consistent within the
# project.
library(readr)
url <- "https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv"
sarcasm <- read_csv(url)

sarcasm <- sarcasm %>%
  filter(subreddit %in% c("apple", "iphone", "Android", "GooglePixel",
                          "AndroidMasterRace", "windowsphone", "Surface",
                          "GalaxyNote7", "galaxynote4", "lgv20", "pebble",
                          "nvidia", "intel", "Amd", "razer", "hardware",
                          "techsupport", "Steam", "playstation", "PS4",
                          "PS4Pro", "xboxone", "NintendoSwitch", "NintendoNX",
                          "wiiu", "askcarsales", "cars", "Autos", "BMW",
                          "Volkswagen", "SubaruForester", "subaru", "Miata",
                          "FocusST", "Datsun", "streetwear", "StreetwearSales",
                          "sneakermarket", "Sneakers", "FashionReps",
                          "supremeclothing", "bapeheads", "goodyearwelt",
                          "frugalmalefashion", "BeautyBoxes", "MakeupAddiction",
                          "walmart", "starbucks", "tacobell", "TalesFromRetail",
                          "Justrolledintotheshop", "Random_Acts_Of_Amazon",
                          "netflix", "boxoffice", "moviecritic", "television",
                          "movies", "music", "headphones", "GameDeals",
                          "buildapc", "buildapcsales", "pcmasterrace"))

cat("Rows after subreddit filter:", nrow(sarcasm), "\n")
table(sarcasm$label)   # already balanced 0/1


# ==================================================================
# 3. DATA CLEANING
# ==================================================================
# We need BOTH comment and parent_comment to be usable text, since
# feature 5 (parent-reply incongruity) depends on parent_comment.
# "[deleted]" is Reddit's placeholder for removed content, not real
# text -- drop it like a missing value.
sarcasm_clean <- sarcasm %>%
  filter(!is.na(label), !is.na(comment), !is.na(parent_comment)) %>%
  filter(comment != "", parent_comment != "") %>%
  filter(comment != "[deleted]", parent_comment != "[deleted]") %>%
  mutate(label_f = factor(label, levels = c(0, 1),
                          labels = c("Non-sarcastic", "Sarcastic")))

cat("Rows used for feature engineering:", nrow(sarcasm_clean), "\n")
# NOTE: the VADER step below runs one Python call per comment AND
# one per parent_comment via reticulate, over the FULL filtered
# dataset (no subsampling) -- this can take a long time on a large
# dataset. Consider running it once and caching the result with
# saveRDS()/readRDS() so you don't have to re-score on every re-run.


# ==================================================================
# 4. FEATURE ENGINEERING
# ==================================================================

# --- Feature 1: Punctuation (regex counts) --------------------------
# Exclamation marks, question marks, ellipses, and quotation marks
# are the punctuation markers most consistently linked to sarcasm
# and irony in the literature (emphatic/expressive punctuation).
sarcasm_clean <- sarcasm_clean %>%
  mutate(
    excl_count      = str_count(comment, "!"),
    quest_count     = str_count(comment, "\\?"),
    ellipsis_count = str_count(comment, "\\.{3,}|\\?{3,}"),
    quote_count     = str_count(comment, '"'),
    punct_total     = excl_count + quest_count + ellipsis_count + quote_count
  )

# --- Feature 3: Capitalization ---------------------------------------
# ALL-CAPS words (length >= 2, so single-letter "I" doesn't count as
# shouting/emphasis) are a common intensity/emphasis marker.
sarcasm_clean <- sarcasm_clean %>%
  mutate(
    word_count     = str_count(comment, "\\S+"),
    caps_count     = str_count(comment, "\\b[A-Z]{2,}\\b"),
    caps_ratio     = ifelse(word_count > 0, caps_count / word_count, 0)
  )

# --- Feature 2: Sentiment (VADER) + Feature 4: Intra-comment ----------
# incongruity + Feature 5 setup ---------------------------------------
# One VADER call per comment gives us pos/neu/neg/compound in a
# single pass -- feature 2 (sentiment) and feature 4 (intra-comment
# incongruity) are both derived from this same call, no extra cost.
vader_cache_file <- "vader_scores_rq1.rds"

if (file.exists(vader_cache_file)) {
  cat("Loading cached VADER scores from", vader_cache_file, "...\n")
  vader_cache <- readRDS(vader_cache_file)
  VADER_comment_df <- vader_cache$comment
  VADER_parent_df  <- vader_cache$parent
} else {
  cat("Scoring comment sentiment with VADER on the FULL filtered dataset",
      "(", nrow(sarcasm_clean), "rows -- this can take a long time)...\n")
  VADER_comment <- lapply(sarcasm_clean$comment, VADER_analyzer$polarity_scores)
  VADER_comment_df <- do.call(rbind, lapply(VADER_comment, as.data.frame)) %>%
    rename(VADER_neg = neg, VADER_neu = neu, VADER_pos = pos,
           VADER_compound = compound)
  
  cat("Scoring parent-comment sentiment with VADER...\n")
  VADER_parent <- lapply(sarcasm_clean$parent_comment, VADER_analyzer$polarity_scores)
  VADER_parent_df <- do.call(rbind, lapply(VADER_parent, as.data.frame)) %>%
    select(compound) %>%
    rename(VADER_parent_compound = compound)
  
  saveRDS(list(comment = VADER_comment_df, parent = VADER_parent_df),
          vader_cache_file)
  cat("Cached VADER scores to", vader_cache_file, "for future runs.\n")
}

sarcasm_clean <- bind_cols(sarcasm_clean, VADER_comment_df, VADER_parent_df)

sarcasm_clean <- sarcasm_clean %>%
  mutate(
    # Feature 4: intra-comment incongruity.
    # 2 * min(pos, neg) is highest when a comment contains a
    # substantial amount of BOTH positive- and negative-valenced
    # language at once (e.g. "I absolutely LOVE waiting an hour for
    # nothing"), and is 0 when a comment is purely one-sided --
    # exactly the "clashing valence within one utterance" pattern
    # that sarcasm-detection literature treats as an incongruity cue.
    intra_incongruity = 2 * pmin(VADER_pos, VADER_neg),
    
    # Feature 5: parent-reply incongruity.
    # Absolute difference in compound sentiment between the reply
    # and the comment it responds to -- large values mean the reply
    # has swung to the opposite emotional register of its context.
    parent_incongruity = abs(VADER_compound - VADER_parent_compound),
    
    # 3-level sentiment category for descriptive plots (VADER's own
    # documented compound thresholds).
    sentiment_cat = case_when(
      VADER_compound >=  0.05 ~ "Positive",
      VADER_compound <= -0.05 ~ "Negative",
      TRUE                    ~ "Neutral"
    ),
    sentiment_cat = factor(sentiment_cat, levels = c("Negative", "Neutral", "Positive"))
  )

glimpse(sarcasm_clean %>%
          select(label_f, excl_count, quest_count, ellipsis_count, quote_count,
                 punct_total, caps_count, caps_ratio, VADER_compound,
                 intra_incongruity, parent_incongruity, sentiment_cat))

pal <- c("Non-sarcastic" = "#0072B2", "Sarcastic" = "#D55E00")
theme_report <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))


# ==================================================================
# 5. DESCRIPTIVE STATISTICS
# ==================================================================
cat("\n===== Feature Means by Group =====\n")
print(
  sarcasm_clean %>%
    group_by(label_f) %>%
    summarise(
      n                    = n(),
      mean_punct           = mean(punct_total),
      mean_excl            = mean(excl_count),
      mean_quest           = mean(quest_count),
      mean_caps_ratio      = mean(caps_ratio),
      mean_VADER_compound  = mean(VADER_compound),
      mean_intra_incong    = mean(intra_incongruity),
      mean_parent_incong   = mean(parent_incongruity),
      .groups = "drop"
    )
)

cat("\n===== Sentiment Category Counts (Sarcastic comments only) =====\n")
print(
  sarcasm_clean %>%
    filter(label_f == "Sarcastic") %>%
    count(sentiment_cat) %>%
    mutate(pct = round(100 * n / sum(n), 1))
)


# ==================================================================
# 6. STATISTICAL TESTS (group comparisons per feature)
# ==================================================================
# Wilcoxon rank-sum tests: distribution-free, appropriate for the
# skewed count/ratio features here. Reported alongside Welch's
# t-test as in the RQ3 script for consistency.
feature_vars <- c("punct_total", "excl_count", "quest_count",
                  "caps_ratio", "VADER_compound",
                  "intra_incongruity", "parent_incongruity")

test_results <- map_dfr(feature_vars, function(v) {
  f  <- as.formula(paste(v, "~ label"))
  tt <- t.test(f, data = sarcasm_clean)
  wt <- wilcox.test(f, data = sarcasm_clean)
  tibble(
    feature       = v,
    mean_nonsarc  = tt$estimate[1],
    mean_sarc     = tt$estimate[2],
    t_p_value     = tt$p.value,
    wilcox_p      = wt$p.value
  )
})

cat("\n===== Group Comparison Tests (Non-sarcastic vs Sarcastic) =====\n")
print(test_results)


# ==================================================================
# 7. CLASSIFIER (logistic regression, cross-validated)
# ==================================================================
# The RQ here is about which features DISTINGUISH the two classes,
# so a logistic regression is the natural model: it is directly
# interpretable (coefficients = each feature's independent
# contribution to the log-odds of sarcasm) while caret's CV +
# confusionMatrix workflow follows the same pattern used for the
# text classifiers in the course material.

model_data <- sarcasm_clean %>%
  select(label_f, punct_total, excl_count, quest_count, ellipsis_count,
         quote_count, caps_ratio, VADER_compound, intra_incongruity,
         parent_incongruity) %>%
  drop_na()

set.seed(123)
train_idx   <- createDataPartition(model_data$label_f, p = 0.8, list = FALSE)
train_data  <- model_data[train_idx, ]
test_data   <- model_data[-train_idx, ]

cat("\nTraining set:", nrow(train_data), "| Test set:", nrow(test_data), "\n")

ctrl <- trainControl(method = "cv", number = 10,
                     savePredictions = "final", classProbs = TRUE)

set.seed(123)
levels(train_data$label_f) <- make.names(levels(train_data$label_f))
levels(test_data$label_f)  <- make.names(levels(test_data$label_f))
logit_model <- train(
  label_f ~ .,
  data      = train_data,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "Accuracy"
)

print(logit_model)
print(summary(logit_model$finalModel))

# Feature importance (absolute standardized coefficient / z-value)
importance <- varImp(logit_model)
print(importance)


# ==================================================================
# 8. EVALUATION
# ==================================================================
pred_class <- predict(logit_model, newdata = test_data)
pred_prob  <- predict(logit_model, newdata = test_data, type = "prob")[, "Sarcastic"]

cfm <- confusionMatrix(pred_class, test_data$label_f, positive = "Sarcastic")
print(cfm)

levels(test_data$label_f)

roc_obj <- roc(
  response = test_data$label_f,
  predictor = pred_prob,
  levels = c("Non.sarcastic", "Sarcastic")
)
cat("\nAUC:", round(auc(roc_obj), 3), "\n")


# ==================================================================
# 9. FIGURES
# ==================================================================

# --- FIGURE 1: Sarcastic comments' sentiment distribution -----------
sent_dist <- sarcasm_clean %>%
  filter(label_f == "Sarcastic") %>%
  count(sentiment_cat) %>%
  mutate(pct = n / sum(n))

p1a <- ggplot(sent_dist, aes(sentiment_cat, n, fill = sentiment_cat)) +
  geom_col(width = .6, show.legend = FALSE) +
  geom_text(aes(label = percent(pct, accuracy = 0.1)), vjust = -0.4) +
  scale_fill_manual(values = c(Negative = "#D55E00", Neutral = "#999999", Positive = "#0072B2")) +
  labs(title = "Sentiment of Sarcastic Comments", x = NULL, y = "Number of Comments") +
  theme_report

p1b <- ggplot(sarcasm_clean %>% filter(label_f == "Sarcastic"),
              aes(x = VADER_compound)) +
  geom_histogram(bins = 50, fill = "#D55E00", alpha = .8) +
  geom_vline(xintercept = c(-0.05, 0.05), linetype = "dashed", color = "grey30") +
  labs(title = "VADER Compound Score Distribution", subtitle = "Sarcastic comments only",
       x = "VADER Compound Score", y = "Count") +
  theme_report

(p1a | p1b) +
  plot_annotation(title = "Are Sarcastic Comments More Positive or Negative?",
                  subtitle = "Dashed lines = VADER's neutral band [-0.05, 0.05]")


# --- FIGURE 2: Feature comparison boxplots (sarcastic vs not) -------
long_features <- sarcasm_clean %>%
  select(label_f, punct_total, caps_ratio, VADER_compound,
         intra_incongruity, parent_incongruity) %>%
  pivot_longer(-label_f, names_to = "feature", values_to = "value")

feature_labels <- c(
  punct_total        = "Punctuation Count",
  caps_ratio          = "Capitalization Ratio",
  VADER_compound      = "VADER Sentiment",
  intra_incongruity   = "Intra-comment Incongruity",
  parent_incongruity  = "Parent-Reply Incongruity"
)

ggplot(long_features, aes(label_f, value, fill = label_f)) +
  geom_boxplot(outlier.alpha = 0.05, width = .6, show.legend = FALSE) +
  facet_wrap(~ feature, scales = "free_y",
             labeller = as_labeller(feature_labels)) +
  scale_fill_manual(values = pal) +
  labs(title = "Feature Distributions by Sarcasm Label", x = NULL, y = "Value") +
  theme_report


# --- FIGURE 3: Feature importance (logistic regression) -------------
imp_df <- importance$importance %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

ggplot(imp_df, aes(reorder(feature, Overall), Overall, fill = Overall)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_fill_viridis_c(option = "mako", direction = -1) +
  labs(title = "Feature Importance for Distinguishing Sarcasm",
       subtitle = "Logistic regression (variable importance = |t-value|)",
       x = NULL, y = "Importance") +
  theme_report


# --- FIGURE 4: Confusion matrix heatmap ------------------------------
cfm_df <- as.data.frame(cfm$table)

ggplot(cfm_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_viridis_c(option = "inferno", direction = -1) +
  labs(title = "Confusion Matrix: Sarcasm Classifier",
       subtitle = paste0("Accuracy = ", round(cfm$overall["Accuracy"], 3),
                         " | AUC = ", round(auc(roc_obj), 3)),
       x = "Predicted Label", y = "True Label") +
  theme_report +
  theme(panel.grid = element_blank())


# --- FIGURE 5: ROC curve ----------------------------------------------
roc_df <- tibble(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(color = "#D55E00", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(title = "ROC Curve: Sarcasm Classifier",
       subtitle = paste0("AUC = ", round(auc(roc_obj), 3)),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_report