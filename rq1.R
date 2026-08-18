
# ==================================================================
## RQ1: What features distinguish sarcastic comments from
##      non-sarcastic ones?
##
## The feature set itself lives in features_psycholinguistic.R, which
## is sourced by BOTH this script and rq2_embeddings.R. RQ2 asks
## whether this same block adds anything on top of semantic
## embeddings, and that question is only answerable if the block is
## literally the same object in both places. See that file for the
## per-feature definitions and the literature behind each one.
##
## Structure of this script:
##   1. Setup (packages)
##   2. Build the shared feature dataset
##   3. Descriptive statistics
##   4. Statistical tests (group comparisons + EFFECT SIZES)
##   5. Classifier (logistic regression, CV)
##   6. Evaluation (confusion matrix, ROC/AUC)
##   7. Figures
##
## READ THIS BEFORE INTERPRETING THE OUTPUT:
## The classifier lands around AUC 0.56. That is the correct answer
## for this feature set on this corpus, not a bug. Section 4 reports
## Cohen's d next to every p-value precisely because at n = 51,335 a
## p-value of 1e-148 coexists with an effect size of ~0.03. See
## docs/ for the full diagnosis; the short version is that SARC's
## labels come from authors self-tagging "/s", which was then stripped
## -- so this corpus is the LEAST typographically marked variety of
## sarcasm, and surface cues have little left to detect. That null is
## what motivates RQ2.
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

# The shared feature definitions used by both RQ1 and RQ2. This also
# pulls in the corpus loader and the VADER scoring/caching logic, so
# there is no data-loading or feature code left in this script.
source("features_psycholinguistic.R")


# ==================================================================
# 2. BUILD THE SHARED FEATURE DATASET
# ==================================================================
# Downloads the corpus on first run, applies the shared cleaning
# contract (marketing subreddits, non-empty comment AND parent, no
# "[deleted]" placeholders), and computes all 12 psycholinguistic
# predictors. Everything is cached to features_full.rds, so only the
# very first run is slow. Pass refresh = TRUE to rebuild.
sarcasm_clean <- build_feature_dataset()

cat("Rows used for RQ1:", nrow(sarcasm_clean), "\n")
print(table(sarcasm_clean$label_f))

glimpse(sarcasm_clean %>% select(label_f, all_of(PSYCH_FEATURES), sentiment_cat))


pal <- c("NonSarcastic" = "#0072B2", "Sarcastic" = "#D55E00")
theme_report <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))


# ==================================================================
# 3. DESCRIPTIVE STATISTICS
# ==================================================================
cat("\n===== Feature Means by Group =====\n")
print(
  sarcasm_clean %>%
    group_by(label_f) %>%
    summarise(n = n(), across(all_of(PSYCH_FEATURES), mean), .groups = "drop") %>%
    mutate(across(where(is.numeric), ~round(.x, 4)))
)

cat("\n===== Sentiment Category Counts (Sarcastic comments only) =====\n")
print(
  sarcasm_clean %>%
    filter(label_f == "Sarcastic") %>%
    count(sentiment_cat) %>%
    mutate(pct = round(100 * n / sum(n), 1))
)


# ==================================================================
# 4. STATISTICAL TESTS + EFFECT SIZES
# ==================================================================
# Welch's t-test and the distribution-free Wilcoxon rank-sum test,
# reported together as in the RQ3 script.
#
# Cohen's d and the single-feature AUC are reported ALONGSIDE the
# p-values, and they are the columns that actually matter. At
# n = 51,335 even a trivial difference returns an astronomically small
# p-value: excl_count reaches p ~ 1e-148 on a mean gap of 0.08 marks
# per comment. Reporting significance without effect size here would
# actively mislead. Convention: |d| < 0.2 is negligible.
cohens_d <- function(x, g) {
  x1 <- x[g == 1]; x0 <- x[g == 0]
  s_pooled <- sqrt(((length(x1) - 1) * var(x1) + (length(x0) - 1) * var(x0)) /
                     (length(x) - 2))
  if (s_pooled == 0) return(NA_real_)
  (mean(x1) - mean(x0)) / s_pooled
}

test_results <- map_dfr(PSYCH_FEATURES, function(v) {
  f  <- as.formula(paste(v, "~ label"))
  tt <- t.test(f, data = sarcasm_clean)
  wt <- suppressWarnings(wilcox.test(f, data = sarcasm_clean))
  a  <- as.numeric(pROC::auc(pROC::roc(sarcasm_clean$label, sarcasm_clean[[v]],
                                       quiet = TRUE)))
  tibble(
    feature      = v,
    mean_nonsarc = tt$estimate[1],
    mean_sarc    = tt$estimate[2],
    cohens_d     = cohens_d(sarcasm_clean[[v]], sarcasm_clean$label),
    solo_auc     = max(a, 1 - a),
    pct_zero     = 100 * mean(sarcasm_clean[[v]] == 0),
    t_p_value    = tt$p.value,
    wilcox_p     = wt$p.value
  )
}) %>% arrange(desc(abs(cohens_d)))

cat("\n===== Group Comparisons: NonSarcastic vs Sarcastic =====\n")
cat("(sorted by |Cohen's d|; note how little the p-values discriminate)\n")
print(test_results, n = Inf)

# How often does a comment carry NO typographic marker at all? This is
# the single biggest reason the classifier below is weak: if most
# comments are all-zero on the surface features, the model has nothing
# to separate them with and correctly falls back to the base rate.
sarcasm_clean <- sarcasm_clean %>%
  mutate(no_surface_marker = (excl_count == 0 & quest_count == 0 &
                                ellipsis_count == 0 & quote_count == 0 &
                                caps_count == 0 & interj_count == 0 &
                                laughter_count == 0))

cat("\n===== Coverage: comments carrying no typographic marker =====\n")
cat(sprintf("%.1f%% of comments have zero surface markers.\n",
            100 * mean(sarcasm_clean$no_surface_marker)))
print(
  sarcasm_clean %>%
    group_by(no_surface_marker) %>%
    summarise(n = n(), pct_sarcastic = round(100 * mean(label), 1),
              .groups = "drop")
)
cat("Overall base rate:", round(100 * mean(sarcasm_clean$label), 1), "%\n")


# ==================================================================
# 5. CLASSIFIER (logistic regression, cross-validated)
# ==================================================================
# The RQ is about which features DISTINGUISH the classes, so logistic
# regression is the natural choice: coefficients are directly
# interpretable as each feature's independent contribution to the
# log-odds of sarcasm, while caret's CV + confusionMatrix workflow
# matches the course material.
#
# Predictors come from PSYCH_FEATURES so this model and RQ2's
# psycholinguistic block cannot drift apart. punct_total, caps_count
# and raw word_count are excluded by construction -- see the header of
# features_psycholinguistic.R for why each one is descriptive-only.
model_data <- sarcasm_clean %>%
  select(label_f, all_of(PSYCH_FEATURES)) %>%
  drop_na()

set.seed(123)
train_idx  <- createDataPartition(model_data$label_f, p = 0.8, list = FALSE)
train_data <- model_data[train_idx, ]
test_data  <- model_data[-train_idx, ]

cat("\nTraining set:", nrow(train_data), "| Test set:", nrow(test_data), "\n")

ctrl <- trainControl(method = "cv", number = 10,
                     savePredictions = "final", classProbs = TRUE)

set.seed(123)
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
# 6. EVALUATION
# ==================================================================
pred_class <- predict(logit_model, newdata = test_data)
pred_prob  <- predict(logit_model, newdata = test_data, type = "prob")[, "Sarcastic"]

cfm <- confusionMatrix(pred_class, test_data$label_f, positive = "Sarcastic")
print(cfm)

roc_obj <- roc(response = test_data$label_f, predictor = pred_prob,
               levels = c("NonSarcastic", "Sarcastic"), quiet = TRUE)
auc_ci <- ci.auc(roc_obj)
cat(sprintf("\nAUC: %.3f  95%% CI [%.3f, %.3f]\n",
            as.numeric(auc(roc_obj)), auc_ci[1], auc_ci[3]))


# ==================================================================
# 7. FIGURES
# ==================================================================
# Each figure is written to figures/ as a 300-dpi PNG (publication
# quality for the term paper) and then printed to the Plots pane.
fig_dir <- "figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir)

# --- FIGURE 1: Sarcastic comments' sentiment distribution -----------
sent_dist <- sarcasm_clean %>%
  filter(label_f == "Sarcastic") %>%
  count(sentiment_cat) %>%
  mutate(pct = n / sum(n))

p1a <- ggplot(sent_dist, aes(sentiment_cat, n, fill = sentiment_cat)) +
  geom_col(width = .6, show.legend = FALSE) +
  geom_text(aes(label = percent(pct, accuracy = 0.1)), vjust = -0.4) +
  scale_fill_manual(values = c(Negative = "#D55E00", Neutral = "#999999",
                               Positive = "#0072B2")) +
  labs(title = "Sentiment of Sarcastic Comments", x = NULL,
       y = "Number of Comments") +
  theme_report

p1b <- ggplot(sarcasm_clean %>% filter(label_f == "Sarcastic"),
              aes(x = vader_compound)) +
  geom_histogram(bins = 50, fill = "#D55E00", alpha = .8) +
  geom_vline(xintercept = c(-0.05, 0.05), linetype = "dashed", color = "grey30") +
  labs(title = "VADER Compound Score Distribution",
       subtitle = "Sarcastic comments only",
       x = "VADER Compound Score", y = "Count") +
  theme_report

fig1 <- (p1a | p1b) +
  plot_annotation(title = "Are Sarcastic Comments More Positive or Negative?",
                  subtitle = "Dashed lines = VADER's neutral band [-0.05, 0.05]")

ggsave(file.path(fig_dir, "fig1_sentiment_distribution.png"), fig1,
       width = 11, height = 5, dpi = 300)
print(fig1)


# --- FIGURE 2: Feature comparison boxplots (sarcastic vs not) -------
box_features <- c("excl_count", "caps_ratio", "interj_count",
                  "vader_compound", "intra_incongruity", "parent_incongruity")

long_features <- sarcasm_clean %>%
  select(label_f, all_of(box_features)) %>%
  pivot_longer(-label_f, names_to = "feature", values_to = "value")

fig2 <- ggplot(long_features, aes(label_f, value, fill = label_f)) +
  geom_boxplot(outlier.alpha = 0.05, width = .6, show.legend = FALSE) +
  facet_wrap(~ feature, scales = "free_y",
             labeller = as_labeller(PSYCH_FEATURE_LABELS)) +
  scale_fill_manual(values = pal) +
  labs(title = "Feature Distributions by Sarcasm Label", x = NULL, y = "Value") +
  theme_report

ggsave(file.path(fig_dir, "fig2_feature_boxplots.png"), fig2,
       width = 10, height = 7, dpi = 300)
print(fig2)


# --- FIGURE 3: Effect sizes, NOT p-values ----------------------------
# Deliberately plots Cohen's d rather than significance. The dashed
# line marks |d| = 0.2, the conventional floor for a "small" effect --
# it makes visible at a glance that essentially every feature falls
# below it despite p-values in the 1e-100 range.
fig3 <- ggplot(test_results,
               aes(reorder(feature, abs(cohens_d)), cohens_d,
                   fill = abs(cohens_d) >= 0.2)) +
  geom_col(show.legend = FALSE) +
  geom_hline(yintercept = c(-0.2, 0.2), linetype = "dashed", color = "grey40") +
  coord_flip() +
  scale_x_discrete(labels = PSYCH_FEATURE_LABELS) +
  scale_fill_manual(values = c("FALSE" = "#999999", "TRUE" = "#D55E00")) +
  labs(title = "Effect Size of Each Feature",
       subtitle = paste("Cohen's d, sarcastic vs non-sarcastic.",
                        "Dashed line = |d| = 0.2, the 'small effect' threshold"),
       x = NULL, y = "Cohen's d") +
  theme_report

ggsave(file.path(fig_dir, "fig3_effect_sizes.png"), fig3,
       width = 8, height = 5, dpi = 300)
print(fig3)


# --- FIGURE 4: Feature importance (logistic regression) -------------
imp_df <- importance$importance %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall))

fig4 <- ggplot(imp_df, aes(reorder(feature, Overall), Overall, fill = Overall)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_x_discrete(labels = PSYCH_FEATURE_LABELS) +
  scale_fill_viridis_c(option = "mako", direction = -1) +
  labs(title = "Feature Importance for Distinguishing Sarcasm",
       subtitle = "Logistic regression (variable importance = |z-value|)",
       x = NULL, y = "Importance") +
  theme_report

ggsave(file.path(fig_dir, "fig4_feature_importance.png"), fig4,
       width = 8, height = 5, dpi = 300)
print(fig4)


# --- FIGURE 5: Confusion matrix heatmap ------------------------------
cfm_df <- as.data.frame(cfm$table)

fig5 <- ggplot(cfm_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_viridis_c(option = "inferno", direction = -1) +
  labs(title = "Confusion Matrix: Sarcasm Classifier",
       subtitle = paste0("Accuracy = ", round(cfm$overall["Accuracy"], 3),
                         " | AUC = ", round(auc(roc_obj), 3)),
       x = "Predicted Label", y = "True Label") +
  theme_report +
  theme(panel.grid = element_blank())

ggsave(file.path(fig_dir, "fig5_confusion_matrix.png"), fig5,
       width = 7, height = 5, dpi = 300)
print(fig5)


# --- FIGURE 6: ROC curve ----------------------------------------------
roc_df <- tibble(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities)

fig6 <- ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(color = "#D55E00", linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(title = "ROC Curve: Sarcasm Classifier",
       subtitle = sprintf("AUC = %.3f, 95%% CI [%.3f, %.3f]",
                          as.numeric(auc(roc_obj)), auc_ci[1], auc_ci[3]),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_report

ggsave(file.path(fig_dir, "fig6_roc_curve.png"), fig6,
       width = 6, height = 5, dpi = 300)
print(fig6)

cat("\nFigures saved to", normalizePath(fig_dir), "\n")
