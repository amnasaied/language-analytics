# ============================================================
# save_plots.R — save every figure from Final Project.R as a PNG
# ------------------------------------------------------------
# HOW TO USE:
#   Run this in the SAME RStudio session right AFTER running
#   Final Project.R, so all objects the plots need are already in
#   memory. This does NO expensive recompute (no CSV download, no
#   VADER, no model fitting) — it just redraws the plots.
#
#   In the console:   source("save_plots.R")
#
#   Output: ./plots/plot_001.png, plot_002.png, ...  (in script order)
#   Each plot is wrapped in try(), so if one object is missing that
#   single plot is skipped instead of aborting the whole run.
# ============================================================

library(ggplot2); library(dplyr); library(tidyr)
library(scales);  library(forcats); library(pROC)

dir.create("plots", showWarnings = FALSE)
png(file.path("plots", "plot_%03d.png"), width = 1200, height = 800, res = 150)

# ---------- RQ1 / EDA ----------
try(print(
  ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
    geom_bar(fill = "steelblue") +
    geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
    labs(title = "Class balance: sarcastic vs. non-sarcastic", x = NULL, y = "Count") +
    theme_minimal()
))

try(print(
  data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 15) %>%
    ggplot(aes(x = fct_reorder(subreddit, n), y = n)) +
    geom_col(fill = "darkorange") + coord_flip() +
    labs(title = "Top 15 subreddits by comment count", x = NULL, y = "Comments") +
    theme_minimal()
))

try(print(
  data %>% count(subreddit, label) %>% group_by(subreddit) %>%
    mutate(pct = n / sum(n)) %>%
    filter(subreddit %in% (data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 15) %>% pull(subreddit))) %>%
    ggplot(aes(x = subreddit, y = pct, fill = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
    geom_col(position = "fill") + coord_flip() +
    labs(title = "Label balance within top 15 subreddits", x = NULL, y = "Proportion", fill = NULL) +
    theme_minimal()
))

try(print(
  ggplot(data, aes(x = comment_len)) +
    geom_histogram(binwidth = 2, fill = "steelblue", color = "white") +
    coord_cartesian(xlim = c(0, quantile(data$comment_len, .99))) +
    labs(title = "Comment length distribution (99th pct trimmed)", x = "Word count", y = "Number of comments") +
    theme_minimal()
))

try(print(
  ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")),
                   y = comment_len, fill = factor(label))) +
    geom_boxplot(outlier.alpha = 0.1) +
    coord_cartesian(ylim = c(0, quantile(data$comment_len, .99))) +
    labs(title = "Comment length by label", x = NULL, y = "Word count") +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(data, aes(x = score)) +
    geom_histogram(binwidth = 1, fill = "darkgreen", color = "white") +
    coord_cartesian(xlim = c(-10, 30)) +
    labs(title = "Score distribution (trimmed for display)", x = "Score", y = "Number of comments") +
    theme_minimal()
))

# feature availability (needs overall_long / by_label from Part 1b)
try(print(
  ggplot(overall_long, aes(x = feature, y = pct, fill = status)) +
    geom_col(width = 0.75) +
    geom_text(data = subset(overall_long, status == "Has feature"),
              aes(label = sprintf("%.1f%%", pct)), hjust = -0.1, size = 3) +
    coord_flip() +
    scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
    scale_fill_manual(values = c("No feature" = "grey80", "Has feature" = "#2c7fb8")) +
    labs(title = "Feature availability across all comments", x = NULL, y = "% of comments", fill = NULL) +
    theme_minimal()
))

try(print(
  ggplot(by_label, aes(x = feature, y = pct_present, fill = label_txt)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.7) + coord_flip() +
    scale_y_continuous(labels = label_percent(scale = 1)) +
    scale_fill_manual(values = c("Not sarcastic" = "grey65", "Sarcastic" = "#d95f0e")) +
    labs(title = "Feature availability: sarcastic vs. not sarcastic",
         x = NULL, y = "% of comments containing feature", fill = NULL) +
    theme_minimal()
))

# ---------- RQ1 sentiment (Part 2) ----------
try(print(
  ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")),
                   y = vader_comment, fill = factor(label))) +
    geom_boxplot(outlier.alpha = 0.15) +
    labs(title = "Sentiment (VADER compound) by sarcasm label", x = NULL, y = "VADER compound score") +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(sent_summary, aes(x = label_txt, y = pct,
                           fill = fct_relevel(sentiment_category, "negative","neutral","positive"))) +
    geom_col(position = "stack") +
    geom_text(aes(label = percent(pct, accuracy = 1)), position = position_stack(vjust = 0.5), color = "white", size = 3.5) +
    scale_y_continuous(labels = percent) +
    scale_fill_manual(values = c(negative = "#d73027", neutral = "#999999", positive = "#1a9850"), name = "Sentiment") +
    labs(title = "Sentiment polarity composition", x = NULL, y = "% of comments") +
    theme_minimal()
))

try(print(
  ggplot(diverging_data, aes(x = label_txt, y = pct_signed, fill = sentiment_category)) +
    geom_col() + geom_hline(yintercept = 0) +
    scale_y_continuous(labels = function(x) percent(abs(x))) +
    scale_fill_manual(values = c(negative = "#d73027", positive = "#1a9850")) +
    coord_flip() +
    labs(title = "Positive vs. negative share by label (neutral excluded)", x = NULL, y = "% of comments", fill = "Sentiment") +
    theme_minimal()
))

# ---------- RQ1 descriptive comparisons (Part 3b) ----------
try(print(
  ggplot(density_long, aes(x = label_txt, y = value, fill = label_txt)) +
    geom_boxplot(outlier.alpha = 0.1) + facet_wrap(~ feature, scales = "free_y") +
    labs(title = "Density-type features by sarcasm label", x = NULL, y = NULL) +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(sparse_prevalence_long, aes(x = feature, y = pct, fill = label_txt)) +
    geom_col(position = "dodge") + scale_y_continuous(labels = percent) + coord_flip() +
    labs(title = "Sparse features: % of comments containing each marker", x = NULL, y = "% of comments", fill = NULL) +
    theme_minimal()
))

# ---------- RQ1 separability (Part 7 / 7b) ----------
try(print(
  ggplot(lda_plot_data, aes(x = LD1, fill = label_txt)) +
    geom_density(alpha = 0.5) +
    labs(title = "Are sarcastic and non-sarcastic comments separate clusters?",
         subtitle = "Best-case linear separation using all 11 features",
         x = "Discriminant score", y = "Density", fill = NULL) +
    theme_minimal()
))

try(print(
  ggplot(lda_plot_sample, aes(x = LD1, y = label_txt, color = label_txt)) +
    geom_jitter(height = 0.2, alpha = 0.3, size = 1) +
    labs(title = "Individual comments along the discriminant axis (5,000-point sample)",
         x = "Discriminant score", y = NULL, color = NULL) +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(pca_scores_sample, aes(x = PC1, y = PC2, color = label_txt)) +
    geom_point(alpha = 0.3, size = 1) +
    labs(title = "PCA scatter — do the two labels form visible clusters?",
         subtitle = "5,000-point sample, PC1 vs PC2", color = NULL) +
    theme_minimal()
))

# ---------- RQ2 (embeddings) ----------
try(print(
  final_comparison %>%
    pivot_longer(cols = c(accuracy, precision, recall, f1, auc), names_to = "metric", values_to = "value") %>%
    ggplot(aes(x = model, y = value, fill = model)) +
    geom_col() + facet_wrap(~ metric, scales = "free_y") + coord_flip() +
    labs(title = "Model performance comparison (RQ2)", x = NULL, y = NULL) +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggroc(list("Model 1: embeddings only" = roc_m1, "Model 2: embeddings + features" = roc_m2,
             "XGBoost: embeddings only" = roc_xgb1, "XGBoost: embeddings + features" = roc_xgb2), size = 0.9) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey60") +
    labs(title = "ROC curves — embeddings-only vs. embeddings+features",
         subtitle = "Linear (ridge) and non-linear (XGBoost) classifiers",
         color = NULL, x = "Specificity", y = "Sensitivity") +
    theme_minimal()
))

try(print(
  ggroc(list("Model 1: embeddings only" = roc_m1_sub, "Model 2: embeddings + features" = roc_m2_sub,
             "RQ1 model: features + subreddit FE" = roc_rq1), size = 0.9) +
    geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey60") +
    labs(title = "ROC curves — 3-way comparison including the RQ1 model",
         subtitle = "Evaluated on the RQ1-comparable test subset",
         color = NULL, x = "Specificity", y = "Sensitivity") +
    theme_minimal()
))

try(print(
  ggplot(cm_all, aes(x = Actual, y = Predicted, fill = pct)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(Freq, "\n(", scales::percent(pct, accuracy = 0.1), ")")), size = 3.2) +
    scale_fill_gradient(low = "white", high = "steelblue", labels = scales::percent) +
    facet_wrap(~ model) +
    labs(title = "Confusion matrices by model", fill = "% of test set") +
    theme_minimal()
))

# ---------- RQ3 (score vs sarcasm) ----------
try(print(
  ggplot(data, aes(x = score)) +
    geom_histogram(binwidth = 1, fill = "darkgreen", color = "white") +
    coord_cartesian(xlim = c(-10, 30)) +
    labs(title = "Score distribution after cleaning (trimmed for display)", x = "Score", y = "Number of comments") +
    theme_minimal()
))

try(print(
  ggplot(data, aes(x = as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC"))) +
    geom_histogram(bins = 50, fill = "steelblue", color = "white") +
    labs(title = "Distribution of comment posting times (after cleaning)", x = "Posting date", y = "Number of comments") +
    theme_minimal()
))

try(print(
  ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")),
                   y = score, fill = factor(label))) +
    geom_boxplot(outlier.alpha = 0.1) +
    coord_cartesian(ylim = quantile(data$score, c(.01, .99))) +
    labs(title = "Score by label (1st-99th percentile view)", x = NULL, y = "Score") +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(paired_wide, aes(x = score_0, y = score_1)) +
    geom_point(alpha = 0.3, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
    coord_cartesian(xlim = quantile(c(paired_wide$score_0, paired_wide$score_1), c(.01,.99)),
                    ylim = quantile(c(paired_wide$score_0, paired_wide$score_1), c(.01,.99))) +
    labs(title = "Paired comparison: sarcastic vs. non-sarcastic reply score",
         subtitle = "Above the line = sarcastic scored higher",
         x = "Non-sarcastic reply score", y = "Sarcastic reply score") +
    theme_minimal()
))

try(print(
  ggplot(paired_wide, aes(x = diff)) +
    geom_histogram(binwidth = 1, fill = "darkorange", color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    geom_vline(xintercept = median(paired_wide$diff), color = "red", linewidth = 0.8) +
    coord_cartesian(xlim = quantile(paired_wide$diff, c(.01, .99))) +
    labs(title = "Within-pair score difference (sarcastic - non-sarcastic)",
         subtitle = "Red = median difference, dashed = zero", x = "Score difference", y = "Number of pairs") +
    theme_minimal()
))

try(print(
  ggplot(tibble(boot_median = boot_out$t[, 1]), aes(x = boot_median)) +
    geom_histogram(bins = 40, fill = "seagreen", color = "white") +
    geom_vline(xintercept = boot.ci(boot_out, type = "perc")$percent[4:5], color = "red", linetype = "dashed") +
    geom_vline(xintercept = median(paired_wide$diff), color = "black") +
    labs(title = "Bootstrap distribution of the median paired difference",
         subtitle = "Red dashed = 95% CI, black = observed median",
         x = "Bootstrapped median difference", y = "Count (of 2,000 resamples)") +
    theme_minimal()
))

try(print(
  ggplot(forest_data, aes(x = estimate, y = term)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high), color = "steelblue", linewidth = 0.8) +
    labs(title = "Regression coefficients (95% CI)",
         subtitle = "Outcome: signed-log score. Interval crossing zero = not significant.",
         x = "Estimate (signed-log score units)", y = NULL) +
    theme_minimal()
))

try(print(
  ggplot(data_sample, aes(x = as.POSIXct(created_utc, origin = "1970-01-01", tz = "UTC"),
                          y = score_transformed, color = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
    geom_smooth(method = "loess", se = TRUE) +
    labs(title = "Score trend over time, by sarcasm label",
         subtitle = "Signed-log score, loess-smoothed (20,000-row sample)",
         x = "Posting date", y = "Signed-log score", color = NULL) +
    theme_minimal()
))

try(print(
  ggplot(data_sample, aes(x = comment_len, y = score_transformed,
                          color = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
    geom_smooth(method = "loess", se = TRUE) +
    coord_cartesian(xlim = c(0, quantile(data$comment_len, .95))) +
    labs(title = "Score vs. comment length, by sarcasm label",
         x = "Comment length (words)", y = "Signed-log score", color = NULL) +
    theme_minimal()
))

try(print(
  ggplot(wilcoxon_comparison, aes(x = estimate, y = method)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    geom_pointrange(aes(xmin = conf_low, xmax = conf_high), color = "darkorange", linewidth = 0.9) +
    labs(title = "Score difference (sarcastic - non-sarcastic): unpaired vs. paired",
         subtitle = "Hodges-Lehmann estimate with 95% CI, both in raw score units",
         x = "Estimated score difference", y = NULL) +
    theme_minimal()
))

try(print(
  ggplot(wilcoxon_comparison, aes(x = method, y = effect_size_rank_biserial, fill = method)) +
    geom_col(width = 0.5) + geom_hline(yintercept = 0, color = "grey50") +
    labs(title = "Effect size comparison (rank-biserial correlation)",
         subtitle = "Larger magnitude = stronger association, independent of sample size",
         x = NULL, y = "Rank-biserial r", fill = NULL) +
    theme_minimal() + theme(legend.position = "none")
))

try(print(
  ggplot(diag_data, aes(x = fitted, y = resid)) +
    geom_point(alpha = 0.05) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(se = FALSE, color = "steelblue") +
    labs(title = "Residuals vs. fitted values", x = "Fitted value", y = "Residual") +
    theme_minimal()
))

try(print(
  ggplot(diag_data, aes(sample = resid)) +
    stat_qq(alpha = 0.2) + stat_qq_line(color = "red") +
    labs(title = "Q-Q plot of regression residuals") +
    theme_minimal()
))

try(print(
  ggplot(author_effects, aes(x = effect)) +
    geom_histogram(bins = 60, fill = "purple", alpha = 0.7) +
    labs(title = "Distribution of author-level random effects",
         x = "Author random intercept (signed-log score units)", y = "Number of authors") +
    theme_minimal()
))

try(print(
  ggplot(top_authors, aes(x = effect, y = fct_reorder(author, effect))) +
    geom_point(color = "purple", size = 2) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    labs(title = "15 most extreme author-level effects (excluding deleted accounts)",
         x = "Random intercept (signed-log score units)", y = NULL) +
    theme_minimal()
))

try(print(
  ggplot(subreddit_forest, aes(x = estimate, y = fct_reorder(subreddit, estimate))) +
    geom_point(color = "darkgreen") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    labs(title = "Subreddit fixed effects (relative to reference subreddit)",
         x = "Estimate (signed-log score units)", y = NULL) +
    theme_minimal() + theme(axis.text.y = element_text(size = 7))
))

try(print(
  ggplot(fit_sample, aes(x = predicted, y = observed)) +
    geom_point(alpha = 0.05) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Predicted vs. observed score (signed-log scale)",
         x = "Model-predicted score", y = "Observed score") +
    theme_minimal()
))

dev.off()
cat("Saved", length(list.files("plots", pattern = "\\.png$")), "plots to ./plots/\n")
