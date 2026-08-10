## ==================================================================
## RQ2: Do sarcastic comments receive higher public approval (score)
##      than non-sarcastic comments, and does this hold up once we
##      control for comment length and subreddit?
##
## Structure of this script:
##   1. Setup (packages)
##   2. Load data
##   3. Data quality check (ups/downs)
##   4. Data cleaning & feature engineering
##   5. Descriptive statistics
##   6. Statistical tests (group comparisons)
##   7. Regression models (control variables)
##   8. Figures
## ==================================================================


# ==================================================================
# 1. SETUP
# ==================================================================
library(tidyverse)
library(skimr)
library(lubridate)
library(scales)

if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")
library(patchwork)   # combine ggplot panels into one figure

if (!requireNamespace("ggridges", quietly = TRUE)) install.packages("ggridges")
library(ggridges)    # ridgeline distributions (cleaner than overlapping density)

if (!requireNamespace("ggpubr", quietly = TRUE)) install.packages("ggpubr")
library(ggpubr)      # built-in stat annotations (correlation, p-values) on plots

if (!requireNamespace("hexbin", quietly = TRUE)) install.packages("hexbin")
library(hexbin)      # required by geom_hex()


# ==================================================================
# 2. LOAD DATA
# ==================================================================
#Import data from Hugging Face
library(readr)
url <- "https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv"
sarcasm <- read_csv(url)

#Filter the dataset to keep only marketing-related subreddits
sarcasm <- sarcasm %>%
  filter(subreddit %in% c("apple",
                          "iphone",
                          "Android",
                          "GooglePixel",
                          "AndroidMasterRace",
                          "windowsphone",
                          "Surface",
                          "GalaxyNote7",
                          "galaxynote4",
                          "lgv20",
                          "pebble",
                          "nvidia",
                          "intel",
                          "Amd",
                          "razer",
                          "hardware",
                          "techsupport",
                          "Steam",
                          "playstation",
                          "PS4",
                          "PS4Pro",
                          "xboxone",
                          "NintendoSwitch",
                          "NintendoNX",
                          "wiiu",
                          "askcarsales",
                          "cars",
                          "Autos",
                          "BMW",
                          "Volkswagen",
                          "SubaruForester",
                          "subaru",
                          "Miata",
                          "FocusST",
                          "Datsun",
                          "streetwear",
                          "StreetwearSales",
                          "sneakermarket",
                          "Sneakers",
                          "FashionReps",
                          "supremeclothing",
                          "bapeheads",
                          "goodyearwelt",
                          "frugalmalefashion",
                          "BeautyBoxes",
                          "MakeupAddiction",
                          "walmart",
                          "starbucks",
                          "tacobell",
                          "TalesFromRetail",
                          "Justrolledintotheshop",
                          "Random_Acts_Of_Amazon",
                          "netflix",
                          "boxoffice",
                          "moviecritic",
                          "television",
                          "movies",
                          "music",
                          "headphones",
                          "GameDeals",
                          "buildapc",
                          "buildapcsales",
                          "pcmasterrace"))
View(sarcasm)
unique(sarcasm$subreddit) #Check unique subreddits  
table(sarcasm$label) #Check frequencies of labels => Already balanced 

cat("Rows:", nrow(sarcasm), "\n")
cat("Columns:", ncol(sarcasm), "\n")
glimpse(sarcasm)


# ==================================================================
# 3. DATA QUALITY CHECK: ups / downs
# --------------------------------------------------------------
# This dataset is a well-known Kaggle export of Reddit comments.
# Reddit's API has for years returned fuzzed or placeholder vote
# data: "ups" is frequently identical to "score", and "downs" is
# frequently a non-informative placeholder (0 or -1) rather than a
# true downvote count. Run this check BEFORE trusting any
# ups/downs comparison.
# ==================================================================
quality_check <- sarcasm %>%
  summarise(
    pct_downs_zero_or_neg = mean(downs <= 0, na.rm = TRUE) * 100,
    pct_ups_equals_score  = mean(ups == score, na.rm = TRUE) * 100,
    cor_ups_score         = cor(ups, score, use = "complete.obs")
  )

cat("\n--- Vote Data Quality Check ---\n")
cat("Percent of rows where downs <= 0:   ", round(quality_check$pct_downs_zero_or_neg, 1), "%\n")
cat("Percent of rows where ups == score: ", round(quality_check$pct_ups_equals_score, 1), "%\n")
cat("Correlation between ups and score:  ", round(quality_check$cor_ups_score, 3), "\n")
cat("If these percentages are high, treat 'downs' as unreliable and\n")
cat("interpret 'ups' as effectively a duplicate of 'score', not new info.\n\n")

# DECISION POINT based on this check:
# If pct_downs_zero_or_neg is ~100%, 'downs' has essentially zero
# variance in this dataset and carries no usable signal -- it is
# dropped from all analysis and visualization below.
# 'ups' is kept only as a robustness check against 'score', since
# it is highly (but not perfectly) redundant with it.


# ==================================================================
# 4. DATA CLEANING & FEATURE ENGINEERING
# ==================================================================

# Remove observations with missing values, and create a readable
# factor for label so every table/model/plot below can reuse it
# directly (avoids repeating factor(label) everywhere).
sarcasm_clean <- sarcasm %>%
  filter(!is.na(score), !is.na(label)) %>%
  mutate(label_f = factor(label, levels = c(0, 1),
                          labels = c("Non-sarcastic", "Sarcastic")))

# Control variable: comment length, plus a log-transformed score
# for the robustness-check regression model further down.
sarcasm_clean <- sarcasm_clean %>%
  filter(!is.na(comment)) %>%
  mutate(
    comment_length = nchar(comment),
    log_score      = log1p(pmax(score, 0))
  )

# Shared colorblind-safe palette (Okabe-Ito) and theme so every
# panel looks consistent when combined with patchwork later on.
pal <- c("Non-sarcastic" = "#0072B2", "Sarcastic" = "#D55E00")
theme_report <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))


# ==================================================================
# 5. DESCRIPTIVE STATISTICS
# ==================================================================

# Score, ups, and downs by group
sarcasm_clean %>%
  group_by(label_f) %>%
  summarise(
    n            = n(),
    mean_score   = mean(score),   sd_score = sd(score),   median_score = median(score),
    mean_ups     = mean(ups),     sd_ups   = sd(ups),
    mean_downs   = mean(downs),   sd_downs = sd(downs)
  )

# Comment length by group
cat("\n===== Comment Length Statistics =====\n")
print(
  sarcasm_clean %>%
    group_by(label_f) %>%
    summarise(mean_length   = mean(comment_length),
              median_length = median(comment_length),
              sd_length     = sd(comment_length),
              .groups = "drop")
)


# ==================================================================
# 6. STATISTICAL TESTS (group comparisons)
# ==================================================================
# Research Question:
# Do sarcastic comments generally receive higher public approval
# (scores) than non-sarcastic comments?
# ==================================================================

# --- Score: one-sided Welch's t-test -------------------------------
# H0: Mean score of non-sarcastic comments >= sarcastic comments
# H1: Mean score of sarcastic comments > non-sarcastic comments
t_test_result <- t.test(
  score ~ label,
  data = sarcasm_clean,
  alternative = "less",
  var.equal = FALSE
)
print(t_test_result)

# --- Score: Wilcoxon rank-sum test (robustness check) --------------
# Distribution-free, less sensitive to the heavy skew/outliers
# visible in score.
wilcox_result <- wilcox.test(
  score ~ label,
  data = sarcasm_clean,
  alternative = "less"
)
print(wilcox_result)

# --- Score: effect size (Cohen's d) ---------------------------------
# Important given n > 1,000,000 -- statistical significance alone
# will be nearly guaranteed here.
group_stats <- sarcasm_clean %>%
  group_by(label) %>%
  summarise(mean = mean(score), sd = sd(score), n = n())

pooled_sd <- sqrt((group_stats$sd[1]^2 + group_stats$sd[2]^2) / 2)
cohens_d <- (group_stats$mean[2] - group_stats$mean[1]) / pooled_sd
cat("\nCohen's d (sarcastic - non-sarcastic):", round(cohens_d, 4), "\n")
cat("(~0.2 = small, ~0.5 = medium, ~0.8 = large effect)\n\n")

# --- Comment length: Welch's t-test & Wilcoxon (control check) -----
cat("\nWelch t-test (comment length):\n")
print(t.test(comment_length ~ label, data = sarcasm_clean))

cat("\nWilcoxon test (comment length):\n")
print(wilcox.test(comment_length ~ label, data = sarcasm_clean))


# ==================================================================
# 7. REGRESSION MODELS (control variables)
# ==================================================================

cat("\nRegression: score ~ sarcasm + comment length\n")
model1 <- lm(score ~ label + comment_length, data = sarcasm_clean)
print(summary(model1))

cat("\nRegression: score ~ sarcasm + comment length + subreddit\n")
model2 <- lm(score ~ label + comment_length + subreddit, data = sarcasm_clean)
print(summary(model2))

cat("\nRegression: log(score) ~ sarcasm + comment length\n")
model3 <- lm(log_score ~ label + comment_length, data = sarcasm_clean)
print(summary(model3))


# ==================================================================
# 8. FIGURES
# ==================================================================

# --- FIGURE 1 (combined panel) --------------------------------------
# Left:  ridgeline distribution of scores by group
# Right: mean score & ups by group, with SE and the Wilcoxon-test
#        significance bracket annotated.
# Merged with patchwork so both are read as one figure.
p1a <- ggplot(sarcasm_clean, aes(x = score, y = label_f, fill = label_f)) +
  geom_density_ridges(alpha = 0.75, color = "white", scale = 1.2) +
  coord_cartesian(xlim = c(-20, 150)) +
  scale_fill_manual(values = pal) +
  labs(title = "Score Distribution", x = "Reddit Score", y = NULL) +
  theme_report + theme(legend.position = "none")

summary_scores <- sarcasm_clean %>%
  group_by(label_f) %>%
  summarise(
    across(c(score, ups),
           list(Mean = ~mean(.x), SE = ~sd(.x) / sqrt(n())),
           .names = "{.col}_{.fn}")
  ) %>%
  pivot_longer(-label_f, names_to = c("metric", ".value"),
               names_pattern = "(.*)_(Mean|SE)")

p1b <- ggplot(summary_scores, aes(label_f, Mean, fill = label_f)) +
  geom_col(width = .6) +
  geom_errorbar(aes(ymin = Mean - SE, ymax = Mean + SE), width = .15) +
  stat_pvalue_manual(
    data = tibble(group1 = "Non-sarcastic", group2 = "Sarcastic",
                  y.position = max(summary_scores$Mean + summary_scores$SE) * 1.15,
                  p.signif = ifelse(wilcox_result$p.value < 0.001, "***",
                                    ifelse(wilcox_result$p.value < 0.01, "**",
                                           ifelse(wilcox_result$p.value < 0.05, "*", "ns")))),
    label = "p.signif", tip.length = 0.02
  ) +
  facet_wrap(~ metric, scales = "free_y",
             labeller = as_labeller(c(score = "Score", ups = "Ups"))) +
  labs(title = "Mean Score & Ups", subtitle = "Wilcoxon test significance shown", x = NULL, y = "Average Value") +
  scale_fill_manual(values = pal) +
  theme_report + theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

(p1a | p1b) +
  plot_annotation(
    title = "Do Sarcastic Comments Score Higher?",
    subtitle = "downs excluded: constant (<=0) for 100% of comments, no usable signal"
  )


# --- FIGURE 2 (combined panel) --------------------------------------
# ups vs. score relationship, using a hexbin density heatmap instead
# of a raw scatterplot -- with 1M+ rows, a scatter is just an
# overplotted blob; hexbin actually shows where the mass of points
# sits.
p2 <- ggplot(sarcasm_clean, aes(score, ups)) +
  geom_hex(bins = 60) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "white", linewidth = 0.6) +
  coord_cartesian(xlim = c(-20, 150), ylim = c(-20, 150)) +
  scale_fill_viridis_c(trans = "log10", labels = label_comma()) +
  stat_cor(aes(label = after_stat(r.label)), method = "pearson", color = "white", label.x = -15, label.y = 140) +
  labs(title = "Ups vs. Score Density", subtitle = "Dashed line = perfect equality (ups == score)",
       x = "Score", y = "Ups", fill = "Comments\n(log scale)") +
  theme_report

p2


# --- FIGURE 3 (combined panel) --------------------------------------
# Left:  subreddits with highest average score
# Right: subreddits with highest sarcasm rate
# Merged side-by-side to compare "approval leaders" vs "sarcasm
# leaders" at a glance -- same subreddit universe, two different
# rankings.
top_subs <- sarcasm_clean %>%
  group_by(subreddit) %>%
  summarise(avg_score = mean(score), n = n()) %>%
  filter(n > 1000) %>%
  arrange(desc(avg_score)) %>%
  slice_head(n = 15)

p3a <- ggplot(top_subs, aes(reorder(subreddit, avg_score), avg_score)) +
  geom_col(aes(fill = avg_score), show.legend = FALSE) +
  coord_flip() +
  scale_fill_viridis_c(option = "mako", direction = -1) +
  labs(title = "Highest Avg. Score", x = NULL, y = "Average Score") +
  theme_report

sarcasm_rate <- sarcasm_clean %>%
  group_by(subreddit) %>%
  summarise(SarcasmRate = mean(label), n = n()) %>%
  filter(n > 1000) %>%
  arrange(desc(SarcasmRate)) %>%
  slice_head(n = 15)

p3b <- ggplot(sarcasm_rate, aes(reorder(subreddit, SarcasmRate), SarcasmRate)) +
  geom_col(aes(fill = SarcasmRate), show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(labels = percent) +
  scale_fill_viridis_c(option = "inferno", direction = -1) +
  labs(title = "Highest Sarcasm Rate", x = NULL, y = "Sarcasm Rate") +
  theme_report

(p3a | p3b) +
  plot_annotation(title = "Subreddit Rankings", subtitle = "Subreddits with > 1,000 comments only")


# --- FIGURE 4 (standalone) -------------------------------------------
# Average score by subreddit and sarcasm, most active subreddits
subreddit_scores <- sarcasm_clean %>%
  group_by(subreddit, label_f) %>%
  summarise(avg_score = mean(score), n = n(), .groups = "drop") %>%
  filter(n > 500)

top_subs_active <- subreddit_scores %>%
  group_by(subreddit) %>%
  summarise(total = sum(n)) %>%
  slice_max(total, n = 15)

plot_data <- subreddit_scores %>%
  filter(subreddit %in% top_subs_active$subreddit)

ggplot(plot_data, aes(reorder(subreddit, avg_score), avg_score, fill = label_f)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Average Score by Subreddit and Sarcasm", x = NULL, y = "Average Score", fill = "Comment Type") +
  scale_fill_manual(values = pal) +
  theme_report


# --- FIGURE 5 (combined panel) ---------------------------------------
# Top:    average score over time, with a loess smoother
# Bottom: monthly comment volume by group, sharing the same x-axis
#         -- lets you see whether swings in the top panel line up
#         with low-volume months.
monthly <- sarcasm_clean %>%
  mutate(month = ym(date)) %>%
  group_by(month, label_f) %>%
  summarise(avg_score = mean(score), n = n(), .groups = "drop")

p5a <- ggplot(monthly, aes(month, avg_score, color = label_f)) +
  geom_line(linewidth = 0.6, alpha = 0.5) +
  geom_smooth(se = FALSE, linewidth = 1.2, method = "loess", span = 0.3) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_color_manual(values = pal) +
  labs(title = "Average Reddit Score Over Time", x = NULL, y = "Average Score", color = NULL) +
  theme_report

p5b <- ggplot(monthly, aes(month, n, fill = label_f)) +
  geom_col(position = "stack") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_fill_manual(values = pal) +
  labs(x = "Month", y = "Comments", fill = NULL) +
  theme_report

p5a / p5b + plot_layout(heights = c(2, 1), guides = "collect")


# --- FIGURE 6 (standalone) --------------------------------------------
# Comment length vs. score, by sarcasm group -- visual companion to
# the control-variable regression models above.
ggplot(sarcasm_clean, aes(comment_length, score, color = label_f)) +
  geom_point(alpha = .02) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = pal) +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Comment Length vs Reddit Score",
       x = "Comment Length (characters)",
       y = "Score",
       color = "Comment Type") +
  theme_report
