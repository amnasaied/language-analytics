## ==================================================================
## RQ2: Do sarcastic comments receive higher public approval (score)
##      than non-sarcastic comments, and does this hold up once we
##      control for comment length and subreddit?
##
## Structure of this script:
##   1. Setup (packages)
##   2. Load data
##   3. Vote-data quality check (why score is the only measure)
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
library(lubridate)
library(scales)

if (!requireNamespace("patchwork", quietly = TRUE)) install.packages("patchwork")
library(patchwork)   # combine ggplot panels into one figure

if (!requireNamespace("ggridges", quietly = TRUE)) install.packages("ggridges")
library(ggridges)    # ridgeline distributions (cleaner than overlapping density)

if (!requireNamespace("ggpubr", quietly = TRUE)) install.packages("ggpubr")
library(ggpubr)      # significance brackets on the group-comparison plot


# ==================================================================
# 2. LOAD DATA
# ==================================================================
#Import data from Hugging Face
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
# 3. VOTE-DATA QUALITY CHECK: WHY score IS THE ONLY APPROVAL MEASURE
# --------------------------------------------------------------
# The export carries three vote columns -- score, ups, downs -- and it
# is tempting to read ups/downs as independent corroboration of score.
# They are not. Reddit stopped exposing separate up/down tallies around
# 2014 (vote fuzzing, to frustrate bots), so the API returns only the
# net score plus filler values. This scrape spans that change, so every
# row falls into one of exactly two regimes with nothing in between:
#
#   real data    ups == score, downs == 0     comments up to 2016-09
#   placeholder  ups == -1,    downs == -1    comments from 2016-10 on
#
# The three checks below are the evidence for that reading, and they
# are what justifies using score as the single measure of public
# approval in this RQ. Report them in the methods section -- the
# columns are dropped at the end of this block, so nothing downstream
# can quietly reintroduce them.
# ==================================================================

vote_regimes <- sarcasm %>%
  mutate(regime = case_when(
    ups == -1 & downs == -1     ~ "placeholder (-1 / -1)",
    downs == 0 & ups == score   ~ "real (ups == score, downs == 0)",
    TRUE                        ~ "other"
  ))

cat("\n===== Vote-data regimes =====\n")
print(
  vote_regimes %>%
    group_by(regime) %>%
    summarise(n = n(), pct = round(100 * n() / nrow(vote_regimes), 1),
              first_month = min(date), last_month = max(date),
              .groups = "drop")
)

# --- Check 1: the identity score = ups - downs ----------------------
# If these were real tallies the identity would hold everywhere. It
# holds in every real-data row and almost nowhere else -- and the few
# placeholder rows where it "holds" only do so because -1 - (-1) = 0
# coincides with a score of 0.
cat("\n--- Check 1: does score == ups - downs? ---\n")
print(
  vote_regimes %>%
    group_by(regime) %>%
    summarise(n = n(),
              pct_identity_holds = round(100 * mean(score == ups - downs), 1),
              .groups = "drop")
)

# --- Check 2: counts that go negative -------------------------------
# A tally of votes cast cannot be below zero. Both columns are.
cat("\n--- Check 2: negative 'counts' ---\n")
cat(sprintf("rows with downs < 0: %d (%.1f%%)   |   rows with ups < 0: %d (%.1f%%)\n",
            sum(sarcasm$downs < 0), 100 * mean(sarcasm$downs < 0),
            sum(sarcasm$ups   < 0), 100 * mean(sarcasm$ups   < 0)))

# --- Check 3: no downvote is ever recorded --------------------------
# max(downs) is 0 across the whole corpus, yet thousands of comments
# carry a negative score -- which by definition requires downvotes.
# The column does not record downvotes at all.
cat("\n--- Check 3: downvotes that must exist but are never recorded ---\n")
cat(sprintf("max(downs) = %d   |   comments with score < 0: %d (%.1f%%)\n",
            max(sarcasm$downs),
            sum(sarcasm$score < 0), 100 * mean(sarcasm$score < 0)))

# --- Redundancy of ups where it IS real -----------------------------
cat("\n--- Where ups is real, it is a copy of score ---\n")
cat(sprintf("ups == score in %.1f%% of rows   |   cor(ups, score) = %.3f\n",
            100 * mean(sarcasm$ups == sarcasm$score),
            cor(sarcasm$ups, sarcasm$score)))

# DECISION -- both columns are dropped here, for two distinct reasons:
#
#   downs  carries no signal at all: it is <= 0 in 100% of rows and
#          never records a single downvote.
#   ups    adds nothing score does not already carry. In the 86% of
#          rows where it is real it is an exact copy of score; in the
#          other 14% it is a -1 sentinel. Averaging the two together
#          produces a number that is neither -- which is precisely why
#          mean ups (5.16 / 6.13) sits BELOW mean score (6.59 / 7.32)
#          despite "ups" supposedly counting only positive votes.
#
# The missingness is also not noise: it is a clean time slice, with a
# hard cutoff between 2016-09 and 2016-10 (see the regime table above;
# the file is not sorted by date, which is why it looks interleaved).
# Restricting to rows with real vote data would therefore silently
# drop the last three months of the corpus. score covers the full
# range 2009-09 to 2016-12 and is the measure used from here on.
sarcasm <- sarcasm %>% select(-ups, -downs)


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

# Score by group. ups/downs are deliberately absent -- see section 3.
cat("\n===== Score Statistics =====\n")
print(
  sarcasm_clean %>%
    group_by(label_f) %>%
    summarise(n            = n(),
              mean_score   = mean(score),
              sd_score     = sd(score),
              median_score = median(score),
              .groups = "drop")
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
# Right: mean score by group, with SE and the Wilcoxon-test
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
  summarise(Mean = mean(score), SE = sd(score) / sqrt(n()), .groups = "drop")

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
  labs(title = "Mean Score", subtitle = "Wilcoxon test significance shown",
       x = NULL, y = "Average Score") +
  scale_fill_manual(values = pal) +
  theme_report + theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

(p1a | p1b) +
  plot_annotation(
    title = "Do Sarcastic Comments Score Higher?",
    subtitle = "score is the only vote measure used: ups is a copy of it or a -1 sentinel (section 3)"
  )


# --- FIGURE 2 (combined panel) --------------------------------------
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


# --- FIGURE 3 (standalone) -------------------------------------------
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


# --- FIGURE 4 (combined panel) ---------------------------------------
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


# --- FIGURE 5 (standalone) --------------------------------------------
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
