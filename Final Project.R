#Import data from Hugging Face
library(readr)
url <- "https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv"
sarcasm <- read_csv(url)

#Filter the dataset to keep only marketing-related subreddits
library(dplyr)
data <- sarcasm %>%
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
View(data)
unique(data$subreddit) #Check unique subreddits  
table(data$label) #Check frequencies of labels => Already balanced 

# Data cleaning — SARC Reddit sarcasm dataset (marketing-related subreddits)
library(dplyr)
library(stringr)
library(tidyr)

# 1. Check for missingness
colSums(is.na(data))
# remove null values
data <- data %>%
  filter(!is.na(comment), str_trim(comment) != "",
         !is.na(parent_comment), str_trim(parent_comment) != "")

# 2. CRITICAL — strip the literal /s sarcasm marker
#    SARC's label comes from the author appending /s. If left in the text,
#    any classifier (feature-based or embedding-based) will just detect the
#    literal marker instead of learning sarcasm. This must be removed
#    unconditionally, from BOTH comment and parent_comment.
strip_s_marker <- function(x) {
  x %>%
    str_remove_all("(?i)\\s*/s\\b\\.?$") %>%   # trailing /s at end of comment
    str_remove_all("(?i)\\s*/s\\s") %>%         # /s occurring mid-text
    str_trim()
}

data <- data %>%
  mutate(
    comment        = strip_s_marker(comment),
    parent_comment  = strip_s_marker(parent_comment)
  )

# Sanity check: literal "/s" should now be gone (allow a handful of false
# positives, e.g. genuine "m/s" units, but the count should be near-zero)
sum(str_detect(data$comment, "(?i)/s\\b"))

# 3. Remove Reddit markup / HTML artifacts (noise, not signal)
clean_markup <- function(x) {
  x %>%
    str_replace_all("&gt;", ">") %>%
    str_replace_all("&lt;", "<") %>%
    str_replace_all("&amp;", "&") %>%
    str_replace_all("&#x200B;", "") %>%          # zero-width space artifact
    str_replace_all("\\[([^\\]]+)\\]\\([^\\)]+\\)", "\\1") %>%  # [text](url) -> text
    str_replace_all("https?://\\S+", "[URL]") %>%  # bare links -> placeholder
    str_replace_all("\\*{1,2}([^\\*]+)\\*{1,2}", "\\1") %>%   # **bold**/*italic* markers
    str_replace_all("\\s+", " ") %>%              # collapse whitespace
    str_trim()
}

data <- data %>%
  mutate(
    comment        = clean_markup(comment),
    parent_comment  = clean_markup(parent_comment)
  )

# 4. Remove duplicate / near-duplicate comments
#    Copypasta and bot replies can inflate feature importance artificially
n_before <- nrow(data)
data <- data %>% distinct(comment, parent_comment, .keep_all = TRUE)
n_before - nrow(data)   # number of exact duplicates removed

# Re-check label balance after all filtering
table(data$label)
prop.table(table(data$label))

#Final check — inspect a few rows to confirm cleaning worked as intended
data %>% select(comment, parent_comment, label) %>% slice_sample(n = 10)

# ============================================================
# RQ1: What features distinguish sarcastic from non-sarcastic comments?
# Input: cleaned `data` object (from clean_sarcasm_data.R)
#        must contain: comment, parent_comment, label, subreddit
# 11 features (within-comment incongruity dropped; sentiment_flip dropped
# as redundant with corrected abs(sentiment_diff); year dropped as a
# control; ?!/has_qe dropped — caused a separation artifact in the
# regression: OR 1347, non-significant p-value, AUC 0.5, zero RF importance)
# ============================================================

library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(forcats)
library(scales)
library(vader)
library(pROC)
library(car)
library(survival)
library(randomForest)

set.seed(1)

# ============================================================
# PART 0 — EXPLORATORY DATA ANALYSIS
# ============================================================

# ensure comment_len exists before any EDA that uses it
# (Part 1 recomputes this too — harmless, just guarantees Part 0 can run standalone)
data <- data %>% mutate(comment_len = str_count(comment, "\\S+"))

# ---- 0.1 Basic structure ----
dim(data)
glimpse(data)
colSums(is.na(data))

# ---- 0.2 Label balance ----
table(data$label)
prop.table(table(data$label))

ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
  geom_bar(fill = "steelblue") +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3) +
  labs(title = "Class balance: sarcastic vs. non-sarcastic", x = NULL, y = "Count") +
  theme_minimal()

# ---- 0.3 Subreddit distribution (top 15) ----
data %>% count(subreddit, sort = TRUE) %>% print(n = 15)

data %>%
  count(subreddit, sort = TRUE) %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = fct_reorder(subreddit, n), y = n)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Top 15 subreddits by comment count", x = NULL, y = "Comments") +
  theme_minimal()

# Check whether label balance holds within subreddits too, not just overall
data %>%
  count(subreddit, label) %>%
  group_by(subreddit) %>%
  mutate(pct = n / sum(n)) %>%
  filter(subreddit %in% (data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 15) %>% pull(subreddit))) %>%
  ggplot(aes(x = subreddit, y = pct, fill = factor(label, labels = c("Not sarcastic","Sarcastic")))) +
  geom_col(position = "fill") +
  coord_flip() +
  labs(title = "Label balance within top 15 subreddits", x = NULL, y = "Proportion", fill = NULL) +
  theme_minimal()

# ---- 0.4 Comment length distribution ----
summary(data$comment_len)
quantile(data$comment_len, probs = c(.01, .05, .25, .5, .75, .95, .99))

ggplot(data, aes(x = comment_len)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "white") +
  coord_cartesian(xlim = c(0, quantile(data$comment_len, .99))) +
  labs(title = "Comment length distribution (word count, 99th pct trimmed for display)",
       x = "Word count", y = "Number of comments") +
  theme_minimal()

ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")),
                 y = comment_len, fill = factor(label))) +
  geom_boxplot(outlier.alpha = 0.1) +
  coord_cartesian(ylim = c(0, quantile(data$comment_len, .99))) +
  labs(title = "Comment length by label", x = NULL, y = "Word count") +
  theme_minimal() + theme(legend.position = "none")

# ---- 0.5 Score distribution (context for RQ3, useful to see now) ----
summary(data$score)
quantile(data$score, probs = c(.01, .05, .25, .5, .75, .95, .99), na.rm = TRUE)

ggplot(data, aes(x = score)) +
  geom_histogram(binwidth = 1, fill = "darkgreen", color = "white") +
  coord_cartesian(xlim = c(-10, 30)) +
  labs(title = "Score distribution (trimmed for display — heavily skewed)",
       x = "Score", y = "Number of comments") +
  theme_minimal()

# ---- 0.6 Duplicate / data quality spot-check ----
n_distinct(data$comment)
nrow(data) - n_distinct(data$comment)   # residual duplicates, if any survived cleaning
n_distinct(data$parent_comment)
n_distinct(data$author)

# ============================================================
# PART 1 — FEATURE ENGINEERING (11 hand-crafted features)
# ============================================================

# ---- ensure length field exists regardless of what the cleaning step produced ----
# comment length is defined as word count (not character count)
data <- data %>%
  mutate(comment_len = str_count(comment, "\\S+"))

# ---- word lists (documented sources — extend/cite as needed) ----
interjections <- c("oh","wow","ugh","ah","aha","hmm","geez","argh","yay",
                   "ouch","phew","hmph","ew","meh","huh","alas","gosh",
                   "damn","dang","yikes")

intensifiers  <- c("very","really","totally","absolutely","extremely",
                   "completely","so","super","incredibly","utterly",
                   "literally","seriously","truly","definitely")

laughter_pattern <- "(?i)\\b(lol+|lmao+|rofl+|ha(ha)+|hehe+|lel+)\\b"

# Manually curated emoticon list (exactly as discussed — extend here if needed)
emoticons <- c(":)", ":-)", ";)", ";-)", ":(", ":-(", ":/", ":-/",
               ":D", ":-D", ":P", ":-P", ":p", ":-p", "XD", "xD",
               "<3", "</3", ":'(")
# Escape regex-special characters ( ) before building the alternation pattern
emoticons_escaped <- str_replace_all(emoticons, "([()])", "\\\\\\1")
emoticon_pattern  <- paste(emoticons_escaped, collapse = "|")
interj_pattern   <- paste0("(?i)\\b(", paste(interjections, collapse = "|"), ")\\b")
intens_pattern   <- paste0("(?i)\\b(", paste(intensifiers, collapse = "|"), ")\\b")

# ---- Feature 1: Capitalization (count) ----
data <- data %>%
  mutate(caps_count = str_count(comment, "\\b[A-Z]{2,}\\b"))

# ---- Feature 2: Comment length (control + feature) ----
data <- data %>% mutate(log_length = log1p(comment_len))

# ---- Feature 3: VADER sentiment (whole comment + parent, for feature 4) ----
# NOTE: score each UNIQUE text once, then join back — many rows share the
# same parent_comment, so scoring every row separately repeats work.
unique_comments <- tibble(comment = unique(data$comment)) %>%
  mutate(vader_comment = sapply(comment, function(x)
    tryCatch(get_vader(x)["compound"], error = function(e) NA_real_)) %>% as.numeric())

unique_parents <- tibble(parent_comment = unique(data$parent_comment)) %>%
  mutate(vader_parent = sapply(parent_comment, function(x)
    tryCatch(get_vader(x)["compound"], error = function(e) NA_real_)) %>% as.numeric())

data <- data %>%
  left_join(unique_comments, by = "comment") %>%
  left_join(unique_parents, by = "parent_comment")

# ---- Feature 4: Cross-comment sentiment incongruity (magnitude, not direction) ----
# abs() is essential here: the hypothesis is that sarcastic comments show a
# LARGER polarity gap with their parent, regardless of which direction it
# shifts. A signed difference would let opposite-direction shifts cancel out.
data <- data %>%
  mutate(sentiment_diff = abs(vader_comment - vader_parent))

# ---- Feature 5: Exclamation (count) ----
data <- data %>% mutate(excl_count = str_count(comment, "!"))

# ---- Feature 6: Ellipsis (count) ----
data <- data %>% mutate(ellipsis_count = str_count(comment, "\\.{3,}|…"))

# ---- Feature 7: Interjections (count) ----
data <- data %>% mutate(interjection_count = str_count(comment, interj_pattern))

# ---- Feature 8: Emoticons (count) ----
data <- data %>% mutate(emoticon_count = str_count(comment, emoticon_pattern))

# ---- Feature 9: Laughter (count) ----
data <- data %>% mutate(laughter_count = str_count(comment, laughter_pattern))

# ---- Feature 10: Quotation marks (count) ----
data <- data %>% mutate(quote_count = str_count(comment, '"') + str_count(comment, "['']"))

# ---- Feature 11: Intensifiers (count) ----
data <- data %>% mutate(intensifier_count = str_count(comment, intens_pattern))

# ---- Prevalence check — confirms sparse features are worth keeping ----
data %>%
  summarise(
    pct_ellipsis    = mean(ellipsis_count > 0) * 100,
    pct_quote       = mean(quote_count > 0) * 100,
    pct_emoticon    = mean(emoticon_count > 0) * 100,
    pct_interject   = mean(interjection_count > 0) * 100,
    pct_intensifier = mean(intensifier_count > 0) * 100,
    pct_laughter    = mean(laughter_count > 0) * 100
  )

# ---- Ratios derived ONLY for descriptive reporting (not fed into regression) ----
data <- data %>%
  mutate(
    caps_ratio        = caps_count / pmax(comment_len, 1),
    excl_ratio         = excl_count / pmax(comment_len, 1),
    ellipsis_ratio     = ellipsis_count / pmax(comment_len, 1),
    interjection_ratio = interjection_count / pmax(comment_len, 1) * 100,
    emoticon_ratio     = emoticon_count / pmax(comment_len, 1) * 100,
    laughter_ratio     = laughter_count / pmax(comment_len, 1) * 100,
    quote_ratio        = quote_count / pmax(comment_len, 1) * 100,
    intensifier_ratio  = intensifier_count / pmax(comment_len, 1) * 100,
    sentiment_category = case_when(
      vader_comment >= 0.05  ~ "positive",
      vader_comment <= -0.05 ~ "negative",
      TRUE                    ~ "neutral"
    )
  )

data <- data %>% dplyr::select(-any_of(c("ups", "downs")))

# ============================================================
# PART 2 — VISUALIZE SENTIMENT DISTRIBUTION & ANSWER "positive or negative?"
# ============================================================

# ---- 1. Full distribution shape by label (boxplot) ----
ggplot(data, aes(x = factor(label, labels = c("Not sarcastic","Sarcastic")),
                 y = vader_comment, fill = factor(label))) +
  geom_boxplot(outlier.alpha = 0.15) +
  labs(title = "Sentiment (VADER compound score) distribution by sarcasm label",
       x = NULL, y = "VADER compound score") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- 2. Direct answer to "positive or negative?" — % positive/negative/neutral by label ----
sent_summary <- data %>%
  count(label, sentiment_category) %>%
  group_by(label) %>%
  mutate(pct = n / sum(n)) %>%
  ungroup() %>%
  mutate(label_txt = factor(label, labels = c("Not sarcastic","Sarcastic")))

ggplot(sent_summary, aes(x = label_txt, y = pct,
                         fill = fct_relevel(sentiment_category, "negative","neutral","positive"))) +
  geom_col(position = "stack") +
  geom_text(aes(label = percent(pct, accuracy = 1)),
            position = position_stack(vjust = 0.5), color = "white", size = 3.5) +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c(negative = "#d73027", neutral = "#999999", positive = "#1a9850"),
                    name = "Sentiment") +
  labs(title = "Sentiment polarity composition: sarcastic vs. non-sarcastic comments",
       x = NULL, y = "% of comments") +
  theme_minimal()

# ---- 3. Diverging bar — the cleanest single chart for "which way does it skew" ----
diverging_data <- sent_summary %>%
  filter(sentiment_category != "neutral") %>%
  mutate(pct_signed = if_else(sentiment_category == "negative", -pct, pct))

ggplot(diverging_data, aes(x = label_txt, y = pct_signed, fill = sentiment_category)) +
  geom_col() +
  geom_hline(yintercept = 0) +
  scale_y_continuous(labels = function(x) percent(abs(x))) +
  scale_fill_manual(values = c(negative = "#d73027", positive = "#1a9850")) +
  coord_flip() +
  labs(title = "Positive vs. negative share by label (neutral excluded)",
       x = NULL, y = "% of comments", fill = "Sentiment") +
  theme_minimal()

# ---- Statistical test backing the visual ----
sent_table <- table(data$label, data$sentiment_category)
sent_table
prop.table(sent_table, margin = 1)
chisq.test(sent_table)

# Continuous check: is the compound score distribution itself shifted?
wilcox.test(vader_comment ~ label, data = data, conf.int = TRUE)
data %>% group_by(label) %>%
  summarise(median_sentiment = median(vader_comment, na.rm = TRUE),
            mean_sentiment    = mean(vader_comment, na.rm = TRUE))

# ============================================================
# PART 3 — DESCRIPTIVE COMPARISON, VISUALIZED + SIGNIFICANCE TESTS
# ============================================================
# Split by feature TYPE, because the zero-inflation in count features
# makes median-based comparison uninformative for them (median = 0 for
# both groups even when the RATE of occurrence clearly differs).
# Density-type features (rates spread continuously): compare via median.
# Sparse features (mostly zero): compare via % of comments containing it.

density_features <- c("caps_ratio","log_length","vader_comment","sentiment_diff")
sparse_features  <- c("excl_ratio","ellipsis_ratio","interjection_ratio",
                      "emoticon_ratio","laughter_ratio","quote_ratio",
                      "intensifier_ratio")
# has_qe dropped entirely — caused a separation artifact in the regression
# (OR 1347, non-significant p-value, AUC 0.5, zero RF importance)

# Underlying raw-count columns for sparse features, used to compute prevalence
sparse_raw_map <- c(excl_ratio = "excl_count", ellipsis_ratio = "ellipsis_count",
                    interjection_ratio = "interjection_count",
                    emoticon_ratio = "emoticon_count",
                    laughter_ratio = "laughter_count",
                    quote_ratio = "quote_count",
                    intensifier_ratio = "intensifier_count")

# ---- Wilcoxon test + effect size, for ANY continuous/ratio feature ----
run_wilcox <- function(feature) {
  f <- as.formula(paste(feature, "~ label"))
  wt <- wilcox.test(f, data = data, conf.int = TRUE)
  n1 <- sum(data$label == 0); n2 <- sum(data$label == 1)
  r_rb <- 1 - (2 * wt$statistic) / (n1 * n2)
  means <- data %>% group_by(label) %>%
    summarise(median = median(.data[[feature]], na.rm = TRUE), .groups = "drop")
  tibble(
    feature = feature,
    median_not_sarcastic = means$median[means$label == 0],
    median_sarcastic      = means$median[means$label == 1],
    p_value = wt$p.value,
    effect_size = as.numeric(r_rb),
    test = "Wilcoxon"
  )
}

# ---- Chi-square + Cramer's V, for prevalence (any feature > 0, or binary) ----
run_prevalence_chisq <- function(feature_label, raw_col) {
  presence <- as.integer(data[[raw_col]] > 0)
  tab <- table(data$label, presence)
  ct <- chisq.test(tab)
  cv <- sqrt(ct$statistic / (sum(tab) * (min(dim(tab)) - 1)))
  props <- prop.table(tab, 1)
  tibble(
    feature = feature_label,
    pct_not_sarcastic = props["0", "1"],
    pct_sarcastic       = props["1", "1"],
    p_value = ct$p.value,
    effect_size = as.numeric(cv),
    test = "Chi-square (prevalence)"
  )
}

descriptive_density <- bind_rows(lapply(density_features, run_wilcox))

descriptive_sparse <- bind_rows(lapply(names(sparse_raw_map), function(rf)
  run_prevalence_chisq(rf, sparse_raw_map[[rf]])))

descriptive_density %>% arrange(p_value)
descriptive_sparse %>% arrange(p_value)

# ============================================================
# PART 3b — VISUALIZE ALL COMPARISONS
# ============================================================

# ---- Density-type features: boxplots, faceted ----
density_long <- data %>%
  select(label, all_of(density_features)) %>%
  pivot_longer(-label, names_to = "feature", values_to = "value") %>%
  mutate(label_txt = factor(label, labels = c("Not sarcastic","Sarcastic")))

ggplot(density_long, aes(x = label_txt, y = value, fill = label_txt)) +
  geom_boxplot(outlier.alpha = 0.1) +
  facet_wrap(~ feature, scales = "free_y") +
  labs(title = "Density-type features by sarcasm label", x = NULL, y = NULL) +
  theme_minimal() +
  theme(legend.position = "none")

# ---- Sparse features: % of comments containing it, faceted bar chart ----
sparse_prevalence_long <- bind_rows(lapply(names(sparse_raw_map), function(feat) {
  data %>%
    mutate(present = as.integer(.data[[sparse_raw_map[[feat]]]] > 0)) %>%
    group_by(label) %>%
    summarise(pct = mean(present), .groups = "drop") %>%
    mutate(feature = feat)
})) %>%
  mutate(label_txt = factor(label, labels = c("Not sarcastic","Sarcastic")))

ggplot(sparse_prevalence_long, aes(x = feature, y = pct, fill = label_txt)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = percent) +
  coord_flip() +
  labs(title = "Sparse features: % of comments containing each marker",
       x = NULL, y = "% of comments", fill = NULL) +
  theme_minimal()

# ============================================================
# PART 3c — UNIFIED SUMMARY TABLE OF ALL COMPARISONS
# ============================================================

summary_table <- bind_rows(
  descriptive_density %>%
    transmute(feature, group_not_sarcastic = median_not_sarcastic,
              group_sarcastic = median_sarcastic, statistic = "median",
              p_value, effect_size, test),
  descriptive_sparse %>%
    transmute(feature, group_not_sarcastic = pct_not_sarcastic,
              group_sarcastic = pct_sarcastic, statistic = "% present",
              p_value, effect_size, test)
) %>%
  arrange(p_value)

summary_table

# ============================================================
# PART 4 — MULTIVARIATE LOGISTIC REGRESSION (main model, full sample)
# ============================================================

count_features <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                    "emoticon_count","laughter_count","quote_count","intensifier_count")

data_scaled <- data %>%
  mutate(across(all_of(c(count_features, "log_length","vader_comment","sentiment_diff")),
                ~ as.numeric(scale(.))))

# Restrict subreddit dummies to top N for tractability
top_subs <- data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 30) %>% pull(subreddit)

main_model <- glm(
  label ~ caps_count + excl_count + ellipsis_count + interjection_count +
    emoticon_count + laughter_count + quote_count +
    intensifier_count + vader_comment + sentiment_diff +
    log_length + factor(subreddit),
  data = data_scaled %>% filter(subreddit %in% top_subs),
  family = binomial
)

summary(main_model)
exp(coef(main_model))          # odds ratios
vif(main_model)                 # multicollinearity check (ignore subreddit dummies' VIF)

# ============================================================
# PART 5 — PAIRED ROBUSTNESS CHECK (conditional logistic regression)
# ============================================================

paired_data <- data_scaled %>%
  group_by(parent_comment) %>%
  filter(n_distinct(label) == 2, n() == 2) %>%
  ungroup()

nrow(paired_data)
n_distinct(paired_data$parent_comment)

clogit_model <- clogit(
  label ~ caps_count + excl_count + ellipsis_count + interjection_count +
    emoticon_count + laughter_count + quote_count +
    intensifier_count + vader_comment + sentiment_diff +
    log_length + strata(parent_comment),
  data = paired_data
)

summary(clogit_model)

# ============================================================
# PART 6 — RANK FEATURES BY DISCRIMINATING POWER
# ============================================================

# Method 1: standardized coefficients from main model
main_coefs <- summary(main_model)$coefficients
feat_names <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                "emoticon_count","laughter_count","quote_count",
                "intensifier_count","vader_comment","sentiment_diff","log_length")

rank_coef <- tibble(
  feature = feat_names,
  abs_coef = abs(main_coefs[feat_names, "Estimate"]),
  odds_ratio = exp(main_coefs[feat_names, "Estimate"]),
  p_value = main_coefs[feat_names, "Pr(>|z|)"]
) %>% arrange(desc(abs_coef))

# Method 2: univariate AUC per feature (robust to collinearity)
auc_features <- c(count_features, "log_length","vader_comment","sentiment_diff")

auc_ranking <- sapply(auc_features, function(f) {
  roc_obj <- roc(data$label, data[[f]], quiet = TRUE)
  as.numeric(auc(roc_obj))
})
rank_auc <- tibble(feature = names(auc_ranking), auc = auc_ranking) %>% arrange(desc(auc))

# Method 3: random forest variable importance
rf_data <- data_scaled %>%
  select(label, all_of(feat_names)) %>%
  mutate(label = as.factor(label)) %>%
  na.omit()

rf_model <- randomForest(label ~ ., data = rf_data, importance = TRUE)
rank_rf <- importance(rf_model, type = 1) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("feature") %>%
  arrange(desc(MeanDecreaseAccuracy))

# ---- Combined ranking table ----
final_ranking <- rank_coef %>%
  select(feature, abs_coef, odds_ratio, p_value) %>%
  left_join(rank_auc, by = "feature") %>%
  left_join(rank_rf %>% select(feature, MeanDecreaseAccuracy), by = "feature") %>%
  arrange(desc(abs_coef))

final_ranking

# ============================================================
# PART 7 — DO THE TWO LABELED GROUPS FORM SEPARATE CLUSTERS?
# ============================================================
# LDA finds the single best linear combination of the 11 features for
# separating sarcastic vs. non-sarcastic. If the two groups form distinct
# clusters, their projected scores will show little overlap below.

library(MASS)

# NOTE: MASS defines its own select() which silently overrides dplyr::select()
# for the REST OF THIS R SESSION, not just this script — if you later re-run
# an earlier script (e.g. clean_sarcasm_data.R) in the same session, its
# plain select() calls will break too with an "unused arguments" error.
# Restoring dplyr's version here prevents that regardless of run order.
select <- dplyr::select

lda_input <- data_scaled %>%
  select(label, all_of(feat_names)) %>%
  na.omit()

lda_model <- lda(factor(label) ~ ., data = lda_input)
lda_scores <- predict(lda_model)$x[, 1]

lda_plot_data <- tibble(
  LD1 = lda_scores,
  label_txt = factor(lda_input$label, labels = c("Not sarcastic","Sarcastic"))
)

# Simple, direct answer: do the two distributions separate or overlap?
ggplot(lda_plot_data, aes(x = LD1, fill = label_txt)) +
  geom_density(alpha = 0.5) +
  labs(title = "Are sarcastic and non-sarcastic comments separate clusters?",
       subtitle = "Best-case linear separation using all 11 features",
       x = "Discriminant score", y = "Density", fill = NULL) +
  theme_minimal()

# Same result, shown as actual data points rather than a smoothed curve —
# note LDA with 2 classes only ever produces ONE axis (classes - 1), so
# points are jittered vertically just to avoid overplotting, not because
# there's a second real dimension.
lda_plot_sample <- lda_plot_data %>% slice_sample(n = min(5000, nrow(lda_plot_data)))

ggplot(lda_plot_sample, aes(x = LD1, y = label_txt, color = label_txt)) +
  geom_jitter(height = 0.2, alpha = 0.3, size = 1) +
  labs(title = "Individual comments along the discriminant axis (5,000-point sample)",
       x = "Discriminant score", y = NULL, color = NULL) +
  theme_minimal() +
  theme(legend.position = "none")

# Concrete number to go with the picture: how well would this line classify?
lda_pred <- predict(lda_model)$class
table(Predicted = lda_pred, Actual = lda_input$label)
mean(lda_pred == lda_input$label)   # overall accuracy of the best-case linear split

# ============================================================
# PART 7b — 2D SCATTER PLOT (PCA) — actual point clusters in 2 dimensions
# ============================================================
# LDA only gives ONE axis with 2 classes, so it can't produce a true 2D
# scatter. PCA gives multiple axes regardless of class count, letting us
# plot PC1 vs PC2 as an actual 2D point cloud, colored by label afterward.
# NOTE: PCA doesn't use the label when computing these axes — it's an
# exploratory look at feature-space structure, not a supervised result.

pca_result <- prcomp(lda_input %>% dplyr::select(-label), center = TRUE, scale. = TRUE)

pca_scores <- as_tibble(pca_result$x[, 1:2]) %>%
  mutate(label_txt = factor(lda_input$label, labels = c("Not sarcastic","Sarcastic")))

pca_scores_sample <- pca_scores %>% slice_sample(n = min(5000, nrow(pca_scores)))

ggplot(pca_scores_sample, aes(x = PC1, y = PC2, color = label_txt)) +
  geom_point(alpha = 0.3, size = 1) +
  labs(title = "PCA scatter — do the two labels form visible clusters?",
       subtitle = "5,000-point sample, PC1 vs PC2",
       color = NULL) +
  theme_minimal()

# % of total variance these two axes actually capture — important context:
# if this is low, a lot of the feature space isn't shown in this 2D view
summary(pca_result)$importance[2, 1:2]