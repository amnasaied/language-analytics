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
# PART 1b — FEATURE AVAILABILITY (PREVALENCE) VISUALS
# ------------------------------------------------------------
# "Availability" = does a comment CONTAIN the feature at all (count > 0),
# not how strong it is. Only the presence/absence markers apply here.
# Sentiment, length and sentiment_diff are continuous (effectively always
# present), so they are NOT shown as have/not-have — they belong in the
# distribution comparisons (Parts 2 & 3).
# ============================================================

# marker features + human-readable labels (order = display order)
marker_cols <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                 "emoticon_count","laughter_count","quote_count","intensifier_count")
marker_labels <- c(caps_count = "Capitalization", excl_count = "Exclamation",
                   ellipsis_count = "Ellipsis", interjection_count = "Interjections",
                   emoticon_count = "Emoticons", laughter_count = "Laughter",
                   quote_count = "Quotation marks", intensifier_count = "Intensifiers")

# ---- 1b.1 Overall: % of comments that HAVE vs do NOT have each feature ----
overall_presence <- data %>%
  summarise(across(all_of(marker_cols), ~ mean(.x > 0) * 100)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "Has feature") %>%
  mutate(`No feature` = 100 - `Has feature`,
         feature = marker_labels[feature])

# order features by how common they are (most available at the top)
order_lv <- overall_presence %>% arrange(`Has feature`) %>% pull(feature)

overall_long <- overall_presence %>%
  pivot_longer(c(`Has feature`, `No feature`),
               names_to = "status", values_to = "pct") %>%
  mutate(feature = factor(feature, levels = order_lv),
         status  = factor(status, levels = c("No feature", "Has feature")))

ggplot(overall_long, aes(x = feature, y = pct, fill = status)) +
  geom_col(width = 0.75) +
  geom_text(data = subset(overall_long, status == "Has feature"),
            aes(label = sprintf("%.1f%%", pct)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_y_continuous(labels = label_percent(scale = 1), limits = c(0, 100)) +
  scale_fill_manual(values = c("No feature" = "grey80", "Has feature" = "#2c7fb8")) +
  labs(title = "Feature availability across all comments",
       subtitle = "% of comments that contain each feature vs. that do not",
       x = NULL, y = "% of comments", fill = NULL) +
  theme_minimal()

# ---- 1b.2 Availability by class: sarcastic vs not sarcastic ----
by_label <- data %>%
  group_by(label) %>%
  summarise(across(all_of(marker_cols), ~ mean(.x > 0) * 100), .groups = "drop") %>%
  pivot_longer(-label, names_to = "feature", values_to = "pct_present") %>%
  mutate(feature   = factor(marker_labels[feature], levels = order_lv),
         label_txt = factor(label, labels = c("Not sarcastic", "Sarcastic")))

ggplot(by_label, aes(x = feature, y = pct_present, fill = label_txt)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  coord_flip() +
  scale_y_continuous(labels = label_percent(scale = 1)) +
  scale_fill_manual(values = c("Not sarcastic" = "grey65", "Sarcastic" = "#d95f0e")) +
  labs(title = "Feature availability: sarcastic vs. not sarcastic",
       subtitle = "% of comments containing each feature, by class",
       x = NULL, y = "% of comments containing feature", fill = NULL) +
  theme_minimal()

# ---- 1b.3 Underlying numbers (sorted by biggest sarcastic-vs-not gap) ----
availability_table <- by_label %>%
  tidyr::pivot_wider(id_cols = feature,
                     names_from = label_txt, values_from = pct_present) %>%
  mutate(diff_pp = Sarcastic - `Not sarcastic`) %>%
  arrange(desc(abs(diff_pp)))
print(availability_table)

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

# Benjamini-Hochberg FDR correction — controls false positives across the
# multiple feature tests (judge significance on p_adjusted, not raw p_value)
descriptive_density <- descriptive_density %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH"))
descriptive_sparse <- descriptive_sparse %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH"))

descriptive_density %>% arrange(p_adjusted)
descriptive_sparse %>% arrange(p_adjusted)

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
              p_value, p_adjusted, effect_size, test),
  descriptive_sparse %>%
    transmute(feature, group_not_sarcastic = pct_not_sarcastic,
              group_sarcastic = pct_sarcastic, statistic = "% present",
              p_value, p_adjusted, effect_size, test)
) %>%
  arrange(p_adjusted)

summary_table

# ============================================================
# PART 4 — MULTIVARIATE LOGISTIC REGRESSION (main model, full sample)
# ============================================================

count_features <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                    "emoticon_count","laughter_count","quote_count","intensifier_count")

data_scaled <- data %>%
  mutate(across(all_of(c(count_features, "log_length","vader_comment","sentiment_diff")),
                ~ as.numeric(scale(.))))

# Keep the full corpus but stabilise subreddit control: the top-30 subreddits
# keep their own dummy; the long tail is pooled into "Other". This avoids
# unstable tiny-subreddit dummies WITHOUT dropping any rows, so the model is
# estimated on the same full sample as the univariate/paired analyses.
top_subs <- data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 30) %>% pull(subreddit)

main_model <- glm(
  label ~ caps_count + excl_count + ellipsis_count + interjection_count +
    emoticon_count + laughter_count + quote_count +
    intensifier_count + vader_comment + sentiment_diff +
    log_length + factor(subreddit),
  data = data_scaled %>%
    mutate(subreddit = if_else(subreddit %in% top_subs, subreddit, "Other")),
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
# PART 6 — RANK FEATURES BY DISCRIMINATING POWER (inferential)
# ============================================================
# Ranked by standardized effect (odds ratio per 1 SD) from the main model,
# with FDR-adjusted significance. This is an explanatory ranking — predictive
# performance of the features is evaluated separately under RQ2.

main_coefs <- summary(main_model)$coefficients
feat_names <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                "emoticon_count","laughter_count","quote_count",
                "intensifier_count","vader_comment","sentiment_diff","log_length")

final_ranking <- tibble(
  feature    = feat_names,
  odds_ratio = exp(main_coefs[feat_names, "Estimate"]),   # per 1 SD (scaled predictors)
  abs_log_or = abs(main_coefs[feat_names, "Estimate"]),   # magnitude, for ranking
  p_value    = main_coefs[feat_names, "Pr(>|z|)"]
) %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH")) %>%
  arrange(desc(abs_log_or))

final_ranking

# ============================================================
# PART 6b — HYPOTHESIS SYNTHESIS (RQ1 VERDICT TABLE)
# ============================================================
# One row per PRE-SPECIFIED feature hypothesis. Compares the theory-expected
# direction against the observed univariate (Part 3) and multivariate (Part 4)
# evidence, with the paired model (Part 5) as a robustness flag. Verdict is
# directional-confirmatory (Confirmed / Reversed / Not supported / Mixed);
# Strength conveys magnitude separately. vader_comment (no directional
# hypothesis) and log_length (control) are reported in Part 4 but not judged here.

# 1. Pre-specified hypotheses: expected sign (+1 = more in sarcastic) + source
hypotheses <- tribble(
  ~feature,                ~model_term,          ~desc_row,            ~expected, ~source,
  "Exclamation",           "excl_count",         "excl_ratio",          1, "Gonzalez-Ibanez 2011; Hutto 2014",
  "Capitalization",        "caps_count",         "caps_ratio",          1, "Hutto 2014",
  "Ellipsis",              "ellipsis_count",     "ellipsis_ratio",      1, "Gonzalez-Ibanez 2011",
  "Interjections",         "interjection_count", "interjection_ratio",  1, "Kreuz & Caucci 2007",
  "Intensifiers",          "intensifier_count",  "intensifier_ratio",   1, "Khodak 2018 (SARC)",
  "Quotation marks",       "quote_count",        "quote_ratio",         1, "Gonzalez-Ibanez 2011",
  "Emoticons",             "emoticon_count",     "emoticon_ratio",     -1, "Gonzalez-Ibanez 2011; SARC",
  "Laughter",              "laughter_count",     "laughter_ratio",     -1, "Gonzalez-Ibanez 2011; SARC",
  "Sentiment incongruity", "sentiment_diff",     "sentiment_diff",      1, "Riloff 2013; Joshi 2016"
)

# 2. Observed univariate evidence (effect size, FDR p, direction) from Part 3
univ <- bind_rows(
  descriptive_sparse  %>% transmute(desc_row = feature, u_effect = effect_size,
                                    u_padj = p_adjusted,
                                    u_dir = sign(pct_sarcastic - pct_not_sarcastic)),
  descriptive_density %>% transmute(desc_row = feature, u_effect = effect_size,
                                    u_padj = p_adjusted,
                                    u_dir = sign(effect_size))
)

# 3. Model evidence: sign + FDR-adjusted p, for main (Part 4) and paired (Part 5)
model_terms <- function(model, terms) {
  co <- summary(model)$coefficients
  tibble(model_term = terms, est = co[terms, 1], p = co[terms, ncol(co)])
}
mv <- model_terms(main_model,   hypotheses$model_term) %>%
  transmute(model_term, OR = exp(est), mv_dir = sign(est), mv_padj = p.adjust(p, "BH"))
pr <- model_terms(clogit_model, hypotheses$model_term) %>%
  transmute(model_term, pr_dir = sign(est), pr_padj = p.adjust(p, "BH"))

# 4. Assemble + derive verdict (confirmatory logic: direction + FDR significance)
alpha <- 0.05
rq1_verdict <- hypotheses %>%
  left_join(univ, by = "desc_row") %>%
  left_join(mv,   by = "model_term") %>%
  left_join(pr,   by = "model_term") %>%
  mutate(
    verdict = case_when(
      u_padj <  alpha & u_dir == -expected                      ~ "Reversed",
      u_padj <  alpha & mv_padj < alpha & mv_dir == expected     ~ "Confirmed",
      u_padj >= alpha                                            ~ "Not supported",
      TRUE                                                       ~ "Mixed"
    ),
    strength = case_when(abs(u_effect) >= 0.10 ~ "moderate",
                         abs(u_effect) >= 0.05 ~ "small",
                         TRUE                  ~ "negligible"),
    paired_robust = if_else(!is.na(pr_padj) & pr_padj < alpha & pr_dir == mv_dir,
                            "yes", "no")
  ) %>%
  transmute(
    Feature      = feature,
    Expected     = if_else(expected > 0, "up sarcastic", "up non-sarcastic"),
    Observed     = if_else(u_dir > 0, "up sarcastic", "up non-sarcastic"),
    Effect_size  = round(abs(u_effect), 3),   # |Cramer's V| (sparse) / |rank-biserial| (density)
    Strength     = strength,
    Univ_padj    = signif(u_padj, 2),
    OR_per_SD    = round(OR, 3),
    Multi_padj   = signif(mv_padj, 2),
    Paired_robust = paired_robust,
    Verdict      = verdict,
    Source       = source
  ) %>%
  arrange(desc(Effect_size))

rq1_verdict

# ============================================================
# PART 7 — SUPPLEMENTARY (EXPLORATORY): DO THE GROUPS SEPARATE?
# ============================================================
# Descriptive separability check only — NOT a predictive claim.
# LDA finds the single best linear combination of the 11 features; if the two
# groups separated cleanly their projected scores would show little overlap.
# Heavy overlap here reinforces that the features are individually weak.

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


# RQ2: Can integrating semantic embeddings with psycholinguistic features
#      improve sarcasm detection over embeddings alone?
#
# Model 1: semantic embeddings only (parent + child comment, with separator)
# Model 2: semantic embeddings + 11 hand-crafted features (from RQ1)
# + RQ1's own model refit on the same split, for a 3-way comparison
#
# Input: `data` object already carrying the 11 RQ1 features
#        (i.e., run rq1_full_pipeline.R Parts 0-1 first)
# ============================================================

library(dplyr)
library(reticulate)
library(glmnet)
library(pROC)
library(tibble)

set.seed(1)

# ============================================================
# PART 1 — GENERATE EMBEDDINGS (microsoft/harrier-oss-v1-270m)
# ============================================================
# This model is decoder-only (Gemma3-based) with last-token pooling and
# instruction-tuned prompts — an architecture the `text` package's generic
# textEmbed() wrapper doesn't handle correctly (it silently failed in
# testing, returning empty results with no error). The model's own
# documentation recommends loading it directly via sentence-transformers,
# so we do that here through reticulate instead.
#
# ONE-TIME SETUP (run once, in your existing textrpp_condaenv environment):
#   reticulate::py_install("sentence-transformers", envname = "textrpp_condaenv")

reticulate::use_condaenv("textrpp_condaenv", required = TRUE)

sentence_transformers <- import("sentence_transformers")
embed_model <- sentence_transformers$SentenceTransformer("microsoft/harrier-oss-v1-270m")

# Cap sequence length: a few very long comments blow up MPS memory (attention
# cost grows with length^2). 256 tokens comfortably covers Reddit comments +
# their parent context; longer inputs are truncated.
embed_model$max_seq_length <- 256L

# Smaller batch keeps peak GPU memory well under the MPS limit (32 caused an
# out-of-memory error). Lower further to 4 if OOM persists.
get_embeddings <- function(texts, batch_size = 8L) {
  embed_model$encode(texts, batch_size = as.integer(batch_size), show_progress_bar = TRUE)
}

# ---- Combine parent + child comment with a natural-language separator ----
# This model is decoder-only (not BERT-style), so it has no trained [SEP]
# convention for paired input — a clear textual label works better than an
# arbitrary special token for signaling structure to this kind of model.
combined_text <- paste0("Context: ", data$parent_comment, "\nReply: ", data$comment)

# ---- Embedding generation is expensive — cache to disk, don't recompute ----
embedding_cache_path <- "comment_context_embeddings.rds"   # renamed — different input than before

if (file.exists(embedding_cache_path)) {
  embeddings <- readRDS(embedding_cache_path)
} else {
  # sanity check on a tiny sample first — confirms the model loads and
  # encodes correctly before committing to the full dataset
  test_embed <- get_embeddings(c(
    "Context: nice weather today\nReply: oh sure, love the rain",
    "Context: I got promoted\nReply: congratulations, well deserved"
  ))
  stopifnot(nrow(test_embed) == 2, ncol(test_embed) > 0)
  
  embeddings <- get_embeddings(combined_text)
  saveRDS(embeddings, embedding_cache_path)
}

embeddings <- as.data.frame(embeddings)
colnames(embeddings) <- paste0("emb_", seq_len(ncol(embeddings)))

# Sanity check: embeddings must align row-for-row with `data`
stopifnot(nrow(embeddings) == nrow(data))

# ============================================================
# PART 2 — LEAKAGE-SAFE TRAIN/TEST SPLIT
# ============================================================
# Split by parent_comment (not randomly at the row level) — otherwise the
# two replies in a matched pair could land on opposite sides of the split,
# letting the model implicitly "see" thread context it shouldn't have.

unique_parents <- unique(data$parent_comment)
train_parents  <- sample(unique_parents, size = floor(0.8 * length(unique_parents)))

data <- data %>%
  mutate(split = if_else(parent_comment %in% train_parents, "train", "test"))

table(data$split)
prop.table(table(data$split, data$label), margin = 1)   # confirm balance held across split

train_idx <- which(data$split == "train")
test_idx  <- which(data$split == "test")

# ============================================================
# PART 3 — PREPARE FEATURE MATRICES
# ============================================================

feat_names <- c("caps_count","excl_count","ellipsis_count","interjection_count",
                "emoticon_count","laughter_count","quote_count",
                "intensifier_count","vader_comment","sentiment_diff","log_length")
# has_qe excluded — caused a separation artifact in RQ1's regression
# (OR 1347, non-significant p-value, AUC 0.5, zero RF importance)

# Scale hand-crafted features using TRAIN statistics only — applying test-set
# means/SDs would leak test-set information into the "standardization"
train_means <- sapply(data[train_idx, feat_names], mean, na.rm = TRUE)
train_sds   <- sapply(data[train_idx, feat_names], sd, na.rm = TRUE)

scale_with_train_stats <- function(df) {
  scaled <- sweep(as.matrix(df[, feat_names]), 2, train_means, "-")
  scaled <- sweep(scaled, 2, train_sds, "/")
  scaled
}

features_scaled <- scale_with_train_stats(data)

# Mean-impute any NA (mainly sentiment_diff, where the parent comment could not
# be VADER-scored). After train-scaling, 0 == the training mean, so this is a
# neutral imputation and keeps every row (glmnet rejects NAs outright).
cat("NA cells imputed in hand-crafted features:", sum(is.na(features_scaled)), "\n")
features_scaled[is.na(features_scaled)] <- 0

# ---- Model 1 input: embeddings only ----
X_emb <- as.matrix(embeddings)

# ---- Model 2 input: embeddings + hand-crafted features ----
X_combined <- cbind(X_emb, features_scaled)

y <- data$label

X_emb_train <- X_emb[train_idx, ]
X_emb_test  <- X_emb[test_idx, ]

X_combined_train <- X_combined[train_idx, ]
X_combined_test  <- X_combined[test_idx, ]

y_train <- y[train_idx]
y_test  <- y[test_idx]

# ============================================================
# PART 4 — FIT MODEL 1 (embeddings only) — ridge logistic regression
# ============================================================

model1 <- cv.glmnet(X_emb_train, y_train, family = "binomial", alpha = 0)

model1_pred_prob <- predict(model1, newx = X_emb_test, s = "lambda.min", type = "response")[, 1]

# ============================================================
# PART 5 — FIT MODEL 2 (embeddings + features) — ridge logistic regression
# ============================================================

model2 <- cv.glmnet(X_combined_train, y_train, family = "binomial", alpha = 0)

model2_pred_prob <- predict(model2, newx = X_combined_test, s = "lambda.min", type = "response")[, 1]

# ============================================================
# PART 6 — REFIT THE RQ1 MODEL ON THE SAME SPLIT (for fair comparison)
# ============================================================
# Same formula as RQ1 (12 features + subreddit fixed effects), just fit on
# the training portion only and evaluated on the held-out test set — RQ1's
# original fit used the full dataset for inference, which isn't a fair
# comparison point until it's evaluated out-of-sample like the others.

# Pool subreddits (top-30 individually, the rest as "Other") — the same control
# as RQ1 Part 4. Every subreddit maps to a known level, so NO test rows are
# dropped: the RQ1 model is evaluated on the FULL test set, exactly like Model 1
# and Model 2 → all three models are compared on the same comments.
top_subs <- data %>% count(subreddit, sort = TRUE) %>% slice_head(n = 30) %>% pull(subreddit)
pool_sub <- function(df) df %>%
  mutate(subreddit = if_else(subreddit %in% top_subs, subreddit, "Other"))

rq1_train <- pool_sub(data[train_idx, ])
rq1_test  <- pool_sub(data[test_idx, ])

rq1_model_refit <- glm(
  label ~ caps_count + excl_count + ellipsis_count + interjection_count +
    emoticon_count + laughter_count + quote_count +
    intensifier_count + vader_comment + sentiment_diff +
    log_length + factor(subreddit),
  data = rq1_train,
  family = binomial
)

rq1_pred_prob <- predict(rq1_model_refit, newdata = rq1_test, type = "response")
rq1_actual    <- rq1_test$label

# ============================================================
# PART 7 — EVALUATION: accuracy, precision, recall, F1, AUC
# ============================================================

evaluate_model <- function(pred_prob, actual, threshold = 0.5, model_name) {
  pred_class <- as.integer(pred_prob > threshold)
  cm <- table(factor(pred_class, levels = c(0,1)), factor(actual, levels = c(0,1)))
  
  accuracy  <- sum(diag(cm)) / sum(cm)
  precision <- cm["1","1"] / sum(cm["1", ])
  recall    <- cm["1","1"] / sum(cm[, "1"])
  f1        <- 2 * precision * recall / (precision + recall)
  auc_val   <- as.numeric(auc(actual, pred_prob, quiet = TRUE))
  
  tibble(
    model = model_name,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    auc = auc_val
  )
}

results <- bind_rows(
  evaluate_model(model1_pred_prob, y_test, model_name = "Model 1: embeddings only"),
  evaluate_model(model2_pred_prob, y_test, model_name = "Model 2: embeddings + features"),
  evaluate_model(rq1_pred_prob, rq1_actual, model_name = "RQ1 model: features + subreddit FE")
)

results

# ============================================================
# PART 7b — REPEATED-SPLIT ROBUSTNESS (Model 1 vs Model 2)
# ============================================================
# Why: the Model 2 gain over Model 1 is small, so we repeat the leakage-safe
# split -> fit -> evaluate over 10 seeds to confirm it holds across partitions
# rather than being one-split noise (Session 4: single splits are noisy; report
# performance across folds as mean +/- SD).

run_one_split <- function(seed) {
  set.seed(seed)
  tr_parents <- sample(unique_parents, floor(0.8 * length(unique_parents)))
  tr <- which(data$parent_comment %in% tr_parents)
  te <- setdiff(seq_len(nrow(data)), tr)

  # scale hand-crafted features with THIS split's train stats (no leakage)
  tm <- sapply(data[tr, feat_names], mean, na.rm = TRUE)
  ts <- sapply(data[tr, feat_names], sd,   na.rm = TRUE)
  fs <- sweep(sweep(as.matrix(data[, feat_names]), 2, tm, "-"), 2, ts, "/")
  fs[is.na(fs)] <- 0
  Xc <- cbind(X_emb, fs)

  m1 <- cv.glmnet(X_emb[tr, ], y[tr], family = "binomial", alpha = 0)
  m2 <- cv.glmnet(Xc[tr, ],    y[tr], family = "binomial", alpha = 0)
  p1 <- predict(m1, X_emb[te, ], s = "lambda.min", type = "response")[, 1]
  p2 <- predict(m2, Xc[te, ],    s = "lambda.min", type = "response")[, 1]

  bind_rows(
    evaluate_model(p1, y[te], model_name = "Model 1: embeddings only"),
    evaluate_model(p2, y[te], model_name = "Model 2: embeddings + features")
  ) %>% mutate(seed = seed)
}

seeds <- 1:10
multi <- bind_rows(lapply(seeds, run_one_split))

# mean +/- SD per model across the 10 splits
multi_summary <- multi %>% group_by(model) %>%
  summarise(f1_mean = mean(f1), f1_sd = sd(f1),
            auc_mean = mean(auc), auc_sd = sd(auc), .groups = "drop")
multi_summary

# paired improvement (Model 2 - Model 1) and how often Model 2 wins
m1v <- multi %>% filter(model == "Model 1: embeddings only")       %>% arrange(seed)
m2v <- multi %>% filter(model == "Model 2: embeddings + features") %>% arrange(seed)
robustness_summary <- tibble(
  mean_delta_f1  = mean(m2v$f1  - m1v$f1),
  sd_delta_f1    = sd(m2v$f1  - m1v$f1),
  mean_delta_auc = mean(m2v$auc - m1v$auc),
  win_rate_f1    = mean((m2v$f1 - m1v$f1) > 0)
)
robustness_summary

# ============================================================
# PART 8 (OPTIONAL) — GRADIENT BOOSTING ROBUSTNESS CHECK
# ============================================================
# Checks whether the embeddings-vs-embeddings+features conclusion holds
# under a non-linear classifier too, not just regularized logistic regression.

library(xgboost)

# Leakage-safe early stopping: carve a validation fold out of TRAIN and watch
# THAT for early stopping. The test set is touched only for final prediction —
# watching it (as before) would tune the stopping round on the test data and
# inflate the reported metrics.
set.seed(1)
n_tr     <- length(y_train)
val_rows <- sample(n_tr, size = floor(0.2 * n_tr))
sub_rows <- setdiff(seq_len(n_tr), val_rows)

# Model 1 (embeddings only)
dsub_m1  <- xgb.DMatrix(data = X_emb_train[sub_rows, ], label = y_train[sub_rows])
dval_m1  <- xgb.DMatrix(data = X_emb_train[val_rows, ], label = y_train[val_rows])
dtest_m1 <- xgb.DMatrix(data = X_emb_test, label = y_test)

# Model 2 (embeddings + features)
dsub_m2  <- xgb.DMatrix(data = X_combined_train[sub_rows, ], label = y_train[sub_rows])
dval_m2  <- xgb.DMatrix(data = X_combined_train[val_rows, ], label = y_train[val_rows])
dtest_m2 <- xgb.DMatrix(data = X_combined_test, label = y_test)

xgb_params <- list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 6, eta = 0.1)

xgb_model1 <- xgb.train(xgb_params, dsub_m1, nrounds = 200,
                        watchlist = list(val = dval_m1), early_stopping_rounds = 15, verbose = 0)
xgb_model2 <- xgb.train(xgb_params, dsub_m2, nrounds = 200,
                        watchlist = list(val = dval_m2), early_stopping_rounds = 15, verbose = 0)

xgb_pred1 <- predict(xgb_model1, dtest_m1)
xgb_pred2 <- predict(xgb_model2, dtest_m2)

results_xgb <- bind_rows(
  evaluate_model(xgb_pred1, y_test, model_name = "XGBoost: embeddings only"),
  evaluate_model(xgb_pred2, y_test, model_name = "XGBoost: embeddings + features")
)

results_xgb

# ============================================================
# PART 9 — FINAL COMBINED COMPARISON TABLE
# ============================================================

final_comparison <- bind_rows(results, results_xgb) %>%
  arrange(desc(f1))

final_comparison

# ============================================================
# PART 10 — ARE THE PERFORMANCE DIFFERENCES SIGNIFICANT?
# ============================================================
# All three core models were evaluated on the SAME test comments, so this
# is a paired comparison — McNemar's test checks whether one model gets
# significantly more of those specific comments right than another.

compare_models_mcnemar <- function(pred_prob_a, actual_a, pred_prob_b, actual_b,
                                   threshold = 0.5, name_a, name_b) {
  stopifnot(length(actual_a) == length(actual_b))
  
  correct_a <- as.integer(pred_prob_a > threshold) == actual_a
  correct_b <- as.integer(pred_prob_b > threshold) == actual_b
  
  tab <- table(correct_a, correct_b)
  test_result <- mcnemar.test(tab)
  
  tibble(
    comparison = paste(name_a, "vs.", name_b),
    p_value = test_result$p.value
  )
}

# With subreddit pooling ("Other"), the RQ1 model now predicts on the FULL
# test set — the same rows, in the same order, as Model 1 and Model 2. So every
# test comment is comparable across all three models; no subsetting is needed.
rq1_comparable_idx <- seq_along(y_test)

mcnemar_results <- bind_rows(
  compare_models_mcnemar(model1_pred_prob, y_test, model2_pred_prob, y_test,
                         name_a = "Model 1 (embeddings)", name_b = "Model 2 (embeddings+features)"),
  compare_models_mcnemar(model1_pred_prob[rq1_comparable_idx], y_test[rq1_comparable_idx],
                         rq1_pred_prob, rq1_actual,
                         name_a = "Model 1 (embeddings)", name_b = "RQ1 model"),
  compare_models_mcnemar(model2_pred_prob[rq1_comparable_idx], y_test[rq1_comparable_idx],
                         rq1_pred_prob, rq1_actual,
                         name_a = "Model 2 (embeddings+features)", name_b = "RQ1 model")
)

mcnemar_results

# ============================================================
# PART 11 — VISUALIZE THE COMPARISON
# ============================================================
library(ggplot2)
library(tidyr)

final_comparison %>%
  pivot_longer(cols = c(accuracy, precision, recall, f1, auc),
               names_to = "metric", values_to = "value") %>%
  ggplot(aes(x = model, y = value, fill = model)) +
  geom_col() +
  facet_wrap(~ metric, scales = "free_y") +
  coord_flip() +
  labs(title = "Model performance comparison (RQ2)", x = NULL, y = NULL) +
  theme_minimal() +
  theme(legend.position = "none")

# ============================================================
# PART 12 — ROC CURVES (visual AUC comparison)
# ============================================================

# ---- (a) Model 1 vs Model 2 vs both XGBoost variants — full test set,
#          since all four share the exact same y_test ----
roc_m1     <- roc(y_test, model1_pred_prob, quiet = TRUE)
roc_m2     <- roc(y_test, model2_pred_prob, quiet = TRUE)
roc_xgb1   <- roc(y_test, xgb_pred1, quiet = TRUE)
roc_xgb2   <- roc(y_test, xgb_pred2, quiet = TRUE)

ggroc(list(
  "Model 1: embeddings only"          = roc_m1,
  "Model 2: embeddings + features"     = roc_m2,
  "XGBoost: embeddings only"           = roc_xgb1,
  "XGBoost: embeddings + features"     = roc_xgb2
), size = 0.9) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey60") +
  labs(title = "ROC curves — embeddings-only vs. embeddings+features",
       subtitle = "Linear (ridge) and non-linear (XGBoost) classifiers",
       color = NULL, x = "Specificity", y = "Sensitivity") +
  theme_minimal()

# ---- (b) Three-way including the RQ1 model, on the comparable subset ----
roc_m1_sub  <- roc(y_test[rq1_comparable_idx], model1_pred_prob[rq1_comparable_idx], quiet = TRUE)
roc_m2_sub  <- roc(y_test[rq1_comparable_idx], model2_pred_prob[rq1_comparable_idx], quiet = TRUE)
roc_rq1     <- roc(rq1_actual, rq1_pred_prob, quiet = TRUE)

ggroc(list(
  "Model 1: embeddings only"           = roc_m1_sub,
  "Model 2: embeddings + features"      = roc_m2_sub,
  "RQ1 model: features + subreddit FE"  = roc_rq1
), size = 0.9) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "grey60") +
  labs(title = "ROC curves — 3-way comparison including the RQ1 model",
       subtitle = "Evaluated on the RQ1-comparable test subset",
       color = NULL, x = "Specificity", y = "Sensitivity") +
  theme_minimal()

# ============================================================
# PART 13 — CONFUSION MATRIX HEATMAPS
# ============================================================

make_cm_df <- function(pred_prob, actual, model_name, threshold = 0.5) {
  pred_class <- as.integer(pred_prob > threshold)
  tab <- table(Predicted = pred_class, Actual = actual)
  as.data.frame(tab) %>%
    mutate(model = model_name,
           pct = Freq / sum(Freq))
}

cm_all <- bind_rows(
  make_cm_df(model1_pred_prob, y_test, "Model 1: embeddings only"),
  make_cm_df(model2_pred_prob, y_test, "Model 2: embeddings + features"),
  make_cm_df(rq1_pred_prob, rq1_actual, "RQ1 model: features + subreddit FE")
)

ggplot(cm_all, aes(x = Actual, y = Predicted, fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = paste0(Freq, "\n(", scales::percent(pct, accuracy = 0.1), ")")),
            size = 3.2) +
  scale_fill_gradient(low = "white", high = "steelblue", labels = scales::percent) +
  facet_wrap(~ model) +
  labs(title = "Confusion matrices by model", fill = "% of test set") +
  theme_minimal()

# ============================================================
# PART 14 — FINAL ANNOTATED SUMMARY TABLE
# ============================================================
# Combines performance metrics with pairwise significance flags in one place

metrics_table <- final_comparison %>%
  mutate(across(where(is.numeric), ~ round(., 3)))

metrics_table

significance_table <- mcnemar_results %>%
  mutate(
    p_value = round(p_value, 4),
    significant_at_05 = if_else(p_value < 0.05, "Yes", "No")
  )

significance_table

# ---- Combined narrative table: metrics ranked, with a note on which
#      differences are statistically confirmed ----
cat("\n=== RQ2 Summary ===\n")
cat("\nPerformance metrics (ranked by F1):\n")
print(metrics_table)
cat("\nPairwise significance (McNemar's test, paired on shared test comments):\n")
print(significance_table)

cat("\nCross-split robustness (Model 1 vs Model 2, 10 leakage-safe splits):\n")
print(multi_summary)          # F1/AUC mean +/- SD per model across splits
print(robustness_summary)     # mean delta + win-rate: how consistently Model 2 wins