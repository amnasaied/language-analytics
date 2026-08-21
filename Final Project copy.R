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

# ============================================================
# Data cleaning — SARC Reddit sarcasm dataset (marketing-related subreddits)
# For RQ1: feature-level comparison of sarcastic vs. non-sarcastic comments
library(dplyr)
library(stringr)
library(tidyr)

# 1. Check for missingness
colSums(is.na(data))
# remove null values
data <- data %>%
  filter(!is.na(comment), str_trim(comment) != "",
         !is.na(parent_comment), str_trim(parent_comment) != "")

# ------------------------------------------------------------
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

# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 4. Remove duplicate / near-duplicate comments
#    Copypasta and bot replies can inflate feature importance artificially
n_before <- nrow(data)
data <- data %>% distinct(comment, parent_comment, .keep_all = TRUE)
n_before - nrow(data)   # number of exact duplicates removed

# ------------------------------------------------------------
# 5. Derive basic length/date fields (useful controls & candidate features)
data <- data %>%
  mutate(
    comment_len   = str_length(comment),
    word_count    = str_count(comment, "\\S+"),
    year          = as.integer(str_sub(date, 1, 4))
  )

# ------------------------------------------------------------
# 6. Inspect the length distribution — decide on a minimum length threshold
#    Very short comments ("lol", single emoji) carry little feature signal
# ------------------------------------------------------------
summary(data$word_count)
quantile(data$word_count, probs = c(.01, .05, .5, .95, .99))

# Example threshold — adjust based on what you see above
data <- data %>% filter(word_count >= 3)

# ------------------------------------------------------------
# 7. Re-check label balance after all filtering
#    (filtering can quietly unbalance what started as 50/50)
# ------------------------------------------------------------
table(data$label)
prop.table(table(data$label))

# ------------------------------------------------------------
# 8. Final check — inspect a few rows to confirm cleaning worked as intended
# ------------------------------------------------------------
data %>% select(comment, parent_comment, label) %>% slice_sample(n = 10)

# ------------------------------------------------------------
# Cleaned object: `data`
# Columns available for RQ1 feature comparison: comment, parent_comment,
# label, subreddit, author, score, ups, downs, comment_len, word_count, year
# ------------------------------------------------------------