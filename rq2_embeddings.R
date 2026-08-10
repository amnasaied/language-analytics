## ==================================================================
## RQ: Can the integration of semantic embeddings and psycholinguistic
##     features improve sarcasm detection in user-generated content?
##
## Design:
##   Model 1  -> semantic embeddings ONLY                (harrier-oss-v1-270m)
##   Model 2  -> semantic embeddings + psycholinguistic features
##               (punctuation, capitalization, intra-comment
##               incongruity, parent-reply incongruity)
##   Both models use the identical classifier / CV setup / train-test
##   split, so any performance gap is attributable to the added
##   psycholinguistic features rather than to modeling choices.
##
## Structure of this script:
##   1. Setup (packages)
##   2. Load data
##   3. Sampling & cleaning (compute-feasible subset)
##   4. Psycholinguistic feature engineering
##   5. Semantic embeddings (harrier-oss-v1-270m)
##   6. Assemble Model 1 / Model 2 datasets + train/test split
##   7. Train classifiers (elastic-net logistic regression, 10-fold CV)
##   8. Evaluate (Accuracy, Precision, Recall, F1, AUC)
##   9. Compare Model 1 vs Model 2
##  10. Figures
## ==================================================================


# ==================================================================
# 1. SETUP
# ==================================================================
library(tidyverse)
library(caret)        # train(), confusionMatrix(), createDataPartition()
library(glmnet)        # elastic-net logistic regression engine used by caret

if (!requireNamespace("text", quietly = TRUE)) install.packages("text")
library(text)          # textEmbed(), for the semantic-embedding features

if (!requireNamespace("sentimentr", quietly = TRUE)) install.packages("sentimentr")
library(sentimentr)    # sentence-level sentiment, used for incongruity features

if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")
library(pROC)          # ROC curves / AUC

if (!requireNamespace("doParallel", quietly = TRUE)) install.packages("doParallel")
library(doParallel)    # parallel CV folds

# One-time setup for the {text} package's Python backend (torch/transformers).
# Uncomment on first use in a fresh R session / environment:
# reticulate::install_miniconda()
# textrpp_install()

reticulate::conda_list()

text::textrpp_initialize(save_profile = TRUE)

#test_embed <- textEmbed(
#  texts = c("This is a test.", "Another test sentence."),
#  model = "microsoft/harrier-oss-v1-270m"
#)
#str(test_embed, max.level = 2)

#reticulate::conda_install(
#  envname  = "textrpp_condaenv",
#  packages = "transformers>=4.50",
#  pip      = TRUE
#)


# ==================================================================
# 2. LOAD DATA
# --------------------------------------------------------------
# The full corpus has 1M+ comments -- computing transformer embeddings
# for every row is not feasible on a laptop. Instead of a random
# sample, we cut the dataset down to a coherent, substantively
# motivated subset: comments from marketing-relevant subreddits
# (tech, gaming, cars, fashion, retail, entertainment). This keeps
# the size manageable while keeping every remaining comment relevant
# to a marketing/UGC context, rather than an arbitrary draw.
# ==================================================================
url <- "https://huggingface.co/datasets/marcbishara/sarcasm-on-reddit/resolve/main/train-balanced-sarcasm.csv"
sarcasm_raw <- read_csv(url)

# Filter the dataset to keep only marketing-related subreddits
sarcasm_raw <- sarcasm_raw %>%
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

unique(sarcasm_raw$subreddit) # Check unique subreddits
table(sarcasm_raw$label)      # Check frequencies of labels => already balanced

cat("Rows:", nrow(sarcasm_raw), "\n")
cat("Columns:", ncol(sarcasm_raw), "\n")
glimpse(sarcasm_raw)


# ==================================================================
# 3. CLEANING
# --------------------------------------------------------------
# We require a non-missing, non-empty parent_comment, since the
# parent-reply incongruity feature (Model 2) needs it -- this keeps
# Model 1 and Model 2 on the EXACT SAME rows, so the comparison is
# fair (Model 1 does not get an advantage from a larger row set).
# ==================================================================
sarcasm_clean <- sarcasm_raw %>%
  filter(!is.na(label), !is.na(comment), !is.na(parent_comment)) %>%
  filter(nchar(trimws(comment)) > 0, nchar(trimws(parent_comment)) > 0) %>%
  mutate(label_f = factor(label, levels = c(0, 1),
                          labels = c("NonSarcastic", "Sarcastic"))) %>%
  mutate(row_id = row_number())

cat("\nCleaned dataset for modeling:\n")
print(table(sarcasm_clean$label_f))


# ==================================================================
# 4. PSYCHOLINGUISTIC FEATURE ENGINEERING
# --------------------------------------------------------------
# Same feature family as RQ1: punctuation, capitalization, and two
# incongruity signals (within-comment, and comment-vs-parent).
# ==================================================================

# --- 4a. Punctuation (regex counts) ---------------------------------
# Sarcasm markers frequently cited in the literature: exclamation
# marks, question marks, and ellipses (often used to signal irony).
sarcasm_clean <- sarcasm_clean %>%
  mutate(
    n_exclaim   = str_count(comment, "!"),
    n_question  = str_count(comment, "\\?"),
    n_ellipsis  = str_count(comment, "\\.\\.\\.|???"),
    punct_total = n_exclaim + n_question + n_ellipsis,
    comment_length = nchar(comment),
    punct_density  = punct_total / pmax(comment_length, 1)  # length-normalized
  )

# --- 4b. Capitalization (uppercase word count) -----------------------
# Words written in ALL CAPS (2+ letters, to exclude the standalone "I")
# are a common typographic marker of sarcasm/emphasis.
count_upper_words <- function(x) {
  words <- str_extract_all(x, "\\b[A-Z]{2,}\\b")
  lengths(words)
}
sarcasm_clean <- sarcasm_clean %>%
  mutate(
    n_upper_words   = count_upper_words(comment),
    upper_word_rate = n_upper_words / pmax(str_count(comment, "\\S+"), 1)
  )

# --- 4c. Intra-comment incongruity ------------------------------------
# Sarcasm often mixes sentences/clauses of opposite sentiment within
# the SAME comment (e.g., positive wording, negative intent). We proxy
# this with the standard deviation of sentence-level sentiment within
# a comment -- higher SD = more internal sentiment inconsistency.
# Single-sentence comments get 0 (no internal variation to measure).
comment_sentiment_sentences <- sentiment(get_sentences(sarcasm_clean$comment))

intra_incongruity <- comment_sentiment_sentences %>%
  group_by(element_id) %>%
  summarise(
    intra_comment_incong = ifelse(n() > 1, sd(sentiment, na.rm = TRUE), 0),
    .groups = "drop"
  )

sarcasm_clean <- sarcasm_clean %>%
  mutate(element_id = row_number()) %>%
  left_join(intra_incongruity, by = "element_id") %>%
  mutate(intra_comment_incong = replace_na(intra_comment_incong, 0)) %>%
  select(-element_id)

# --- 4d. Parent-reply incongruity (sentiment-based) -------------------
# Document-level sentiment for the comment and for the comment it is
# replying to; the absolute gap captures cases where a reply flips
# the emotional tone of the conversation -- a classic sarcasm cue
# ("Great, another Monday." after a neutral/positive parent).
sent_comment <- sentiment_by(sarcasm_clean$comment)$ave_sentiment
sent_parent  <- sentiment_by(sarcasm_clean$parent_comment)$ave_sentiment

sarcasm_clean <- sarcasm_clean %>%
  mutate(
    sentiment_comment = sent_comment,
    sentiment_parent  = sent_parent,
    parent_reply_incong = abs(sentiment_comment - sentiment_parent),
    parent_reply_flip   = as.integer(sign(sentiment_comment) != sign(sentiment_parent) &
                                       sentiment_comment != 0 & sentiment_parent != 0)
  )

cat("\n===== Psycholinguistic Feature Summary =====\n")
print(
  sarcasm_clean %>%
    group_by(label_f) %>%
    summarise(
      mean_punct_density   = mean(punct_density),
      mean_upper_word_rate = mean(upper_word_rate),
      mean_intra_incong    = mean(intra_comment_incong),
      mean_parent_incong   = mean(parent_reply_incong),
      .groups = "drop"
    )
)


# ==================================================================
# 5. SEMANTIC EMBEDDINGS (harrier-oss-v1-270m)
# --------------------------------------------------------------
# Document-level (mean-pooled) embeddings for both the comment and
# its parent comment. The comment embeddings are the sole predictors
# for Model 1, and are combined with the psycholinguistic features
# for Model 2. The parent embeddings are only used to derive a
# semantic (as opposed to sentiment-only) incongruity feature.
# ==================================================================
embeddings_comment <- textEmbed(
  texts = sarcasm_clean$comment,
  model = "microsoft/harrier-oss-v1-270m",
  aggregation_from_tokens_to_texts = "mean",
  keep_token_embeddings = FALSE,
  layers = -2
)
saveRDS(embeddings_comment, "embeddings_comment.rds")
# embeddings_comment <- readRDS("embeddings_comment.rds")  # reload if needed

embeddings_parent <- textEmbed(
  texts = sarcasm_clean$parent_comment,
  model = "microsoft/harrier-oss-v1-270m",
  aggregation_from_tokens_to_texts = "mean",
  keep_token_embeddings = FALSE,
  layers = -2
)
saveRDS(embeddings_parent, "embeddings_parent.rds")
# embeddings_parent <- readRDS("embeddings_parent.rds")  # reload if needed

emb_comment <- embeddings_comment$texts$texts   # tibble: Dim1 ... DimN
emb_parent  <- embeddings_parent$texts$texts

cat("\nEmbedding dimensions:", ncol(emb_comment), "\n")

# --- Semantic incongruity: 1 - cosine similarity(comment, parent) ----
# Complements the sentiment-based incongruity above: this captures
# TOPICAL/semantic mismatch between a reply and what it replies to,
# not just a sentiment-sign flip.
cosine_sim <- function(a, b) sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))

semantic_incongruity <- map_dbl(seq_len(nrow(emb_comment)), function(i) {
  1 - cosine_sim(as.numeric(emb_comment[i, ]), as.numeric(emb_parent[i, ]))
})

sarcasm_clean$semantic_incong <- semantic_incongruity
str(embeddings_comment$tokens, max.level = 2)
str(embeddings_comment$word_types, max.level = 2)
packageVersion("text")
?textEmbed

test_embed <- textEmbed(
  texts = c("This is a test.", "Another test sentence."),
  model = "microsoft/harrier-oss-v1-270m"
)
str(test_embed, max.level = 2)


# ==================================================================
# 6. ASSEMBLE MODEL DATASETS + TRAIN/TEST SPLIT
# --------------------------------------------------------------
# Single stratified 80/20 split, reused identically for both models
# so Model 1 and Model 2 are evaluated on the exact same test rows.
# ==================================================================
set.seed(123)
train_idx <- createDataPartition(sarcasm_clean$label_f, p = 0.8, list = FALSE)

split_type <- rep("test", nrow(sarcasm_clean))
split_type[train_idx] <- "train"
sarcasm_clean$Type <- split_type

cat("\nTrain/test split:\n")
print(table(sarcasm_clean$Type, sarcasm_clean$label_f))

# Rename embedding columns to avoid collisions and make both models'
# predictor sets easy to build with a simple bind_cols().
colnames(emb_comment) <- paste0("Emb_", colnames(emb_comment))

psycholinguistic_features <- sarcasm_clean %>%
  select(punct_density, n_exclaim, n_question, n_ellipsis,
         upper_word_rate, n_upper_words,
         intra_comment_incong, parent_reply_incong, parent_reply_flip,
         semantic_incong)

# --- Model 1 dataset: embeddings ONLY ---------------------------------
model1_data <- bind_cols(emb_comment, label = sarcasm_clean$label_f)
model1_data$Type <- sarcasm_clean$Type

# --- Model 2 dataset: embeddings + psycholinguistic features ----------
model2_data <- bind_cols(emb_comment, psycholinguistic_features,
                         label = sarcasm_clean$label_f)
model2_data$Type <- sarcasm_clean$Type

model1_train <- filter(model1_data, Type == "train") %>% select(-Type)
model1_test  <- filter(model1_data, Type == "test")  %>% select(-Type)
model2_train <- filter(model2_data, Type == "train") %>% select(-Type)
model2_test  <- filter(model2_data, Type == "test")  %>% select(-Type)


# ==================================================================
# 7. TRAIN CLASSIFIERS
# --------------------------------------------------------------
# Elastic-net logistic regression (glmnet via caret), 10-fold CV.
# Same algorithm and same CV scheme for both models -- again, so any
# performance difference reflects the added FEATURES, not the model.
# ==================================================================
cl <- makePSOCKcluster(max(parallel::detectCores() - 1, 1))
registerDoParallel(cl)

ctrl <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE, summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(123)
cat("\nTraining Model 1 (embeddings only)...\n")
model1_fit <- train(
  label ~ ., data = model1_train,
  method = "glmnet", family = "binomial",
  trControl = ctrl, metric = "ROC",
  preProcess = c("center", "scale"),
  tuneLength = 5
)

set.seed(123)
cat("Training Model 2 (embeddings + psycholinguistic features)...\n")
model2_fit <- train(
  label ~ ., data = model2_train,
  method = "glmnet", family = "binomial",
  trControl = ctrl, metric = "ROC",
  preProcess = c("center", "scale"),
  tuneLength = 5
)

stopCluster(cl)
registerDoSEQ()

model1_fit
model2_fit


# ==================================================================
# 8. EVALUATE (Accuracy, Precision, Recall, F1, AUC)
# ==================================================================

# --- Model 1 ------------------------------------------------------
m1_pred_class <- predict(model1_fit, newdata = model1_test)
m1_pred_prob  <- predict(model1_fit, newdata = model1_test, type = "prob")$Sarcastic

cfm_model1 <- confusionMatrix(m1_pred_class, model1_test$label,
                              positive = "Sarcastic", mode = "everything")
print(cfm_model1)

roc_model1 <- roc(response = model1_test$label, predictor = m1_pred_prob,
                  levels = c("NonSarcastic", "Sarcastic"))
auc_model1 <- as.numeric(auc(roc_model1))

# --- Model 2 --------------------------------------------------------
m2_pred_class <- predict(model2_fit, newdata = model2_test)
m2_pred_prob  <- predict(model2_fit, newdata = model2_test, type = "prob")$Sarcastic

cfm_model2 <- confusionMatrix(m2_pred_class, model2_test$label,
                              positive = "Sarcastic", mode = "everything")
print(cfm_model2)

roc_model2 <- roc(response = model2_test$label, predictor = m2_pred_prob,
                  levels = c("NonSarcastic", "Sarcastic"))
auc_model2 <- as.numeric(auc(roc_model2))

# --- DeLong test: is the AUC gap statistically significant? ---------
delong_test <- roc.test(roc_model1, roc_model2, method = "delong")
print(delong_test)


# ==================================================================
# 9. COMPARE MODEL 1 vs MODEL 2
# ==================================================================
extract_metrics <- function(cfm, auc_val, model_name) {
  tibble(
    Model     = model_name,
    Accuracy  = cfm$overall["Accuracy"],
    Precision = cfm$byClass["Precision"],
    Recall    = cfm$byClass["Recall"],
    F1        = cfm$byClass["F1"],
    AUC       = auc_val
  )
}

comparison_table <- bind_rows(
  extract_metrics(cfm_model1, auc_model1, "Model 1: Embeddings only"),
  extract_metrics(cfm_model2, auc_model2, "Model 2: Embeddings + Psycholinguistic")
)

cat("\n===== MODEL COMPARISON =====\n")
print(comparison_table)

cat("\nDeLong test p-value (AUC difference):",
    round(delong_test$p.value, 4), "\n")
cat("(p < .05 => the psycholinguistic features produce a\n",
    "statistically significant change in discrimination ability)\n")


# ==================================================================
# 10. FIGURES
# ==================================================================
theme_report <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
pal <- c("Model 1: Embeddings only" = "#0072B2",
         "Model 2: Embeddings + Psycholinguistic" = "#D55E00")

# --- FIGURE 1: metric comparison bar chart ---------------------------
comparison_long <- comparison_table %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(Metric, Value, fill = Model)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.7), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = pal) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Sarcasm Detection: Embeddings-Only vs. Embeddings + Psycholinguistic Features",
       x = NULL, y = "Score") +
  theme_report +
  theme(legend.position = "bottom", axis.text.x = element_text(angle = 15, hjust = 1))

# --- FIGURE 2: ROC curve overlay --------------------------------------
roc_df <- bind_rows(
  tibble(FPR = 1 - roc_model1$specificities, TPR = roc_model1$sensitivities,
         Model = "Model 1: Embeddings only"),
  tibble(FPR = 1 - roc_model2$specificities, TPR = roc_model2$sensitivities,
         Model = "Model 2: Embeddings + Psycholinguistic")
)

ggplot(roc_df, aes(FPR, TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  scale_color_manual(values = pal) +
  labs(title = "ROC Curves: Sarcasm Classifier Comparison",
       subtitle = sprintf("AUC Model 1 = %.3f | AUC Model 2 = %.3f | DeLong p = %.4f",
                          auc_model1, auc_model2, delong_test$p.value),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_report + theme(legend.position = "bottom")

# --- FIGURE 3: confusion matrices, side by side -----------------------
cfm_to_df <- function(cfm, model_name) {
  as.data.frame(cfm$table) %>%
    rename(Predicted = Prediction, Actual = Reference) %>%
    mutate(Model = model_name)
}
cfm_df <- bind_rows(
  cfm_to_df(cfm_model1, "Model 1: Embeddings only"),
  cfm_to_df(cfm_model2, "Model 2: Embeddings + Psycholinguistic")
)

ggplot(cfm_df, aes(Predicted, Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "#f0f0f0", high = "#0072B2") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices") +
  theme_report + theme(legend.position = "none")

# --- FIGURE 4: feature importance for Model 2 (top 20) -----------------
# Shows whether the psycholinguistic features rank among the most
# influential predictors, or whether embeddings still dominate.
imp2 <- varImp(model2_fit)$importance %>%
  rownames_to_column("Feature") %>%
  mutate(IsPsycholinguistic = !str_starts(Feature, "Emb_")) %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 20)

ggplot(imp2, aes(reorder(Feature, Overall), Overall, fill = IsPsycholinguistic)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("FALSE" = "#0072B2", "TRUE" = "#D55E00"),
                    labels = c("Embedding dim.", "Psycholinguistic"),
                    name = NULL) +
  labs(title = "Model 2: Top 20 Most Important Features",
       x = NULL, y = "Importance") +
  theme_report