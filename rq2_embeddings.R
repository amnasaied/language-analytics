## ==================================================================
## RQ2: Can the integration of semantic embeddings and psycholinguistic
##      features improve sarcasm detection in user-generated content?
##
## DESIGN -- read this before changing anything below.
##
##   Model 1 = comment embeddings + parent embeddings + semantic_incong
##   Model 2 = Model 1 + the 12 psycholinguistic features from RQ1
##
## Two things about this design are deliberate and load-bearing:
##
##   (a) The psycholinguistic block is NOT redefined here. It is
##       sourced from features_psycholinguistic.R, the same file RQ1
##       uses, via PSYCH_FEATURES. RQ2 asks whether RQ1's features add
##       anything on top of embeddings; that is only a meaningful
##       question if the block is literally the same object. Earlier
##       versions of these two scripts had drifted -- RQ2 used
##       sentimentr where RQ1 used VADER, defined intra-comment
##       incongruity differently, and silently dropped quote_count.
##
##   (b) Model 1 gets the PARENT embeddings and semantic_incong, even
##       though the "headline" framing is "embeddings only". Without
##       this, Model 1 sees only the comment while Model 2 also sees
##       parent-derived features, so any gain would be confounded with
##       simply having access to conversational context. Giving both
##       models the same INFORMATION SOURCES means the only thing that
##       differs is the feature TYPE -- which is what the RQ asks
##       about. semantic_incong belongs in Model 1 for the same
##       reason: it is computed from embeddings, so it is an embedding
##       feature, not a psycholinguistic one.
##
##   The delta between the two models is therefore attributable to the
##   psycholinguistic features and nothing else.
##
## WHAT TO EXPECT (state this in the paper BEFORE running):
##   RQ1 shows these features reach only AUC ~0.59 alone, while an
##   untuned unigram bag-of-words reaches ~0.70. Sarcasm in this corpus
##   is semantic, not typographic. The prediction is therefore that
##   Model 1 lands well above 0.70 and that Model 2 adds very little.
##   A small or null delta is the expected, publishable result here --
##   it answers the RQ, it does not fail it.
##
## Structure:
##   1. Setup
##   2. Shared feature dataset + stratified subsample
##   3. Semantic embeddings
##   4. Assemble Model 1 / Model 2 + train/test split
##   5. Train (elastic-net logistic regression, 10-fold CV)
##   6. Evaluate (Accuracy, Precision, Recall, F1, AUC)
##   7. Compare Model 1 vs Model 2 (DeLong test)
##   8. Figures
## ==================================================================


# ==================================================================
# 1. SETUP
# ==================================================================
library(tidyverse)
library(caret)         # train(), confusionMatrix(), createDataPartition()
library(glmnet)        # elastic-net engine used by caret

if (!requireNamespace("text", quietly = TRUE)) install.packages("text")
library(text)          # textEmbed(), for the semantic-embedding features

if (!requireNamespace("pROC", quietly = TRUE)) install.packages("pROC")
library(pROC)          # ROC curves / AUC / DeLong test

if (!requireNamespace("doParallel", quietly = TRUE)) install.packages("doParallel")
library(doParallel)    # parallel CV folds

# The shared psycholinguistic feature definitions -- same file RQ1 uses.
source("features_psycholinguistic.R")

# One-time setup for the {text} package's Python backend (torch /
# transformers). Uncomment on first use in a fresh environment:
# reticulate::install_miniconda()
# textrpp_install()
text::textrpp_initialize(save_profile = TRUE)

EMBEDDING_MODEL <- "microsoft/harrier-oss-v1-270m"


# ==================================================================
# 2. SHARED FEATURE DATASET + STRATIFIED SUBSAMPLE
# ==================================================================
# build_feature_dataset() returns the full cleaned corpus (~51k rows)
# with all 12 psycholinguistic features already computed and cached.
# RQ1 uses all of it. RQ2 subsamples, because embedding 2 x 51k texts
# through a 270M-parameter transformer on a laptop is not practical --
# and because every debugging cycle would otherwise cost hours.
#
# Sizing: the number that matters is the TEST set, since that is what
# the DeLong test operates on. N_SUBSAMPLE = 20,000 gives ~4,000 test
# rows, enough to detect a Delta AUC of roughly 0.01-0.015 -- about the
# smallest gain worth claiming. Do NOT drop below ~10,000 total: at
# that point a real psycholinguistic contribution becomes
# indistinguishable from noise, which would sink the RQ.
#
# Set to Inf to use the full corpus. Use 5000 while developing.
N_SUBSAMPLE <- 20000

sarcasm_full <- build_feature_dataset()

set.seed(123)
sarcasm_clean <- if (is.finite(N_SUBSAMPLE) && N_SUBSAMPLE < nrow(sarcasm_full)) {
  # Stratified on label so the subsample keeps the corpus's balance.
  sarcasm_full %>%
    group_by(label_f) %>%
    slice_sample(n = ceiling(N_SUBSAMPLE / 2)) %>%
    ungroup()
} else {
  sarcasm_full
}

cat("\nRQ2 working set:", nrow(sarcasm_clean), "rows",
    sprintf("(%.0f%% of the full corpus)\n",
            100 * nrow(sarcasm_clean) / nrow(sarcasm_full)))
print(table(sarcasm_clean$label_f))

# The subsample is a random stratified draw from the same cleaned
# corpus RQ1 describes, so RQ1's descriptive statistics still apply
# here. Print the feature means side by side to confirm that, and put
# the check in the appendix -- it is what licenses comparing the two
# RQs' results at all.
cat("\n===== Subsample vs full corpus: feature means =====\n")
print(
  bind_rows(
    sarcasm_full  %>% summarise(across(all_of(PSYCH_FEATURES), mean)) %>%
      mutate(set = "full corpus", .before = 1),
    sarcasm_clean %>% summarise(across(all_of(PSYCH_FEATURES), mean)) %>%
      mutate(set = "RQ2 subsample", .before = 1)
  ) %>% mutate(across(where(is.numeric), ~round(.x, 4)))
)


# ==================================================================
# 3. SEMANTIC EMBEDDINGS
# ==================================================================
# Document-level (mean-pooled) embeddings for the comment and for the
# parent comment. Both feed Model 1 -- see design note (b) at the top.
#
# Cached to disk because this is by far the slowest step. The cache
# key includes the row count, so changing N_SUBSAMPLE does not
# silently reuse embeddings computed for a different set of rows.
emb_cache <- sprintf("embeddings_n%d.rds", nrow(sarcasm_clean))

if (file.exists(emb_cache)) {
  cat("\nLoading cached embeddings from", emb_cache, "\n")
  emb <- readRDS(emb_cache)
  emb_comment <- emb$comment
  emb_parent  <- emb$parent
} else {
  cat("\nEmbedding", nrow(sarcasm_clean), "comments with", EMBEDDING_MODEL,
      "-- this is the slow step...\n")
  embeddings_comment <- textEmbed(
    texts = sarcasm_clean$comment,
    model = EMBEDDING_MODEL,
    aggregation_from_tokens_to_texts = "mean",
    keep_token_embeddings = FALSE,
    layers = -2
  )

  cat("Embedding", nrow(sarcasm_clean), "parent comments...\n")
  embeddings_parent <- textEmbed(
    texts = sarcasm_clean$parent_comment,
    model = EMBEDDING_MODEL,
    aggregation_from_tokens_to_texts = "mean",
    keep_token_embeddings = FALSE,
    layers = -2
  )

  emb_comment <- embeddings_comment$texts$texts   # tibble: Dim1 ... DimN
  emb_parent  <- embeddings_parent$texts$texts

  saveRDS(list(comment = emb_comment, parent = emb_parent), emb_cache)
  cat("Cached embeddings to", emb_cache, "\n")
}

cat("Embedding dimensions:", ncol(emb_comment), "per text\n")

# --- Semantic incongruity: 1 - cosine(comment, parent) ---------------
# Topical/semantic mismatch between a reply and what it replies to, as
# opposed to the sentiment-based parent_incongruity in the
# psycholinguistic block. This is an EMBEDDING feature (it is computed
# entirely from the embeddings), so it belongs to Model 1.
cosine_sim <- function(a, b) sum(a * b) / (sqrt(sum(a^2)) * sqrt(sum(b^2)))

mat_c <- as.matrix(emb_comment)
mat_p <- as.matrix(emb_parent)
semantic_incong <- 1 - rowSums(mat_c * mat_p) /
  (sqrt(rowSums(mat_c^2)) * sqrt(rowSums(mat_p^2)))


# ==================================================================
# 4. ASSEMBLE MODEL DATASETS + TRAIN/TEST SPLIT
# ==================================================================
# One stratified 80/20 split, reused identically for both models, so
# Model 1 and Model 2 are evaluated on the exact same test rows. This
# is also what makes the DeLong test valid -- it compares two ROC
# curves computed on the same cases.
set.seed(123)
train_idx  <- createDataPartition(sarcasm_clean$label_f, p = 0.8, list = FALSE)
split_type <- rep("test", nrow(sarcasm_clean))
split_type[train_idx] <- "train"

cat("\nTrain/test split:\n")
print(table(split_type, sarcasm_clean$label_f))

# Prefix embedding columns so they cannot collide with feature names
# and so the figures can tell the two blocks apart.
colnames(emb_comment) <- paste0("EmbC_", colnames(emb_comment))
colnames(emb_parent)  <- paste0("EmbP_", colnames(emb_parent))

# --- The embedding block (Model 1) -----------------------------------
embedding_block <- bind_cols(emb_comment, emb_parent) %>%
  mutate(semantic_incong = semantic_incong)

# --- The psycholinguistic block (what Model 2 adds) ------------------
# Straight from PSYCH_FEATURES -- identical to RQ1's predictor set.
psych_block <- sarcasm_clean %>% select(all_of(PSYCH_FEATURES))

cat("\nBlock sizes -> embeddings:", ncol(embedding_block),
    "| psycholinguistic:", ncol(psych_block), "\n")

model1_data <- bind_cols(embedding_block,
                         label = sarcasm_clean$label_f)
model2_data <- bind_cols(embedding_block, psych_block,
                         label = sarcasm_clean$label_f)

model1_train <- model1_data[split_type == "train", ]
model1_test  <- model1_data[split_type == "test",  ]
model2_train <- model2_data[split_type == "train", ]
model2_test  <- model2_data[split_type == "test",  ]


# ==================================================================
# 5. TRAIN CLASSIFIERS
# ==================================================================
# Elastic-net logistic regression (glmnet via caret), 10-fold CV.
# Same algorithm, same CV scheme, same tuning grid, same seed for both
# models -- so any difference reflects the added FEATURES, not the
# modelling. Regularisation matters here: with ~1,500 embedding
# dimensions, unpenalised logistic regression would overfit badly.
#
# NOTE on centring/scaling: preProcess is essential, because the
# psycholinguistic features are on wildly different scales from the
# embedding dimensions (raw counts vs. small floats). Without it the
# elastic-net penalty would fall on them unevenly and Model 2 could
# look worse than Model 1 for purely numerical reasons.
cl <- makePSOCKcluster(max(parallel::detectCores() - 1, 1))
registerDoParallel(cl)
on.exit({ stopCluster(cl); registerDoSEQ() }, add = TRUE)

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
cat("Training Model 2 (embeddings + psycholinguistic)...\n")
model2_fit <- train(
  label ~ ., data = model2_train,
  method = "glmnet", family = "binomial",
  trControl = ctrl, metric = "ROC",
  preProcess = c("center", "scale"),
  tuneLength = 5
)

stopCluster(cl)
registerDoSEQ()

print(model1_fit)
print(model2_fit)


# ==================================================================
# 6. EVALUATE
# ==================================================================
m1_pred_class <- predict(model1_fit, newdata = model1_test)
m1_pred_prob  <- predict(model1_fit, newdata = model1_test, type = "prob")$Sarcastic
cfm_model1 <- confusionMatrix(m1_pred_class, model1_test$label,
                              positive = "Sarcastic", mode = "everything")
print(cfm_model1)
roc_model1 <- roc(response = model1_test$label, predictor = m1_pred_prob,
                  levels = c("NonSarcastic", "Sarcastic"), quiet = TRUE)

m2_pred_class <- predict(model2_fit, newdata = model2_test)
m2_pred_prob  <- predict(model2_fit, newdata = model2_test, type = "prob")$Sarcastic
cfm_model2 <- confusionMatrix(m2_pred_class, model2_test$label,
                              positive = "Sarcastic", mode = "everything")
print(cfm_model2)
roc_model2 <- roc(response = model2_test$label, predictor = m2_pred_prob,
                  levels = c("NonSarcastic", "Sarcastic"), quiet = TRUE)

auc_model1 <- as.numeric(auc(roc_model1))
auc_model2 <- as.numeric(auc(roc_model2))

# DeLong test for two CORRELATED ROC curves (same test rows, two
# models). This is the actual statistical answer to the RQ.
delong_test <- roc.test(roc_model1, roc_model2, method = "delong")
print(delong_test)


# ==================================================================
# 7. COMPARE MODEL 1 vs MODEL 2
# ==================================================================
extract_metrics <- function(cfm, roc_obj, model_name) {
  ci <- ci.auc(roc_obj)
  tibble(
    Model     = model_name,
    Accuracy  = cfm$overall["Accuracy"],
    Precision = cfm$byClass["Precision"],
    Recall    = cfm$byClass["Recall"],
    F1        = cfm$byClass["F1"],
    AUC       = as.numeric(auc(roc_obj)),
    AUC_lo    = ci[1],
    AUC_hi    = ci[3]
  )
}

comparison_table <- bind_rows(
  extract_metrics(cfm_model1, roc_model1, "Model 1: Embeddings only"),
  extract_metrics(cfm_model2, roc_model2, "Model 2: Embeddings + Psycholinguistic")
)

cat("\n===== MODEL COMPARISON =====\n")
print(comparison_table %>% mutate(across(where(is.numeric), ~round(.x, 4))))

cat(sprintf("\nDelta AUC (Model 2 - Model 1): %+.4f\n", auc_model2 - auc_model1))
cat(sprintf("DeLong test p-value: %.4f\n", delong_test$p.value))
if (delong_test$p.value < 0.05) {
  cat("=> The psycholinguistic features change discrimination ability\n",
      "   by a statistically significant amount.\n")
} else {
  cat("=> No significant change. Given RQ1 (these features reach only\n",
      "   AUC ~0.59 alone, vs ~0.70 for plain unigrams), this is the\n",
      "   EXPECTED result: sarcasm here is semantic, not typographic.\n",
      "   Report it as an answer, not a failure.\n")
}


# ==================================================================
# 8. FIGURES
# ==================================================================
fig_dir <- "figures"
if (!dir.exists(fig_dir)) dir.create(fig_dir)

theme_report <- theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))
pal <- c("Model 1: Embeddings only" = "#0072B2",
         "Model 2: Embeddings + Psycholinguistic" = "#D55E00")

# --- FIGURE 1: metric comparison -------------------------------------
comparison_long <- comparison_table %>%
  select(Model, Accuracy, Precision, Recall, F1, AUC) %>%
  pivot_longer(-Model, names_to = "Metric", values_to = "Value")

fig_r2_1 <- ggplot(comparison_long, aes(Metric, Value, fill = Model)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.7), vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = pal) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(title = "Does Adding Psycholinguistic Features Help?",
       subtitle = sprintf("Delta AUC = %+.4f, DeLong p = %.4f",
                          auc_model2 - auc_model1, delong_test$p.value),
       x = NULL, y = "Score") +
  theme_report +
  theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "fig_rq2_1_metric_comparison.png"), fig_r2_1,
       width = 10, height = 6, dpi = 300)
print(fig_r2_1)

# --- FIGURE 2: ROC overlay -------------------------------------------
roc_df <- bind_rows(
  tibble(FPR = 1 - roc_model1$specificities, TPR = roc_model1$sensitivities,
         Model = "Model 1: Embeddings only"),
  tibble(FPR = 1 - roc_model2$specificities, TPR = roc_model2$sensitivities,
         Model = "Model 2: Embeddings + Psycholinguistic")
)

fig_r2_2 <- ggplot(roc_df, aes(FPR, TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  scale_color_manual(values = pal) +
  labs(title = "ROC Curves: Sarcasm Classifier Comparison",
       subtitle = sprintf("AUC %.3f vs %.3f | DeLong p = %.4f",
                          auc_model1, auc_model2, delong_test$p.value),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_report + theme(legend.position = "bottom")

ggsave(file.path(fig_dir, "fig_rq2_2_roc_comparison.png"), fig_r2_2,
       width = 7, height = 6, dpi = 300)
print(fig_r2_2)

# --- FIGURE 3: confusion matrices ------------------------------------
cfm_to_df <- function(cfm, model_name) {
  as.data.frame(cfm$table) %>%
    rename(Predicted = Prediction, Actual = Reference) %>%
    mutate(Model = model_name)
}
cfm_df <- bind_rows(
  cfm_to_df(cfm_model1, "Model 1: Embeddings only"),
  cfm_to_df(cfm_model2, "Model 2: Embeddings + Psycholinguistic")
)

fig_r2_3 <- ggplot(cfm_df, aes(Predicted, Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "#f0f0f0", high = "#0072B2") +
  facet_wrap(~ Model) +
  labs(title = "Confusion Matrices") +
  theme_report + theme(legend.position = "none")

ggsave(file.path(fig_dir, "fig_rq2_3_confusion_matrices.png"), fig_r2_3,
       width = 10, height = 5, dpi = 300)
print(fig_r2_3)

# --- FIGURE 4: where do the psycholinguistic features rank? ----------
# The key diagnostic figure. If the psycholinguistic features are
# buried among thousands of embedding dimensions, that visually
# explains a null delta in section 7.
imp2 <- varImp(model2_fit)$importance %>%
  rownames_to_column("Feature") %>%
  mutate(Block = if_else(Feature %in% PSYCH_FEATURES,
                         "Psycholinguistic", "Embedding")) %>%
  arrange(desc(Overall))

psych_ranks <- imp2 %>%
  mutate(rank = row_number()) %>%
  filter(Block == "Psycholinguistic")

cat("\n===== Where the psycholinguistic features rank in Model 2 =====\n")
cat("(out of", nrow(imp2), "total predictors)\n")
print(psych_ranks %>% select(rank, Feature, Overall))

fig_r2_4 <- imp2 %>%
  slice_head(n = 30) %>%
  ggplot(aes(reorder(Feature, Overall), Overall, fill = Block)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Embedding" = "#0072B2",
                               "Psycholinguistic" = "#D55E00")) +
  labs(title = "Model 2: Top 30 Predictors by Importance",
       subtitle = sprintf("%d of %d psycholinguistic features rank in the top 30",
                          sum(psych_ranks$rank <= 30), nrow(psych_ranks)),
       x = NULL, y = "Importance") +
  theme_report

ggsave(file.path(fig_dir, "fig_rq2_4_feature_importance.png"), fig_r2_4,
       width = 9, height = 8, dpi = 300)
print(fig_r2_4)

cat("\nRQ2 figures saved to", normalizePath(fig_dir), "\n")
