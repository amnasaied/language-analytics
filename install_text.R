###### Installing the text package ######


# Installation guide: https://r-text.org/articles/Extended_Installation_Guide.html

# LLM prompt: 

# I have a mac M4 chip Tahoe 26.3.1 (replace with your actual model). I want to install the text package in R, which can be a bit tedious sometimes. 
# You are an expert in the installation of the R text package and its Python dependencies and will guide me through all my issues.
# Please provide all solutions STEP BY STEP. This means: Give me the first step for resolving the issue and wait for my response. 


install.packages("text")
install.packages("reticulate")
install.packages("rlang")

library(text)
text::textrpp_install()

# 1. allow to install rust
# 2. install the required package dependencies in your shell/terminal before continuing(e.g., homebrew, libomp; follow the instructions)

.rs.restartR()

text::textrpp_install()

text::textrpp_install(
  update_conda = TRUE,
  force_conda = TRUE
)

text::textrpp_initialize(
)
library(text)

# Check if it works
test_embedding <- textEmbed("hello, my name is Clara")





