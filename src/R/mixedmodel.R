# run this file with `Rscript src/R/mixedmodel.R` from the top-level of the project
# results will be printed to stdout

library(arrow)
library(lme4)
library(lmerTest)  # For p-values in mixed models

# Load the parquet file into a dataframe
runs_df <- read_parquet("output/cache/runs.parquet")


# Full model with all main effects and interactions
full_model <- lmer(entropy ~ text_model * image_model * embedding_model +
                   (1|initial_prompt),
                   data = runs_df)

# Get summary with p-values
summary(full_model)
