# Load required packages
library(ARTool)
library(emmeans)
library(dplyr)

analyze_tda_data <- function(tda_data, dimension) {
  tda_data$image_model <- factor(tda_data$image_model)
  tda_data$text_model <- factor(tda_data$text_model)
  tda_data$initial_prompt <- factor(tda_data$initial_prompt)
  tda_data$homology_dimension <- factor(tda_data$homology_dimension)

  # Check if all factors have at least 2 levels
  if (length(levels(tda_data$image_model)) < 2) {
    stop("Error: image_model must have at least 2 levels")
  }
  if (length(levels(tda_data$text_model)) < 2) {
    stop("Error: text_model must have at least 2 levels")
  }
  if (length(levels(tda_data$initial_prompt)) < 2) {
    stop("Error: initial_prompt must have at least 2 levels")
  }

  # For each homology dimension (showing for dimension 0)
  dim_data <- tda_data %>% filter(homology_dimension == dimension)
  print(dim(dim_data))

  # Full factorial model with all interactions
  # ARTool requires all possible interaction terms between the fixed effects
  m.art <- art(entropy ~ image_model * text_model * initial_prompt, data = dim_data)

  # Check the ANOVA results
  anova_results <- anova(m.art)
  print(anova_results)

  # Run contrast tests for main effects
  # For image_model
  image_contrasts <- art.con(m.art, "image_model")
  print(image_contrasts)

  # For text_model
  caption_contrasts <- art.con(m.art, "text_model")
  print(caption_contrasts)

  # For initial_prompt (if you want to examine differences between initial_prompts)
  initial_prompt_contrasts <- art.con(m.art, "initial_prompt")
  print(initial_prompt_contrasts)

  # For interaction between image_model and text_model (pairwise combinations)
  interaction_contrasts <- art.con(m.art, "image_model:text_model")
  print(interaction_contrasts)

  # For differences of differences in interactions
  diff_of_diff <- art.con(m.art, "image_model:text_model", interaction = TRUE)
  print(diff_of_diff)

  # Return results as a list
  return(list(
    model = m.art,
    anova = anova_results,
    contrasts = list(
      image_model = image_contrasts,
      text_model = caption_contrasts,
      initial_prompt = initial_prompt_contrasts,
      interaction = interaction_contrasts,
      diff_of_diff = diff_of_diff
    )
  ))
}
