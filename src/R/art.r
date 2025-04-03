# Load required packages
library(ARTool)
library(emmeans)
library(dplyr)

analyze_tda_data <- function(tda_data) {
  # Convert all variables to factors
  tda_data$image_model <- factor(tda_data$image_model)
  tda_data$caption_model <- factor(tda_data$caption_model)
  tda_data$prompt <- factor(tda_data$prompt)
  tda_data$homology_dimension <- factor(tda_data$homology_dimension)

  # Check if all factors have at least 2 levels
  if (length(levels(tda_data$image_model)) < 2) {
    stop("Error: image_model must have at least 2 levels")
  }
  if (length(levels(tda_data$caption_model)) < 2) {
    stop("Error: caption_model must have at least 2 levels")
  }
  if (length(levels(tda_data$prompt)) < 2) {
    stop("Error: prompt must have at least 2 levels")
  }

  # For each homology dimension (showing for dimension 0)
  dim0_data <- tda_data %>% filter(homology_dimension == 0)

  # Full factorial model with all interactions
  # ARTool requires all possible interaction terms between the fixed effects
  m.art <- art(entropy ~ image_model * caption_model * prompt, data = dim0_data)

  # Check the ANOVA results
  anova_results <- anova(m.art)
  print(anova_results)

  # Run contrast tests for main effects
  # For image_model
  image_contrasts <- art.con(m.art, "image_model")
  print(image_contrasts)

  # For caption_model
  caption_contrasts <- art.con(m.art, "caption_model")
  print(caption_contrasts)

  # For prompt (if you want to examine differences between prompts)
  prompt_contrasts <- art.con(m.art, "prompt")
  print(prompt_contrasts)

  # For interaction between image_model and caption_model (pairwise combinations)
  interaction_contrasts <- art.con(m.art, "image_model:caption_model")
  print(interaction_contrasts)

  # For differences of differences in interactions
  diff_of_diff <- art.con(m.art, "image_model:caption_model", interaction = TRUE)
  print(diff_of_diff)

  # Return results as a list
  return(list(
    model = m.art,
    anova = anova_results,
    contrasts = list(
      image_model = image_contrasts,
      caption_model = caption_contrasts,
      prompt = prompt_contrasts,
      interaction = interaction_contrasts,
      diff_of_diff = diff_of_diff
    )
  ))
}
