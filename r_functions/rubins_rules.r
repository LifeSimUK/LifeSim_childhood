######################################################################
############################## v1.0 2024/08/19 #######################
######################################################################
##### Rubins rules for coefficients and SEs
rubins_rules <- function(m_beta, m_se) {
  #m_beta       - List of lists of Coefficients
  #m_se         - List of lists of Standard errors

  ### Mean Coefficients 
  lst_beta <- map(m_beta, ~ reduce(.x, `+`) / length(.x))
  
  ##Rename coefficients for transformation
  # Function to strip "outcome_" from names
  strip_outcome_prefix <- function(names_vector) {
    sub("^[^_]*_", "", names_vector)  # Removes anything before and including the first underscore
  }
  
  # Apply this function to each element's names
  lst_beta <- lapply(lst_beta, function(x) {
    names(x) <- strip_outcome_prefix(names(x))
    x
  })
  ## Convert list into dataframe
  # Convert the list into a tibble, where each element is a row
  tib_beta <- enframe(lst_beta, name = "outcome", value = "coefficients")
  
  # Unnest the coefficients so that each one becomes a column
  beta <- tib_beta %>%
    mutate(coefficients = map(coefficients, ~as.data.frame(t(.x)))) %>%
    unnest_wider(coefficients)
  
  ### Standard errors
  ## Within imputation variance
  wiv <- list()
  
  # Loop through each outcome
  for (outcome in names(m_se)) {
    # Retrieve the list of coefficient sets for the current outcome
    coefficient_sets <- m_se[[outcome]]
    
    # Initialize a vector to store the sum of squares
    squared_sums <- numeric(length = length(coefficient_sets[[1]]))
    
    # Loop through each set of coefficients
    for (coefficients in coefficient_sets) {
      # Square the errors
      squared_sum <- coefficients^2
      # Accumulate the sum of squared errors for each coefficient
      squared_sums <- squared_sums + squared_sum
    }
    
    # Store the sum of squared differences for the current outcome
    wiv[[outcome]] <- squared_sums / length(m_se[[outcome]])
  }
  
  ## Between imputation variance
  # Initialize a list to store the sum of squared differences for each outcome
  biv <- list()
  
  # Loop through each outcome
  for (outcome in names(m_beta)) {
    # Retrieve the list of coefficient sets for the current outcome
    coefficient_sets <- m_beta[[outcome]]
    # Retrieve the mean coefficients for the current outcome
    mean_coefficients <- lst_beta[[outcome]]
    
    # Initialize a vector to store the sum of squared differences for each coefficient
    squared_diffs <- numeric(length = length(mean_coefficients))
    
    # Loop through each set of coefficients
    for (coefficients in coefficient_sets) {
      # Subtract the mean coefficients from the current set of coefficients
      differences <- coefficients - mean_coefficients
      # Square the differences
      squared_differences <- differences^2
      # Accumulate the sum of squared differences for each coefficient
      squared_diffs <- squared_diffs + squared_differences
    }
    
    # Store the sum of squared differences for the current outcome
    biv[[outcome]] <- squared_diffs / (length(m_beta[[outcome]]) - 1)
  }
  
  ## Pooled Standard errors
  # Initialize a list to store the final results
  lst_se <- list()
  
  # Loop through each outcome
  for (outcome in names(biv)) {
    # Retrieve the sum of squared differences for the current outcome
    tbiv <- biv[[outcome]]
    # Retrieve the variances for the current outcome
    twiv <- wiv[[outcome]]
    
    # Calculate the square root of the sum of squared differences and variances
    lst_se[[outcome]] <- sqrt(twiv + tbiv + (tbiv/length(m_beta[[outcome]])))
  }
  
  ##Rename errors for transformation
  ##Rename coefficients for transformation
  # Function to strip "outcome_" from names
  strip_outcome_prefix <- function(names_vector) {
    sub("^[^_]*_", "", names_vector)  # Removes anything before and including the first underscore
  }
  
  # Apply this function to each element's names
  lst_se <- lapply(lst_se, function(x) {
    names(x) <- strip_outcome_prefix(names(x))
    x
  })
  ## Convert list into dataframe
  # Convert the list into a tibble, where each element is a row
  tib_se <- enframe(lst_se, name = "outcome", value = "coefficients")
  
  # Unnest the coefficients so that each one becomes a column
  se <- tib_se %>%
    mutate(coefficients = map(coefficients, ~as.data.frame(t(.x)))) %>%
    unnest_wider(coefficients)
  
  return(list(beta = beta, se = se))
}