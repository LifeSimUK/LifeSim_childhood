######################################################################
############################## v1.0 2024/08/19 #######################
######################################################################
##### Functions to extract Coefficients, SEs and Residuals - v1.0 2024/08/19
extract_results <- function(formula, data, outcome_name, wt="wt_uk2", reg = "linear", tab = "NA") {
  #formula      - Formula for regression
  #data         - Data used
  #outcome_name - outcome
  #wt           - weights for regression
  #reg          - regression type ("linear"    - linear,
  #                                "negbinom"  - negative binomial
  #                                "logit"     - logistic)
  #tab          - output type for table ("NA"    - None,
  #                                      "dydx"  - Marginal effects (unweighted)
  #                                      "rrr"   - Relative risk ratio (Binary outcomes only))
  # Extract weights
  weights <- as.vector(t(data_subset[[as.character(wt)]]))
  wts <- as.vector(weights)
  
  #Run regressions
  if (reg == "linear"){
    # model <- lm(formula, data = data, weights = wts)
    model <- do.call(
      lm,
      list(
        formula = formula,
        data = data,
        weights = wts
      )
    )
  } else if (reg == "negbinom"){
    # model <- glm.nb(formula, data = data, weights = wts)
    model <- do.call(
      glm.nb,
      list(
        formula = formula,
        data = data,
        weights = wts
      )
    )
    if (tab == "dydx"){
      # mdl <- negbinmfx(formula, data = data_subset, atmean = TRUE, robust = TRUE)
      # mdl <- lm(formula, data = data, weights = wts)
      mdl <- do.call(
        lm,
        list(
          formula = formula,
          data = data,
          weights = wts
        )
      )
    } else {}
    
  } else if (reg == "logit"){
    # model <- glm(formula, data = data, weights = wts, family = binomial)
    model <- do.call(
      glm,
      list(
        formula = formula,
        data = data,
        weights = wts,
        family = binomial
      )
    )
    if (tab == "dydx"){
      mdl <- marginaleffects::avg_comparisons(model = model, comparison = 'dydx', conf_level = 0.9)
      b <- mdl$estimate
      s <- mdl$std.error
    } else if (tab == "rrr"){
      mdl <- marginaleffects::avg_comparisons(model = model, comparison = 'lnratioavgwts', transform = exp, conf_level = 0.9)
      rre <- mdl$estimate
      rrl <- mdl$conf.low  # CI Lower bound
      rrh <- mdl$conf.high  # CI Upper bound
    } else {}
  } else {}
  
  # Extract coefficient, SEs and residuals for simulation
  coefs <- summary(model)$coefficients
  beta <- coefs[, 1]  # Coefficients
  se <- coefs[, 2]  # Standard errors
  residuals <- residuals(model)
  named_residuals <- setNames(residuals, data$id)
  
  if (reg == "logit"){
    # Return coefficient, SEs and residuals
    if (tab == "dydx"){
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals,
                  b = setNames(b, paste(outcome_name, mdl$term, sep = "_")),
                  s = setNames(s, paste(outcome_name, mdl$term, sep = "_"))))
    } else if (tab == "rrr"){
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals,
                  rre = setNames(rre, paste(outcome_name, mdl$term, sep = "_")),
                  rrl = setNames(rrl, paste(outcome_name, mdl$term, sep = "_")),
                  rrh = setNames(rrh, paste(outcome_name, mdl$term, sep = "_"))))
    } else {
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals))
    }
  } else{  
    # Return coefficient, SEs and residuals
    if (tab == "dydx"){
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals,
                  b = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  s = setNames(se, paste(outcome_name, names(se), sep = "_"))))
    } else if (tab == "rrr"){
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals))
    } else {
      return(list(beta = setNames(beta, paste(outcome_name, names(beta), sep = "_")),
                  se = setNames(se, paste(outcome_name, names(se), sep = "_")),
                  residuals = named_residuals))
    }
  }
}