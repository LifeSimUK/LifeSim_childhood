######################################################################
############################## v1.0 2024/08/19 #######################
######################################################################
##### Function to initialise lists for Results
init_results <- function(out, n_imp, bout, tab = "NA")  {
  #out          - List of outcome variables
  #n_imp        - Number of imputations
  #bout         - List of binary outcome variables
  #wt           - weights for regression
  #tab          - output type for table ("NA"    - None,
  #                                      "dydx"  - Marginal effects (unweighted)
  #                                      "rrr"   - Relative risk ratio (Binary outcomes only))
  
  # print(tab)
  m_beta <- vector("list", length(out))
  names(m_beta) <- out
  m_se <- vector("list", length(out))
  names(m_se) <- out
  m_res <- vector("list", length(out))
  names(m_res) <- out
  
  # Initialize lists for each outcome in the main list
  for (outcome in out) {
    m_beta[[outcome]] <- vector("list", n_imp)
    m_se[[outcome]] <- vector("list", n_imp)
    m_res[[outcome]] <- vector("list", n_imp)
  }
  # Assign results to global environment
  assign("m_beta", m_beta, envir = .GlobalEnv)
  assign("m_se", m_se, envir = .GlobalEnv)
  assign("m_res", m_res, envir = .GlobalEnv)
  
  if (is.character(tab) && tab == "dydx"){
    d_beta <- vector("list", length(out))
    names(d_beta) <- out
    d_se <- vector("list", length(out))
    names(d_se) <- out
    
    # Initialize lists for each outcome in the main list
    for (outcome in out) {
      d_beta[[outcome]] <- vector("list", n_imp)
      d_se[[outcome]] <- vector("list", n_imp)
    }
    # Assign results to global environment
    assign("d_beta", d_beta, envir = .GlobalEnv)
    assign("d_se", d_se, envir = .GlobalEnv)
  } else if (is.character(tab) && tab == "rrr"){
    m_rre <- vector("list", length(out))
    names(m_beta) <- out
    m_rrl <- vector("list", length(out))
    names(m_se) <- out
    m_rrh <- vector("list", length(out))
    names(m_res) <- out
    
    # Initialize lists for each outcome in the main list
    for (outcome in out) {
      d_rre[[outcome]] <- vector("list", n_imp)
      d_rrh[[outcome]] <- vector("list", n_imp)
      d_rrl[[outcome]] <- vector("list", n_imp)
    }
    # Assign results to global environment
    assign("d_rre", d_rre, envir = .GlobalEnv)
    assign("d_rrh", d_rrh, envir = .GlobalEnv)
    assign("d_rrl", d_rrl, envir = .GlobalEnv)
  } else {}
}  