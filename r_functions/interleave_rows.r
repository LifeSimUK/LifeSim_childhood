######################################################################
############################## v1.0 2024/08/19 #######################
######################################################################
##### Function to interleave rows of two data frames (Vertical)
interleave_rows <- function(df1, df2) {
  #df1         - Dataframe of Coefficients
  #df2         - Dataframe of Standard errors

  #Transposing the matrices
  df1 <- df1
  df1 <- df1 %>% remove_rownames %>% column_to_rownames(var="outcome")
  df1 <- t(df1) %>% data.frame()
  df1 <- df1 %>% rownames_to_column(var="outcome")
  
  df2 <- df2
  df2 <- df2 %>% remove_rownames %>% column_to_rownames(var="outcome")
  df2 <- t(df2) %>% data.frame()
  df2 <- df2 %>% rownames_to_column(var="outcome")
  
  #Interleave
  n <- max(nrow(df1), nrow(df2))
  df1 <- rbind(df1, matrix(NA, n - nrow(df1), ncol(df1)))
  df2 <- rbind(df2, matrix(NA, n - nrow(df2), ncol(df2)))
  interleaved <- as.data.frame(matrix(NA, n * 2, ncol(df1)))
  interleaved[seq(1, n * 2, by = 2), ] <- df1
  interleaved[seq(2, n * 2, by = 2), ] <- df2
  colnames(interleaved) <- colnames(df1)
  
  ##Format output for table
  # Limit each value to 3 decimal places
  # Round all values in the interleaved data frame to 3 decimal places
  interleaved[, !colnames(interleaved) %in% "outcome"] <- round(interleaved[, !colnames(interleaved) %in% "outcome"], 3)
  
  # Remove values in the "outcome" column in every even row
  interleaved[seq(2, n * 2, by = 2), "outcome"] <- " "
  
  # Add brackets around each value in the even rows
  for (i in seq(2, n * 2, by = 2)) {
    interleaved[i, !colnames(interleaved) %in% "outcome"] <- paste0("(", interleaved[i, !colnames(interleaved) %in% "outcome"], ")")
  }
  
  return(interleaved)
}