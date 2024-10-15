library(kappalab, quietly = TRUE)
library(jsonlite)

minimum_variance <- function(alternatives, nb_criteria, preferences, k) {
  pref_kappa <- matrix(nrow = 0, ncol = 2 * nb_criteria + 1)
  n <- nrow(preferences)
  for (i in 1:n) {
    idx_a <- preferences[i,1]
    idx_b <- preferences[i,2]
    pref <- c(alternatives[idx_a,], alternatives[idx_b,], preferences[i,3])
    if (length(pref) != 2 * nb_criteria + 1) {
      stop("Error")
    }
    pref_kappa <- rbind(pref_kappa, pref)
  }
  mv <- mini.var.capa.ident(nb_criteria, k, A.Choquet.preorder = pref_kappa)
  mv
}

generalized_least_squares <- function(alternatives, nb_criteria, preferences, k, sigf_param) {
  rownames(alternatives) <- 1:nrow(alternatives)
  pref_kappa <- matrix(preferences[,1:2], ncol = 2)
  delta <- preferences[1,3]
  gls <- ls.ranking.capa.ident(nb_criteria, k, alternatives, pref_kappa, delta, sigf = sigf_param)
  gls$obj <- sum((gls$glob.eval - gls$Choquet.C)^2)
  gls
}

call_identification_func <- function(alternatives, nb_criteria, preferences, k, approach_type, sigf_param) {
  if (approach_type == "Minimum Variance") {
    return(minimum_variance(alternatives, nb_criteria, preferences, k))
  }
  if (approach_type == "Generalized Least Squares") {
    return(generalized_least_squares(alternatives, nb_criteria, preferences, k, sigf = sigf_param))
  }
  return(NULL)
}

main <- function(input) {
  alternatives <- input$alternatives
  nb_criteria <- length(alternatives[1,])
  preferences <- input$preferences
  k <- input$k
  approach_type <- input$approachType
  sigf_param <- input$significantFigures
  res <- call_identification_func(alternatives, nb_criteria, preferences, k, approach_type, sigf_param)
  interaction_indices <- interaction.indices(res$solution)
  interaction_indices[is.na(interaction_indices)] <- 0
  output <- list(capacities = res$solution@data,
                 interactionIndices = interaction_indices,
                 shapleyValues = Shapley.value(res$solution))
  if ("obj" %in% names(res)) {
    output$obj <- res$obj
  }
  output
}