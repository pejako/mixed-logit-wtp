#' MNL elasticities at a representative design
#'
#' Closed-form expressions matching the Python implementation:
#'   eta_ii =  beta_p * p_i * (1 - P_i)
#'   eta_ij = -beta_p * p_j * P_j   for i != j
#'
#' @param mnl_result Output of `fit_mnl_r`.
#' @param design A numeric matrix with one row per alternative and one
#'   column per attribute, in the same order as `mnl_result$coef_names`.
#' @param price_attr Character. Name of the price attribute.
#' @param alt_labels Optional character vector of alternative names.
#' @return A list with `matrix` (n_alt x n_alt elasticities) and `alt_labels`.
#' @export
mnl_elasticities_r <- function(mnl_result,
                               design,
                               price_attr = "price",
                               alt_labels = NULL) {
  beta <- mnl_result$coefficients
  names(beta) <- mnl_result$coef_names

  if (!(price_attr %in% names(beta))) {
    stop(sprintf("Price attribute '%s' not in coefficients: %s",
                 price_attr, paste(names(beta), collapse = ", ")))
  }
  beta_p <- unname(beta[price_attr])
  p_idx <- which(names(beta) == price_attr)
  prices <- design[, p_idx]
  n_alt <- nrow(design)

  # Compute representative shares
  V <- as.numeric(design %*% beta)
  V <- V - max(V)
  P <- exp(V) / sum(exp(V))

  E <- matrix(0, n_alt, n_alt)
  for (i in seq_len(n_alt)) {
    for (j in seq_len(n_alt)) {
      if (i == j) {
        E[i, j] <- beta_p * prices[i] * (1 - P[i])
      } else {
        E[i, j] <- -beta_p * prices[j] * P[j]
      }
    }
  }

  if (is.null(alt_labels)) {
    alt_labels <- paste("Alt", seq_len(n_alt))
  }
  rownames(E) <- alt_labels
  colnames(E) <- alt_labels
  list(matrix = E, alt_labels = alt_labels, model = "MNL")
}


#' MXL elasticities at a representative design (simulation-based)
#'
#' Computed by drawing from the estimated population coefficient
#' distribution and averaging the per-draw choice-probability gradients.
#' Mirrors the Python implementation algorithm-for-algorithm.
#'
#' @param mxl_result Output of `fit_mxl_r`.
#' @param design A numeric matrix (n_alt x n_attr) at which to evaluate
#'   elasticities. Column order must match the order in which attributes
#'   appear in `mxl_result$coef_names_python` (i.e., random first, then fixed).
#' @param price_attr Character. Name of the price attribute.
#' @param n_draws Number of Halton draws for the simulation.
#' @param halton_seed Integer seed for Halton scrambling.
#' @param alt_labels Optional character vector of alternative names.
#' @return A list with `matrix` and `alt_labels`.
#' @export
mxl_elasticities_r <- function(mxl_result,
                               design,
                               price_attr = "price",
                               n_draws = 2000,
                               halton_seed = 0,
                               alt_labels = NULL) {

  # Reconstruct the per-draw beta matrix from the MXL fit.
  random_attrs <- mxl_result$random_attrs
  fixed_attrs <- mxl_result$fixed_attrs
  coefs <- mxl_result$coefficients
  names(coefs) <- mxl_result$coef_names

  attr_names <- c(names(random_attrs), fixed_attrs)
  n_attr <- length(attr_names)
  if (ncol(design) != n_attr) {
    stop(sprintf("design must have %d columns (got %d)", n_attr, ncol(design)))
  }

  # Standard normal draws for each random coefficient
  set.seed(halton_seed)
  z <- matrix(stats::qnorm(stats::runif(n_draws * length(random_attrs))),
              nrow = n_draws,
              ncol = length(random_attrs))

  # Build (n_draws, n_attr) beta matrix
  betas <- matrix(0, nrow = n_draws, ncol = n_attr)
  colnames(betas) <- attr_names

  z_col <- 1
  for (i in seq_along(attr_names)) {
    nm <- attr_names[i]
    if (nm %in% names(random_attrs)) {
      mean_val <- coefs[nm]
      sd_val <- coefs[paste0("sd.", nm)]
      if (random_attrs[[nm]] == "n") {
        betas[, i] <- mean_val + sd_val * z[, z_col]
      } else if (random_attrs[[nm]] == "ln") {
        betas[, i] <- exp(mean_val + sd_val * z[, z_col])
      } else {
        stop(sprintf("Unsupported distribution for %s: %s",
                     nm, random_attrs[[nm]]))
      }
      z_col <- z_col + 1
    } else {
      betas[, i] <- coefs[nm]
    }
  }

  p_idx <- which(attr_names == price_attr)
  prices <- design[, p_idx]
  n_alt <- nrow(design)

  # Per-draw probabilities at the design
  V <- design %*% t(betas)        # (n_alt x n_draws)
  V <- t(V)                        # (n_draws x n_alt)
  V <- V - apply(V, 1, max)
  expV <- exp(V)
  P <- expV / rowSums(expV)        # (n_draws x n_alt)
  Pbar <- colMeans(P)

  beta_p_per_draw <- betas[, p_idx]

  E <- matrix(0, n_alt, n_alt)
  for (i in seq_len(n_alt)) {
    for (j in seq_len(n_alt)) {
      if (i == j) {
        inner <- mean(beta_p_per_draw * P[, i] * (1 - P[, i]))
        E[i, j] <- (prices[i] / Pbar[i]) * inner
      } else {
        inner <- mean(beta_p_per_draw * P[, i] * P[, j])
        E[i, j] <- -(prices[j] / Pbar[i]) * inner
      }
    }
  }

  if (is.null(alt_labels)) {
    alt_labels <- paste("Alt", seq_len(n_alt))
  }
  rownames(E) <- alt_labels
  colnames(E) <- alt_labels
  list(matrix = E, alt_labels = alt_labels, model = "MXL")
}
