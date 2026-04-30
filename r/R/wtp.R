#' Compute willingness-to-pay distributions
#'
#' Simulates WTP = -beta_k / beta_p across the estimated population
#' coefficient distribution. Mirrors the Python `compute_wtp_samples`
#' implementation and includes the same Cauchy-trap warning when the
#' price coefficient distribution allows zero crossings.
#'
#' @param mxl_result Output of `fit_mxl_r`.
#' @param price_attr Character. Name of the price attribute.
#' @param n_draws Number of Monte Carlo samples.
#' @param halton_seed Integer seed.
#' @return A named list. Each element corresponds to a non-price attribute
#'   and contains: `samples` (the WTP draws), `mean`, `median`, `std`,
#'   `quantiles` (named numeric for 5/25/50/75/95), `iqr`, and a logical
#'   `price_can_be_zero` flag.
#' @export
compute_wtp_r <- function(mxl_result,
                          price_attr = "price",
                          n_draws = 10000,
                          halton_seed = 0) {

  random_attrs <- mxl_result$random_attrs
  fixed_attrs <- mxl_result$fixed_attrs
  coefs <- mxl_result$coefficients
  names(coefs) <- mxl_result$coef_names

  attr_names <- c(names(random_attrs), fixed_attrs)
  if (!(price_attr %in% attr_names)) {
    stop(sprintf("Price '%s' not in attributes: %s",
                 price_attr, paste(attr_names, collapse = ", ")))
  }

  # Detect dangerous regime: normal price coef with mass near zero
  price_can_be_zero <- FALSE
  if (price_attr %in% names(random_attrs)) {
    if (random_attrs[[price_attr]] == "n") {
      mean_p <- coefs[price_attr]
      sd_p <- coefs[paste0("sd.", price_attr)]
      if (sd_p > 0 && abs(mean_p / sd_p) < 3) {
        price_can_be_zero <- TRUE
        warning(sprintf(
          "Price coef '%s' has mean/sd = %.2f, below safety threshold of 3. ",
          price_attr, mean_p / sd_p),
          "WTP ratios may have undefined mean/variance (Cauchy-like). ",
          "Use median, IQR, and trimmed_mean instead. ",
          "Consider re-estimating with a lognormal price coefficient."
        )
      }
    }
  } else if (price_attr %in% fixed_attrs) {
    if (abs(coefs[price_attr]) < 1e-3) price_can_be_zero <- TRUE
  }

  # Build (n_draws, n_attr) beta matrix using the same machinery as elasticities
  set.seed(halton_seed)
  z <- matrix(stats::qnorm(stats::runif(n_draws * length(random_attrs))),
              nrow = n_draws,
              ncol = length(random_attrs))
  n_attr <- length(attr_names)
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
      }
      z_col <- z_col + 1
    } else {
      betas[, i] <- coefs[nm]
    }
  }

  p_idx <- which(attr_names == price_attr)
  beta_p <- betas[, p_idx]

  out <- list()
  for (i in seq_along(attr_names)) {
    nm <- attr_names[i]
    if (i == p_idx) next
    samples <- -betas[, i] / beta_p
    qs <- stats::quantile(samples, probs = c(0.05, 0.25, 0.50, 0.75, 0.95),
                          na.rm = TRUE, names = FALSE)
    names(qs) <- c("p05", "p25", "p50", "p75", "p95")
    trimmed <- mean(samples[samples >= qs["p05"] & samples <= qs["p95"]],
                    na.rm = TRUE)
    out[[nm]] <- list(
      attr_name = nm,
      samples = samples,
      mean = mean(samples, na.rm = TRUE),
      median = stats::median(samples, na.rm = TRUE),
      std = stats::sd(samples, na.rm = TRUE),
      trimmed_mean_5pct = trimmed,
      quantiles = qs,
      iqr = unname(qs["p75"] - qs["p25"]),
      price_can_be_zero = price_can_be_zero
    )
  }
  out
}


#' Feature preference ranking
#'
#' Builds a comparable ranking table to the Python `feature_preference_ranking`.
#' Default sort is by population median (robust to Cauchy-tail effects).
#'
#' @param wtps Output of `compute_wtp_r`.
#' @param by Character. One of `"median"`, `"mean"`, `"trimmed_mean_5pct"`.
#' @return A data.frame with one row per non-price attribute, sorted descending.
#' @export
feature_ranking_r <- function(wtps, by = "median") {
  valid <- c("median", "mean", "trimmed_mean_5pct")
  if (!(by %in% valid)) {
    stop(sprintf("by must be one of %s, got '%s'",
                 paste(shQuote(valid), collapse = ", "), by))
  }
  rows <- lapply(wtps, function(w) {
    data.frame(
      attribute = w$attr_name,
      mean = w$mean,
      median = w$median,
      trimmed_mean_5pct = w$trimmed_mean_5pct,
      std = w$std,
      p05 = w$quantiles["p05"],
      p25 = w$quantiles["p25"],
      p50 = w$quantiles["p50"],
      p75 = w$quantiles["p75"],
      p95 = w$quantiles["p95"],
      iqr = w$iqr,
      stringsAsFactors = FALSE,
      row.names = NULL
    )
  })
  df <- do.call(rbind, rows)
  rownames(df) <- df$attribute
  df <- df[, names(df) != "attribute", drop = FALSE]
  df[order(-df[[by]]), , drop = FALSE]
}
