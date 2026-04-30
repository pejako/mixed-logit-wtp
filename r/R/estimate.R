#' Fit a Multinomial Logit model on the canonical dataset
#'
#' Wraps `mlogit::mlogit()` with the formula and data conventions used by
#' the Python side, and returns a list whose field names mirror the Python
#' `MNLResult` dataclass for cross-language comparison.
#'
#' @param df Long-format data.frame with columns `individual_id`,
#'   `situation_id`, `alt_id`, `chosen`, plus the attribute columns named
#'   in `attr_names`.
#' @param attr_names Character vector of attribute columns to use as
#'   alternative-specific regressors with generic coefficients.
#' @return A list with elements `coef_names`, `coefficients`, `std_errors`,
#'   `loglik`, `loglik_null`, `n_observations`, `mcfadden_r2`, `t_values`,
#'   plus the raw `mlogit_object`.
#' @export
fit_mnl_r <- function(df,
                      attr_names = c("price", "quality", "brand_known")) {
  if (!requireNamespace("mlogit", quietly = TRUE)) {
    stop("Package 'mlogit' is required. Install with install.packages('mlogit').")
  }
  if (!requireNamespace("dfidx", quietly = TRUE)) {
    stop("Package 'dfidx' is required.")
  }

  # mlogit's dfidx takes the long-format data with a nested index spec:
  # situation_id is nested within individual_id, and alt_id distinguishes
  # alternatives within each situation. The list-of-vectors form tells
  # dfidx the situation_id is the choice-occasion index nested within
  # individual_id (the panel index needed for panel = TRUE).
  d <- dfidx::dfidx(
    df,
    idx = list(c("situation_id", "individual_id"), "alt_id"),
    choice = "chosen"
  )

  # Build the formula: chosen ~ price + quality + brand_known | 0
  # The "| 0" suppresses alternative-specific intercepts (we have none in
  # the DGP), so coefficients are alternative-generic.
  rhs <- paste(attr_names, collapse = " + ")
  fml <- stats::as.formula(paste("chosen ~", rhs, "| 0"))

  fit <- mlogit::mlogit(fml, data = d)
  s <- summary(fit)

  coefs <- stats::coef(fit)
  ses <- s$CoefTable[, "Std. Error"]
  tvals <- s$CoefTable[, "z-value"]

  ll <- as.numeric(stats::logLik(fit))
  # Null log-likelihood: equal shares
  n_situations <- length(unique(df$situation_id))
  n_alts <- length(unique(df$alt_id))
  ll_null <- -n_situations * log(n_alts)

  # Convergence check: mlogit doesn't expose a single boolean flag in a
  # stable location across versions. The robust signal is that we have
  # finite coefficients, finite standard errors, and a finite log-likelihood.
  converged <- all(is.finite(coefs)) &&
               all(is.finite(ses)) &&
               is.finite(ll)

  list(
    coef_names = names(coefs),
    coefficients = unname(coefs),
    std_errors = unname(ses),
    t_values = unname(tvals),
    loglik = ll,
    loglik_null = ll_null,
    mcfadden_r2 = 1 - ll / ll_null,
    n_observations = n_situations,
    converged = converged,
    mlogit_object = fit
  )
}


#' Fit a Mixed Logit model on the canonical dataset
#'
#' Wraps `mlogit::mlogit()` with `rpar` arguments specifying the random
#' coefficient distributions. Defaults match the Python `default_config`:
#' price and quality are normal, brand_known is fixed.
#'
#' Note: `mlogit` uses the convention that `rpar = c(price = "n")` declares
#' price as having a Normal random coefficient. The estimated parameters
#' are the mean and the standard deviation of that distribution.
#'
#' @param df Long-format data.frame.
#' @param random_attrs Named character vector. Names are attribute columns;
#'   values are the `mlogit` distribution code: "n" = normal, "ln" = lognormal,
#'   "u" = uniform, "t" = triangular. Default is `c(price = "n", quality = "n")`.
#' @param fixed_attrs Character vector of attribute columns that are fixed
#'   (homogeneous) coefficients. Default is `"brand_known"`.
#' @param n_draws Number of Halton draws per individual (passed as `R = ...`).
#' @return A list with elements similar to `fit_mnl_r`, plus the SD entries
#'   for random coefficients.
#' @export
fit_mxl_r <- function(df,
                      random_attrs = c(price = "n", quality = "n"),
                      fixed_attrs = "brand_known",
                      n_draws = 200) {
  if (!requireNamespace("mlogit", quietly = TRUE)) {
    stop("Package 'mlogit' is required.")
  }
  if (!requireNamespace("dfidx", quietly = TRUE)) {
    stop("Package 'dfidx' is required.")
  }

  # Nested index: situation_id within individual_id, alternatives within situations.
  # The individual index is REQUIRED when fitting with panel = TRUE, since
  # mlogit needs to know which observations to group together when
  # integrating over the random coefficient distribution.
  d <- dfidx::dfidx(
    df,
    idx = list(c("situation_id", "individual_id"), "alt_id"),
    choice = "chosen"
  )

  all_attrs <- c(names(random_attrs), fixed_attrs)
  rhs <- paste(all_attrs, collapse = " + ")
  fml <- stats::as.formula(paste("chosen ~", rhs, "| 0"))

  fit <- mlogit::mlogit(
    fml,
    data = d,
    rpar = random_attrs,
    R = n_draws,
    halton = NA,           # use Halton draws (NA = default Halton config)
    panel = TRUE,           # respect the within-individual panel structure
    print.level = 0
  )
  s <- summary(fit)

  coefs <- stats::coef(fit)
  ses <- s$CoefTable[, "Std. Error"]
  tvals <- s$CoefTable[, "z-value"]

  ll <- as.numeric(stats::logLik(fit))
  n_situations <- length(unique(df$situation_id))
  n_alts <- length(unique(df$alt_id))
  ll_null <- -n_situations * log(n_alts)

  # mlogit uses parameter names like "price" (mean) and "sd.price" (sd) for
  # a normal random coefficient. Build a python-aligned label vector for
  # easier cross-language comparison.
  python_labels <- vapply(
    names(coefs),
    function(nm) {
      if (startsWith(nm, "sd.")) {
        attr_name <- substring(nm, 4)
        sprintf("%s [sd]", attr_name)
      } else if (nm %in% names(random_attrs)) {
        sprintf("%s [mean]", nm)
      } else {
        nm  # fixed coef
      }
    },
    character(1)
  )

  # Same robust convergence check as MNL: finite coefs/SEs/log-likelihood.
  converged <- all(is.finite(coefs)) &&
               all(is.finite(ses)) &&
               is.finite(ll)

  list(
    coef_names = names(coefs),
    coef_names_python = python_labels,
    coefficients = unname(coefs),
    std_errors = unname(ses),
    t_values = unname(tvals),
    loglik = ll,
    loglik_null = ll_null,
    mcfadden_r2 = 1 - ll / ll_null,
    n_observations = n_situations,
    n_individuals = length(unique(df$individual_id)),
    n_draws = n_draws,
    random_attrs = random_attrs,
    fixed_attrs = fixed_attrs,
    converged = converged,
    mlogit_object = fit
  )
}
