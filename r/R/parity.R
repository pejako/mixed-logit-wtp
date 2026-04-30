#' Compare R estimates against Python estimates
#'
#' Reads a Python-side results JSON (see `python/scripts/export_results.py`)
#' and produces a side-by-side comparison table.
#'
#' Cross-language differences should be small but non-zero — the SML estimator
#' has simulation noise from the Halton draws, the optimizer convergence
#' criteria differ slightly, and the numerical Hessian SEs depend on step
#' size. A tolerance of ~5% on the means and ~10% on the SDs is typical.
#'
#' @param r_result Output of `fit_mnl_r` or `fit_mxl_r`.
#' @param python_json_path Path to a JSON file produced by the Python side.
#' @return A data.frame with one row per parameter, comparing estimates.
#' @export
compare_with_python <- function(r_result, python_json_path) {
  if (!file.exists(python_json_path)) {
    stop(sprintf("Python results file not found: %s", python_json_path))
  }
  py <- jsonlite::fromJSON(python_json_path, simplifyVector = FALSE)

  # Use Python-aligned labels if available (MXL case), else raw R names
  r_labels <- if (!is.null(r_result$coef_names_python)) {
    r_result$coef_names_python
  } else {
    r_result$coef_names
  }

  rows <- list()
  for (i in seq_along(r_labels)) {
    name <- r_labels[i]
    r_est <- r_result$coefficients[i]
    r_se <- r_result$std_errors[i]
    if (name %in% names(py$coefficients)) {
      py_est <- py$coefficients[[name]]
      py_se <- py$std_errors[[name]]
      gap_pct <- if (abs(py_est) > 1e-6) {
        100 * (r_est - py_est) / abs(py_est)
      } else {
        NA
      }
      rows[[length(rows) + 1]] <- data.frame(
        parameter = name,
        R_est = r_est,
        Python_est = py_est,
        gap_pct = gap_pct,
        R_se = r_se,
        Python_se = py_se,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}
