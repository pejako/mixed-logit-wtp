# Parity tests — these are the heart of the R/Python alignment claim.
#
# Each test:
#   1. skips if mlogit is not installed,
#   2. loads the canonical synthetic dataset (same data Python tests use),
#   3. fits the model in R via mlogit,
#   4. compares against the Python results JSON.
#
# Acceptable cross-language gap:
#   - Coefficient *values*: within 5% relative (or 2 R standard errors,
#     whichever is wider). MNL is deterministic so the gap should be tiny.
#   - For MXL, simulation noise from the Halton draws matters; we allow
#     wider tolerance on the SDs (10%).

# Helper to find the Python results JSON regardless of working directory.
find_python_results <- function(filename) {
  candidates <- c(
    file.path(getwd(), "data", filename),
    file.path(getwd(), "..", "..", "data", filename),
    file.path(getwd(), "..", "data", filename),
    file.path(getwd(), "r", "data", filename)
  )
  for (c in candidates) {
    if (file.exists(c)) return(normalizePath(c))
  }
  NULL
}


test_that("MNL coefficients match Python within 5%", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mnl <- fit_mnl_r(df)
  expect_true(r_mnl$converged)

  py_path <- find_python_results("python_mnl_results.json")
  skip_if(is.null(py_path), "Python MNL results JSON not found")

  cmp <- compare_with_python(r_mnl, py_path)
  expect_true(all(abs(cmp$gap_pct) < 5),
              info = paste("Cross-language gaps:",
                           paste(capture.output(print(cmp)), collapse = "\n")))
})

test_that("MXL means match Python within 5%, SDs within 10%", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mxl <- fit_mxl_r(df, n_draws = 200)
  expect_true(r_mxl$converged)

  py_path <- find_python_results("python_mxl_results.json")
  skip_if(is.null(py_path), "Python MXL results JSON not found")

  cmp <- compare_with_python(r_mxl, py_path)

  is_sd <- grepl("\\[sd\\]", cmp$parameter)
  mean_gaps <- abs(cmp$gap_pct[!is_sd])
  sd_gaps <- abs(cmp$gap_pct[is_sd])

  expect_true(all(mean_gaps < 5),
              info = paste("Mean gaps over 5%:", toString(round(mean_gaps, 2))))
  expect_true(all(sd_gaps < 10),
              info = paste("SD gaps over 10%:", toString(round(sd_gaps, 2))))
})
