test_that("MNL elasticities satisfy IIA (constant column off-diagonals)", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mnl <- fit_mnl_r(df)

  design <- rbind(
    c(2.0,  1.0, 1.0),
    c(1.0,  0.0, 1.0),
    c(0.5, -1.0, 0.0)
  )
  e <- mnl_elasticities_r(r_mnl, design,
                          alt_labels = c("Premium", "Mid", "Budget"))

  # IIA: all off-diagonals in each column are identical
  for (j in seq_len(ncol(e$matrix))) {
    col <- e$matrix[, j]
    off <- col[-j]
    expect_equal(diff(range(off)), 0, tolerance = 1e-10,
                 info = sprintf("Column %d off-diagonals not constant", j))
  }
})

test_that("Diagonal own-elasticities are negative for normal goods", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mnl <- fit_mnl_r(df)
  design <- rbind(
    c(2.0,  1.0, 1.0),
    c(1.0,  0.0, 1.0),
    c(0.5, -1.0, 0.0)
  )
  e <- mnl_elasticities_r(r_mnl, design)
  for (i in seq_len(nrow(e$matrix))) {
    expect_lt(e$matrix[i, i], 0)
  }
})

test_that("MXL elasticities violate IIA on differentiated design", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mxl <- fit_mxl_r(df, n_draws = 100)

  design <- rbind(
    c(2.0,  1.0, 1.0),
    c(1.0,  0.0, 1.0),
    c(0.5, -1.0, 0.0)
  )
  e <- mxl_elasticities_r(r_mxl, design, n_draws = 1000)

  # At least one column should have non-trivial spread
  found_violation <- FALSE
  for (j in seq_len(ncol(e$matrix))) {
    col <- e$matrix[, j]
    off <- col[-j]
    if (diff(range(off)) > 0.05) {
      found_violation <- TRUE
      break
    }
  }
  expect_true(found_violation,
              info = "MXL did not produce row-varying cross-elasticities")
})

test_that("compute_wtp_r produces sensible median for default config", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mxl <- fit_mxl_r(df, n_draws = 100)

  # The default config triggers the Cauchy warning; suppress it
  wtps <- suppressWarnings(compute_wtp_r(r_mxl, n_draws = 5000))

  # True median quality WTP: -0.8 / -1.2 = 0.667
  expect_lt(abs(wtps$quality$median - 0.667), 0.10)
  # True median brand WTP: -0.6 / -1.2 = 0.500
  expect_lt(abs(wtps$brand_known$median - 0.500), 0.15)
})

test_that("feature_ranking_r sorts descending by chosen statistic", {
  skip_if_not_installed("mlogit")
  skip_if_not_installed("dfidx")

  df <- load_synthetic_data()
  r_mxl <- fit_mxl_r(df, n_draws = 100)
  wtps <- suppressWarnings(compute_wtp_r(r_mxl, n_draws = 2000))
  ranking <- feature_ranking_r(wtps, by = "median")

  # Should be sorted descending by median
  meds <- ranking$median
  expect_true(all(diff(meds) <= 0))
  # Top should be quality (true median ~0.67) above brand (~0.50)
  expect_equal(rownames(ranking)[1], "quality")
})
