test_that("load_synthetic_data reads the canonical CSV", {
  df <- load_synthetic_data()
  expect_s3_class(df, "data.frame")
  expect_true(all(c("individual_id", "situation_id", "alt_id", "chosen",
                    "price", "quality", "brand_known") %in% names(df)))
  expect_gt(nrow(df), 0)
  # Each situation has exactly one chosen alternative
  chosen_per_sit <- aggregate(chosen ~ situation_id, data = df, sum)
  expect_true(all(chosen_per_sit$chosen == 1))
})

test_that("load_ground_truth returns expected structure", {
  truth <- load_ground_truth()
  expect_type(truth, "list")
  expect_true("attributes" %in% names(truth))
  attr_names <- vapply(truth$attributes, `[[`, character(1), "name")
  expect_setequal(attr_names, c("price", "quality", "brand_known"))
})

test_that("CSV row count matches DGP-implied count", {
  df <- load_synthetic_data()
  truth <- load_ground_truth()
  expected_n <- truth$n_individuals *
    truth$n_situations_per_individual *
    truth$n_alternatives
  expect_equal(nrow(df), expected_n)
})
