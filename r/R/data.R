#' Load the canonical synthetic choice dataset
#'
#' Reads the long-format CSV that the Python side generates with
#' `python -m mixedlogit.export_csv`. Both languages read identical data
#' so cross-language coefficient comparisons are meaningful.
#'
#' @param path Optional path to the CSV. Defaults to the bundled
#'   `r/data/synthetic_choices.csv`.
#' @return A data.frame with columns `individual_id`, `situation_id`,
#'   `alt_id`, `chosen`, `price`, `quality`, `brand_known`.
#' @export
load_synthetic_data <- function(path = NULL) {
  if (is.null(path)) {
    # Try several candidate paths relative to common working directories.
    # testthat runs tests from the project root or the testthat/ directory
    # depending on how it's invoked, and `sys.frame(1)$ofile` is unreliable
    # under testthat (it can be NULL), so we search rather than relying on
    # source-file introspection.
    candidates <- c(
      file.path(getwd(), "data", "synthetic_choices.csv"),                  # from r/
      file.path(getwd(), "..", "..", "data", "synthetic_choices.csv"),       # from r/tests/testthat/
      file.path(getwd(), "..", "data", "synthetic_choices.csv"),             # from r/tests/
      file.path(getwd(), "r", "data", "synthetic_choices.csv")               # from project root
    )
    for (c in candidates) {
      if (file.exists(c)) {
        path <- normalizePath(c)
        break
      }
    }
    if (is.null(path)) {
      stop(
        "Could not locate synthetic_choices.csv. Generate it from the ",
        "Python side with `python -m mixedlogit.export_csv`, or pass an ",
        "explicit `path` argument. Searched from: ", getwd()
      )
    }
  }
  read.csv(path)
}


#' Load the ground-truth metadata
#'
#' Reads the JSON that the Python side writes alongside the CSV. Contains
#' the true population parameters (means, SDs, distribution types) used to
#' generate the data, so parity tests can validate that R recovers the
#' same truth Python does.
#'
#' @param path Optional path. Defaults to the bundled JSON.
#' @return A list mirroring the Python `df.attrs["true_params"]`.
#' @export
load_ground_truth <- function(path = NULL) {
  if (is.null(path)) {
    candidates <- c(
      file.path(getwd(), "data", "ground_truth.json"),
      file.path(getwd(), "..", "..", "data", "ground_truth.json"),
      file.path(getwd(), "..", "data", "ground_truth.json"),
      file.path(getwd(), "r", "data", "ground_truth.json")
    )
    for (c in candidates) {
      if (file.exists(c)) {
        path <- normalizePath(c)
        break
      }
    }
    if (is.null(path)) {
      stop(
        "Could not locate ground_truth.json. Run the Python export first."
      )
    }
  }
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}
