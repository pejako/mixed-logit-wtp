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
    # Find the data dir relative to this source file
    here <- tryCatch(
      dirname(sys.frame(1)$ofile),
      error = function(e) getwd()
    )
    candidates <- c(
      file.path(here, "..", "data", "synthetic_choices.csv"),
      file.path(getwd(), "data", "synthetic_choices.csv"),
      file.path(getwd(), "r", "data", "synthetic_choices.csv")
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
        "explicit `path` argument."
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
    here <- tryCatch(
      dirname(sys.frame(1)$ofile),
      error = function(e) getwd()
    )
    candidates <- c(
      file.path(here, "..", "data", "ground_truth.json"),
      file.path(getwd(), "data", "ground_truth.json"),
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
