# testthat automatically sources every helper-*.R file in this directory
# before running tests. We use that to load the R/ source files when running
# `testthat::test_dir("tests/testthat")` directly (which is what the CI does
# and what most local workflows use).
#
# In contrast, `R CMD check` / `devtools::test()` load the package itself and
# don't need this helper. The helper is harmless in both cases.

local({
  # Find the R/ source directory regardless of where we're called from.
  candidates <- c(
    file.path("..", "..", "R"),                    # from tests/testthat
    file.path("R"),                                 # from r/
    file.path("r", "R"),                            # from project root
    normalizePath(file.path(getwd(), "..", "..", "R"), mustWork = FALSE)
  )
  r_dir <- NULL
  for (c in candidates) {
    if (dir.exists(c)) {
      r_dir <- c
      break
    }
  }
  if (is.null(r_dir)) {
    warning("helper.R: could not locate R/ source directory")
    return(invisible(NULL))
  }

  files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (f in files) {
    source(f, local = FALSE)  # source into global env so tests can see them
  }
})
