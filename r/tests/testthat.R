library(testthat)

# Source all R files under R/
r_dir <- file.path("..", "..", "R")
for (f in list.files(r_dir, pattern = "\\.R$", full.names = TRUE)) {
  source(f)
}

test_check("mixedlogitr")
