# install_reference.R -- install the pinned R trisk.model for the parity check.
# Created with the support of AI tools.
#
#   Rscript tests/r/install_reference.R [lib]
#
# Installs Theia-Finance-Labs/trisk.model at commit 748ef51 (v2.7.0, main on
# 2026-08-04) plus jsonlite. Run on a box with R
# (>= 4.1) and internet access, ideally a private library so other R work on the
# box is untouched. Read-only use of the upstream repo: never push to it.

args <- commandArgs(trailingOnly = TRUE)
lib <- if (length(args) >= 1) args[[1]] else Sys.getenv("R_LIBS_USER")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
repos <- "https://cloud.r-project.org"
for (p in c("remotes", "jsonlite")) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, repos = repos)
}
remotes::install_github("Theia-Finance-Labs/trisk.model@748ef51", lib = lib,
                        upgrade = "never", dependencies = TRUE)
cat("trisk.model", as.character(utils::packageVersion("trisk.model", lib.loc = lib)), "in", lib, "\n")
