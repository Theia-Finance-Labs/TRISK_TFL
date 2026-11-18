# run_reference.R -- run one test case on R trisk.model 2.7.0 and write expected/.
# Created with the support of AI tools.
#
#   Rscript tests/r/run_reference.R tests/cases/<name>
#
# Reads <case>/case.json and <case>/inputs/,
# calls trisk.model::run_trisk_model() once per run with exactly the arguments
# in case.json, and writes <case>/expected/<run_id>_{npv,pd,trajectories}.csv
# plus expected/sessioninfo.txt. Refuses any trisk.model other than the version
# named in case.json, so a golden file always says which R produced it.

suppressPackageStartupMessages({
  library(jsonlite)
  library(trisk.model)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("usage: Rscript tests/r/run_reference.R <case folder>")
case_dir <- args[[1]]
case <- jsonlite::fromJSON(file.path(case_dir, "case.json"), simplifyVector = FALSE)

want <- case$reference$version
have <- as.character(utils::packageVersion("trisk.model"))
if (!identical(want, have)) {
  stop(sprintf("case wants trisk.model %s, installed is %s (see install_reference.R)", want, have))
}

inputs <- trisk.model::st_read_agnostic(file.path(case_dir, "inputs"))
out_dir <- file.path(case_dir, "expected")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_out <- function(df, name) {
  path <- file.path(out_dir, name)
  con <- file(path, open = "w", encoding = "UTF-8")
  writeLines(sprintf("# R reference output, trisk.model %s", have), con)
  utils::write.csv(df, con, row.names = FALSE, na = "")
  close(con)
}

assets <- inputs$assets_data
countries <- unlist(case$countries)
if (length(countries) > 0) assets <- assets[assets$country_iso2 %in% countries, ]

for (run in case$runs) {
  scen <- inputs$scenarios_data
  sectors <- unlist(run$sectors)
  if (length(sectors) > 0) scen <- scen[scen$sector %in% sectors, ]
  a <- run$args
  res <- trisk.model::run_trisk_model(
    assets_data = assets,
    scenarios_data = scen,
    financial_data = inputs$financial_data,
    carbon_data = inputs$carbon_data,
    baseline_scenario = a$baseline_scenario,
    target_scenario = a$target_scenario,
    scenario_geography = a$scenario_geography,
    carbon_price_model = a$carbon_price_model,
    risk_free_rate = a$risk_free_rate,
    discount_rate = a$discount_rate,
    growth_rate = a$growth_rate,
    div_netprofit_prop_coef = a$div_netprofit_prop_coef,
    shock_year = a$shock_year,
    market_passthrough_power = a$market_passthrough_power,
    market_passthrough_primary = a$market_passthrough_primary,
    run_id = run$run_id
  )
  write_out(res$npv_results, paste0(run$run_id, "_npv.csv"))
  write_out(res$pd_results, paste0(run$run_id, "_pd.csv"))
  write_out(res$company_trajectories, paste0(run$run_id, "_trajectories.csv"))
  cat(sprintf("%s: npv %d, pd %d, trajectories %d rows\n", run$run_id,
              nrow(res$npv_results), nrow(res$pd_results), nrow(res$company_trajectories)))
}

writeLines(capture.output(utils::sessionInfo()), file.path(out_dir, "sessioninfo.txt"))
