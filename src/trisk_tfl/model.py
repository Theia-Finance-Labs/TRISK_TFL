"""model.py — TRISK in Python: Theia Finance Labs' Python implementation of trisk.model 2.7.0.

Created with the support of AI tools.

`run_trisk_model()` takes the same four data frames and the same arguments, with the same
defaults, as R trisk.model 2.7.0 (github.com/Theia-Finance-Labs/trisk.model, commit 748ef51),
and returns the same three tables: npv_results, pd_results and company_trajectories.
Needs only pandas (2.2 or 3.x), numpy and scipy.

Licence: MIT (see LICENSE). This module translates trisk.model into Python. Theia Finance Labs,
the copyright holder of trisk.model, publishes the translation under the MIT licence.

PARITY: .github/workflows/parity.yml runs R trisk.model 2.7.0 and this module on the cases in
tests/cases/ and compares every output table.

Each stage names the R file it ports. Deliberate deviations from R, kept small:
  D1  set_baseline_trajectory: R's fill-down is ungrouped; here it is grouped by
      asset. Identical unless an asset has no production value at all, where R
      would carry the previous asset's value into it.
  D2  The dividend discount model numbers years by sorted year within an asset; R
      numbers them in row order, which is year order in every path we traced.
"""
from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
from scipy.stats import norm

ENGINE_VERSION = "0.1.0"     # this port; bump on any change to the maths
R_VERSION = "2.7.0"
R_COMMIT = "748ef51"
MAX_POSSIBLE_YEAR = 2050
HOURS_TO_YEAR = 24 * 365
KEYS = ["asset_id", "company_id", "sector", "technology"]

DEFAULTS = dict(
    baseline_scenario="", target_scenario="", scenario_geography="Global",
    carbon_price_model="no_carbon_tax", risk_free_rate=0.045, discount_rate=0.09,
    growth_rate=0.03, div_netprofit_prop_coef=1.0, shock_year=2030,
    market_passthrough_power=0.6, market_passthrough_primary=0.9,
)


# --- helpers -------------------------------------------------------------------------
def _gid(df, by):
    return df.groupby(by, sort=False, dropna=False).ngroup()


def _first(df, by, col, skipna=False):
    """dplyr::first() per group in current row order (NA kept unless skipna)."""
    s = df[col]
    gid = _gid(df, by)
    if skipna:
        ok = s.notna()
        sub = pd.Series(s[ok].values, index=gid[ok].values)
        m = sub[~sub.index.duplicated(keep="first")]
    else:
        pos = ~gid.duplicated(keep="first")
        m = pd.Series(s[pos].values, index=gid[pos].values)
    return gid.map(m)


def _last(df, by, col, skipna=False):
    s = df[col]
    gid = _gid(df, by)
    if skipna:
        ok = s.notna()
        sub = pd.Series(s[ok].values, index=gid[ok].values)
        m = sub[~sub.index.duplicated(keep="last")]
    else:
        pos = ~gid.duplicated(keep="last")
        m = pd.Series(s[pos].values, index=gid[pos].values)
    return gid.map(m)


def _r_first(s):
    """dplyr::first in an agg: the first row, NA included (pandas 'first' skips NA)."""
    return s.iloc[0]


def _r_last(s):
    return s.iloc[-1]


def _gmax_where(df, by, col, mask, empty=-np.inf):
    """max(col[mask]) per group, -Inf (R's max of empty) where none."""
    gid = _gid(df, by)
    m = df[col].where(mask).groupby(gid).max()
    return gid.map(m).fillna(empty)


def _r_first_true_year(df, by, cond):
    """R: ifelse(any(cond), min(year[cond]), NA) with three-valued `cond` (NaN = NA).
    min() over a subset holding NA is NA, so a group with any NA in cond gives NA."""
    gid = _gid(df, by)
    has_na = cond.isna().groupby(gid).any()
    y = df["year"].where(cond.fillna(False).astype(bool))
    first = y.groupby(gid).min()
    first[has_na] = np.nan
    return gid.map(first)


# --- R: run_trisk.R ------------------------------------------------------------------
def retype_data(assets_data, scenarios_data, financial_data, carbon_data):
    a = assets_data[["asset_id", "asset_name", "company_id", "company_name", "country_iso2",
                     "production_year", "sector", "technology", "capacity",
                     "capacity_factor", "emission_factor"]].copy()
    for c in ("asset_id", "asset_name", "company_id", "company_name", "country_iso2",
              "sector", "technology"):
        a[c] = a[c].map(lambda v: None if pd.isna(v) else str(v))
    for c in ("production_year", "capacity", "capacity_factor", "emission_factor"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    s = scenarios_data.copy()
    for c in ("scenario_geography", "scenario", "scenario_type", "sector", "technology",
              "country_iso2_list"):
        s[c] = s[c].astype(object).where(s[c].notna(), None)
    for c in ("scenario_year", "scenario_price", "scenario_capacity_factor",
              "scenario_pathway"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s["country_iso2_list"] = s["country_iso2_list"].replace({"": None, "nan": None})
    f = financial_data[["company_id", "pd", "net_profit_margin", "debt_equity_ratio",
                        "volatility"]].copy()
    f["company_id"] = f["company_id"].astype(str)
    for c in ("pd", "net_profit_margin", "debt_equity_ratio", "volatility"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    k = carbon_data[["year", "model", "scenario", "scenario_geography", "variable", "unit",
                     "carbon_tax"]].copy()
    k["year"] = pd.to_numeric(k["year"], errors="coerce")
    k["carbon_tax"] = pd.to_numeric(k["carbon_tax"], errors="coerce")
    return a, s, f, k


def run_trisk_model(assets_data, scenarios_data, financial_data, carbon_data,
                    baseline_scenario="", target_scenario="", scenario_geography="Global",
                    carbon_price_model="no_carbon_tax", risk_free_rate=0.045,
                    discount_rate=0.09, growth_rate=0.03, div_netprofit_prop_coef=1,
                    shock_year=2030, market_passthrough_power=0.6,
                    market_passthrough_primary=0.9, run_id=None):
    """Port of trisk.model::run_trisk_model() 2.7.0. Returns dict of three DataFrames."""
    if run_id is None:
        run_id = str(uuid.uuid4())
    a, s, f, k = retype_data(assets_data, scenarios_data, financial_data, carbon_data)
    assets = process_assets_data(a, f)
    scenarios = process_scenarios_data(s, baseline_scenario, target_scenario,
                                       scenario_geography)
    assets_scenarios = merge_assets_and_scenarios_data(assets, scenarios)
    trisk_model_input, proximity = process_trisk_input(assets_scenarios)

    start_year = trisk_model_input["year"].min()
    end_analysis = trisk_model_input["year"].max()

    out = extend_assets_trajectories(trisk_model_input, start_year, shock_year)
    out = apply_zero_trajectory_logic(out)
    carbon = process_carbon_data(k, start_year, end_analysis, carbon_price_model)
    out = out.merge(proximity, on=KEYS, how="inner")
    profits = calculate_net_profits(out, carbon, shock_year, market_passthrough_power,
                                    market_passthrough_primary)
    annual = calculate_annual_profits(profits, start_year, end_analysis, discount_rate,
                                      growth_rate)
    npv = calculate_asset_value_at_risk(annual, shock_year, start_year,
                                        div_netprofit_prop_coef)
    pdc = calculate_pd_change_overall(annual, start_year, end_analysis, risk_free_rate)
    return {
        "npv_results": prepare_npv_results(npv, trisk_model_input, run_id),
        "pd_results": prepare_pd_results(pdc, run_id),
        "company_trajectories": prepare_company_trajectories(annual, run_id),
    }


# --- R: proc_input_data.R + calc_fair_share_perc.R ----------------------------------
def process_assets_data(assets_data, financial_data):
    d = assets_data.merge(financial_data, on="company_id", how="inner")
    d["production_plan_company_technology"] = d["capacity"] * d["capacity_factor"]
    return d.drop(columns=["capacity", "capacity_factor"])


def process_scenarios_data(scenarios_data, baseline_scenario, target_scenario,
                           scenario_geography):
    geos = [scenario_geography] if isinstance(scenario_geography, str) else list(scenario_geography)
    d = scenarios_data[scenarios_data["scenario"].isin([baseline_scenario, target_scenario])]
    d = d[d["scenario_geography"].isin(geos)]
    d = d.sort_values("scenario_year", kind="stable").reset_index(drop=True)
    tech = ["scenario", "sector", "scenario_geography", "technology"]
    # add_technology_fair_share_ratio: grouped by tech, arranged by year
    d = d.sort_values(tech + ["scenario_year"], kind="stable").reset_index(drop=True)
    first_p = _first(d, tech, "scenario_pathway")
    with np.errstate(divide="ignore", invalid="ignore"):
        d["tmsr"] = (d["scenario_pathway"] - first_p) / first_p
    # add_market_fair_share_percentage: R arranges by (scenario, sector, geo, year)
    yr = ["scenario", "sector", "scenario_geography", "scenario_year"]
    d = d.sort_values(yr, kind="stable").reset_index(drop=True)
    d["_sector_total"] = d.groupby(yr, dropna=False)["scenario_pathway"].transform("sum")
    first_p = _first(d, tech, "scenario_pathway")
    first_tot = _first(d, tech, "_sector_total")
    with np.errstate(divide="ignore", invalid="ignore"):
        d["smsp"] = (d["scenario_pathway"] - first_p) / first_tot
    d = d.drop(columns="_sector_total")
    # calculate_fair_share_perc; is.na() catches NaN, not Inf
    fs = np.where(d["technology_type"] == "carbontech", d["tmsr"],
                  np.where(d["technology_type"] == "greentech", d["smsp"], np.nan))
    d["fair_share_perc"] = pd.Series(fs, index=d.index).fillna(0.0)
    return d


# --- R: merge_trisk_inputs.R ---------------------------------------------------------
def merge_assets_and_scenarios_data(assets_data, scenarios_data):
    techs = scenarios_data["technology"].drop_duplicates()
    a = assets_data[assets_data["technology"].isin(techs)]
    countries = scenarios_data["country_iso2_list"].drop_duplicates().tolist()
    if len(countries) > 1:
        raise ValueError("scenario geography carries more than one country_iso2_list "
                         f"({countries}); R errors here")
    if countries and countries[0] is not None and not pd.isna(countries[0]):
        a = a[a["country_iso2"].isin(str(countries[0]).split(","))]
    if a.empty:
        raise ValueError("no asset matches the scenario technologies and countries")
    start = scenarios_data["scenario_year"].min()
    end = min(scenarios_data["scenario_year"].max(), MAX_POSSIBLE_YEAR)
    full = extend_to_full_analysis_timeframe(a, start, end)
    d = full.merge(scenarios_data, left_on=["sector", "technology", "production_year"],
                   right_on=["sector", "technology", "scenario_year"], how="inner")
    return d.drop(columns="scenario_year").rename(columns={"production_year": "year"})


def extend_to_full_analysis_timeframe(data, start_analysis, end_analysis):
    lo, hi = data["production_year"].min(), data["production_year"].max()
    if not (lo >= start_analysis and hi <= end_analysis):
        raise ValueError(f"asset production years {lo:.0f}..{hi:.0f} fall outside the "
                         f"scenario window {start_analysis:.0f}..{end_analysis:.0f} "
                         "(R: stopifnot in merge_trisk_inputs.R)")
    years = pd.DataFrame({"production_year": np.arange(int(start_analysis),
                                                       int(end_analysis) + 1, dtype=float)})
    keys = data[KEYS].drop_duplicates()
    grid = keys.merge(years, how="cross")
    d = grid.merge(data, on=KEYS + ["production_year"], how="left")
    d = d.sort_values(KEYS + ["production_year"], kind="stable").reset_index(drop=True)
    down = ["asset_name", "company_name", "country_iso2", "emission_factor", "pd",
            "net_profit_margin", "debt_equity_ratio", "volatility"]
    g = d.groupby(KEYS, sort=False, dropna=False)
    d[down] = g[down].ffill()
    up = down + ["production_plan_company_technology"]
    d[up] = d.groupby(KEYS, sort=False, dropna=False)[up].bfill()
    return d


# --- R: proc_trisk_input.R + calculate_proximity_to_target ---------------------------
def process_trisk_input(assets_scenarios):
    d, proximity = create_base_production_trajectories(assets_scenarios)
    d = lag_scenario_productions(d)
    piv = pivot_to_baseline_target_columns(d)
    attrs = ["asset_name", "company_name", "asset_id", "company_id", "sector", "technology",
             "technology_type", "country_iso2", "debt_equity_ratio", "net_profit_margin",
             "pd", "scenario_geography", "year", "emission_factor", "volatility"]
    tmi = assets_scenarios[attrs].drop_duplicates()
    tmi = tmi.merge(piv, on=KEYS + ["year"], how="inner")
    return tmi.reset_index(drop=True), proximity


def create_base_production_trajectories(data):
    d = data.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    d["initial_technology_production"] = _first(d, KEYS, "production_plan_company_technology",
                                                skipna=True)
    d["final_technology_production"] = _last(d, KEYS, "production_plan_company_technology",
                                             skipna=True)
    d["production_scenario"] = d["initial_technology_production"] * (1 + d["fair_share_perc"])
    d.loc[d["production_scenario"] < 0, "production_scenario"] = 0.0
    proximity = calculate_proximity_to_target(d)
    power = d["sector"] == "Power"
    factor = np.where(power, d["scenario_capacity_factor"] * HOURS_TO_YEAR,
                      d["scenario_capacity_factor"])
    d["production_plan_company_technology"] = d["production_plan_company_technology"] * factor
    d["production_scenario"] = d["production_scenario"] * factor
    return d, proximity


def calculate_proximity_to_target(data):
    d = data.copy()
    d["_last_non_na_year"] = _gmax_where(d, KEYS, "year",
                                         d["production_plan_company_technology"].notna())
    d = d[(d["year"] <= d["_last_non_na_year"]) & (d["scenario_type"] == "target")]
    d = d.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    init = _first(d, KEYS, "production_plan_company_technology", skipna=True)
    d["required_change"] = d["production_scenario"] - init
    d["realised_change"] = d["production_plan_company_technology"] - init
    agg = (d.groupby(KEYS, dropna=False)[["required_change", "realised_change"]]
           .sum(min_count=0).reset_index())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = agg["realised_change"] / agg["required_change"]
    agg["proximity_to_target"] = np.where(ratio < 0, 0.0, np.where(ratio > 1, 1.0, ratio))
    return agg[KEYS + ["proximity_to_target"]]


def lag_scenario_productions(data):
    order = ["scenario_type", "company_id", "asset_id", "sector", "technology", "year"]
    d = data.sort_values(order, kind="stable").reset_index(drop=True)
    g = d.groupby("scenario_type", sort=False, dropna=False)
    change = d["production_scenario"] - g["production_scenario"].shift(1)
    min_year = g["year"].transform("min")
    mask = d["production_plan_company_technology"].notna() | (d["year"] == min_year)
    d["production_change_scenario"] = change.where(~mask).fillna(0.0)
    return d


def pivot_to_baseline_target_columns(data):
    idx = KEYS + ["year"]
    vals = ["production_scenario", "scenario_price", "production_plan_company_technology",
            "production_change_scenario"]
    b = data[data["scenario_type"] == "baseline"][idx + vals]
    b = b.rename(columns={v: f"{v}_baseline" for v in vals}).rename(
        columns={"production_plan_company_technology_baseline":
                 "production_plan_company_technology"})
    t = data[data["scenario_type"] == "target"][idx + vals]
    t = t.rename(columns={v: f"{v}_target" for v in vals}).drop(
        columns="production_plan_company_technology_target")
    for part, name in ((b, "baseline"), (t, "target")):
        if part.duplicated(idx).any():
            raise ValueError(f"duplicate {name} scenario rows per asset-year; R would build "
                             "list columns here")
    return b.merge(t, on=idx, how="inner")


# --- R: calc_trisk_trajectory.R ------------------------------------------------------
def extend_assets_trajectories(trisk_model_input, start_year, shock_year):
    d = trisk_model_input.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    d = set_baseline_trajectory(d)
    d = set_trisk_trajectory(d, start_year, shock_year)
    d = apply_scenario_prices(d, shock_year)
    cols = ["year", "asset_id", "asset_name", "company_id", "company_name", "sector",
            "technology", "production_plan_company_technology",
            "production_scenario_baseline", "production_scenario_target",
            "production_change_scenario_baseline", "production_change_scenario_target",
            "production_asset_baseline", "late_sudden", "overshoot_direction",
            "scenario_price_baseline", "scenario_price_target", "late_sudden_price"]
    d = d[cols]
    attrs = trisk_model_input[KEYS + ["technology_type", "scenario_geography", "year",
                                      "emission_factor", "debt_equity_ratio",
                                      "net_profit_margin", "pd", "volatility",
                                      "country_iso2"]].drop_duplicates()
    return d.merge(attrs, on=KEYS + ["year"], how="left")


def set_baseline_trajectory(data):
    d = data.copy()
    g = d.groupby(KEYS, sort=False, dropna=False)
    base = g["production_plan_company_technology"].ffill()              # D1: grouped
    d["production_asset_baseline"] = base + g["production_change_scenario_baseline"].cumsum()
    d.loc[d["production_asset_baseline"] < 0, "production_asset_baseline"] = 0.0
    return d


def set_trisk_trajectory(data, start_year, shock_year):
    ls = calc_late_sudden_traj(data, shock_year)
    d = data.merge(ls, on=KEYS + ["year"], how="left")
    by = ["company_id", "asset_id", "technology"]
    d["_lfy"] = _gmax_where(d, by, "year", d["production_plan_company_technology"].notna())
    after = d["year"] > d["_lfy"]
    cond = pd.Series(np.where(after, np.where(d["late_sudden"].isna(), np.nan,
                                              d["late_sudden"] == 0), 0.0), index=d.index)
    first_zero = _r_first_true_year(d, by, cond)
    zero = first_zero.notna() & (d["year"] >= first_zero) & after
    d.loc[zero, "late_sudden"] = 0.0
    return d.drop(columns="_lfy")


def calc_late_sudden_traj(data, year_of_shock):
    cols = KEYS + ["year", "production_plan_company_technology", "production_scenario_target",
                   "production_change_scenario_target", "production_change_scenario_baseline"]
    d = data[cols].sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    g = d.groupby(KEYS, sort=False, dropna=False)
    first_t = _first(d, KEYS, "production_scenario_target")
    last_t = _last(d, KEYS, "production_scenario_target")
    diff = first_t - last_t
    d["overshoot_direction"] = np.where(diff.isna(), None,
                                        np.where(diff > 0, "Decreasing", "Increasing"))
    d["scenario_change_baseline_cumsum"] = g["production_change_scenario_baseline"].cumsum()
    d["scenario_change_cumsum"] = g["production_change_scenario_target"].cumsum()
    d["production_plan_company_technology_filled"] = g["production_plan_company_technology"].ffill()

    d["_last_non_na_year"] = _gmax_where(d, KEYS, "year",
                                         d["production_plan_company_technology"].notna())
    d["_min_year"] = g["year"].transform("min")
    w = d[(d["year"] > d["_min_year"]) & (d["year"] <= d["_last_non_na_year"])]
    if len(w):
        agg = w.groupby(KEYS, sort=False, dropna=False).agg(
            prod_to_follow=("production_scenario_target", "sum"),
            real_prod=("production_plan_company_technology", "sum"),
            overshoot_direction=("overshoot_direction", _r_first)).reset_index()
        dec = agg["overshoot_direction"] == "Decreasing"
        agg["requires_overshoot_correction"] = (
            (dec & (agg["prod_to_follow"] < agg["real_prod"])) |
            (~dec & (agg["prod_to_follow"] > agg["real_prod"])))
    else:
        agg = pd.DataFrame(columns=KEYS + ["prod_to_follow", "real_prod",
                                           "overshoot_direction",
                                           "requires_overshoot_correction"])
    d = d.drop(columns=["_last_non_na_year", "_min_year", "overshoot_direction"])

    parts = []
    comp = agg[agg["requires_overshoot_correction"].astype(bool)][KEYS]
    if len(comp):
        c = d.merge(comp, on=KEYS, how="inner")
        c["late_sudden"] = np.where(
            c["year"] <= year_of_shock,
            c["production_plan_company_technology_filled"] + c["scenario_change_baseline_cumsum"],
            0.0)
        pre = c[c["year"] <= year_of_shock - 1].groupby(KEYS, sort=False, dropna=False).agg(
            late_sudden_pre_shock_val=("late_sudden", _r_last),
            late_sudden_pre_shock_tot=("late_sudden", "sum")).reset_index()
        tot = c.groupby(KEYS, sort=False, dropna=False).agg(
            production_scenario_target_total_sum=("production_scenario_target", "sum"),
            _last_year=("year", _r_last)).reset_index()
        tot["n_shocked_years"] = tot["_last_year"] - year_of_shock + 1
        x = pre.merge(tot, on=KEYS, how="left")
        n = x["n_shocked_years"]
        x["x"] = ((x["production_scenario_target_total_sum"] - x["late_sudden_pre_shock_tot"]
                   - n * x["late_sudden_pre_shock_val"]) / (-(n * (n + 1) / 2)))
        late = c[c["year"] >= year_of_shock].merge(pre, on=KEYS, how="left").merge(
            x[KEYS + ["x"]], on=KEYS, how="left")
        yd = (late["year"] - year_of_shock + 1).clip(lower=0)
        late["late_sudden"] = late["late_sudden_pre_shock_val"] - yd * late["x"]
        parts += [c[c["year"] < year_of_shock][KEYS + ["year", "late_sudden"]],
                  late[KEYS + ["year", "late_sudden"]]]
    keep = agg[~agg["requires_overshoot_correction"].astype(bool)][KEYS]
    if len(keep):
        n = d.merge(keep, on=KEYS, how="inner")
        n["late_sudden"] = n["production_plan_company_technology_filled"] + n["scenario_change_cumsum"]
        parts.append(n[KEYS + ["year", "late_sudden"]])
    out = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=KEYS + ["year", "late_sudden"]))
    out = out.merge(agg[KEYS + ["overshoot_direction"]].drop_duplicates(), on=KEYS, how="left")
    out["late_sudden"] = pd.to_numeric(out["late_sudden"]).clip(lower=0)
    return out


def apply_zero_trajectory_logic(data):
    d = data.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    for col in ("production_asset_baseline", "late_sudden", "production_scenario_baseline",
                "production_scenario_target"):
        cond = pd.Series(np.where(d[col].isna(), np.nan, d[col] == 0), index=d.index)
        first_zero = _r_first_true_year(d, KEYS, cond)
        d.loc[first_zero.notna() & (d["year"] >= first_zero), col] = 0.0
    return d


# --- R: calc_scenario_prices.R -------------------------------------------------------
def apply_scenario_prices(data, shock_year):
    d = data
    before = d[d["year"] <= shock_year][["asset_id", "company_id", "technology", "year",
                                         "scenario_price_baseline"]].rename(
        columns={"scenario_price_baseline": "late_sudden_price"})
    by = ["asset_id", "company_id", "technology"]
    a = d[d["year"] > shock_year - 1].sort_values(by + ["year"], kind="stable")
    rows = []
    if len(a):
        s = a.groupby(by, sort=False, dropna=False).agg(
            p0=("scenario_price_baseline", _r_first), p1=("scenario_price_target", _r_last),
            y0=("year", "min"), y1=("year", "max")).reset_index()
        for r in s.itertuples(index=False):
            if r.y1 <= r.y0:
                continue      # shock in the last year: R's seq() runs backwards; refused upstream
            years = np.arange(r.y0 + 1, r.y1 + 1)
            price = r.p0 + (r.p1 - r.p0) * (years - r.y0) / (r.y1 - r.y0)
            rows.append(pd.DataFrame({"asset_id": r.asset_id, "company_id": r.company_id,
                                      "technology": r.technology, "year": years,
                                      "late_sudden_price": price}))
    after = pd.concat(rows, ignore_index=True) if rows else before.iloc[0:0]
    prices = pd.concat([before, after], ignore_index=True)
    return d.merge(prices, on=by + ["year"], how="inner")


# --- R: read_inputs.R process_carbon_data + calc_net_profits.R -----------------------
def process_carbon_data(data, start_year, end_year, carbon_price_model):
    if data is None:
        return None
    d = data[(data["year"] >= start_year) & (data["year"] <= end_year)]
    d = d.drop(columns="scenario_geography")
    return d[d["scenario"] == carbon_price_model].reset_index(drop=True)


def calculate_net_profits(data, carbon_data, shock_year, market_passthrough_power,
                          market_passthrough_primary):
    baseline = data.copy()
    baseline["net_profits_baseline"] = (baseline["production_asset_baseline"] *
                                        baseline["scenario_price_baseline"] *
                                        baseline["net_profit_margin"])
    baseline = baseline[["company_id", "asset_id", "year", "sector", "technology",
                         "net_profits_baseline"]]
    green = _net_profits_greentech(data[data["technology_type"] == "greentech"])
    carbon = _net_profits_carbontech(data[data["technology_type"] == "carbontech"],
                                     shock_year, carbon_data, market_passthrough_power,
                                     market_passthrough_primary)
    d = pd.concat([green, carbon], ignore_index=True)
    return d.merge(baseline, on=["company_id", "asset_id", "year", "sector", "technology"],
                   how="outer")


def _ls_profit(d, carbon_cost):
    inc = d["overshoot_direction"] == "Increasing"
    comp = np.where(inc, d["late_sudden"] - d["production_asset_baseline"], 0.0)
    return (d["late_sudden"] * d["late_sudden_price"] * d["net_profit_margin"] - carbon_cost
            - comp * d["late_sudden_price"] * d["net_profit_margin"] * (1 - d["proximity_to_target"]))


def _net_profits_greentech(data):
    d = data[data["overshoot_direction"].isin(["Increasing", "Decreasing"])].copy()
    d["net_profits_ls"] = _ls_profit(d, 0.0)
    return d.drop(columns="proximity_to_target")


def _net_profits_carbontech(data, shock_year, carbon_data, pt_power, pt_primary):
    k = carbon_data.copy()
    inc50 = k["scenario"] == "increasing_carbon_tax_50"
    k["carbon_tax"] = np.where(inc50 & (k["year"] < shock_year), 0.0,
                      np.where(inc50 & (k["year"] == shock_year), 50.0,
                      np.where(inc50, k["carbon_tax"] * 1.04 ** (k["year"] - shock_year),
                               k["carbon_tax"])))
    d = data.merge(k, on="year", how="left")
    d = d[d["overshoot_direction"].isin(["Increasing", "Decreasing"])].copy()
    pt = np.where(d["sector"] == "Power", pt_power, pt_primary)
    tax = np.where(d["year"] < shock_year, 0.0, d["carbon_tax"])
    d["carbon_tax"] = tax
    cost = d["late_sudden"] * (1 - pt) * tax * d["emission_factor"]
    d["net_profits_ls"] = _ls_profit(d, cost)
    d.loc[d["net_profits_ls"] < 0, "net_profits_ls"] = 0.0
    return d.drop(columns="proximity_to_target")


# --- R: calc_annual_profts.R ---------------------------------------------------------
def calculate_annual_profits(data, start_year, end_year, discount_rate, growth_rate):
    d = data.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)
    t = d.groupby(KEYS, sort=False, dropna=False).cumcount()           # D2
    disc = (1 + discount_rate) ** t
    d["discounted_net_profit_baseline"] = d["net_profits_baseline"] / disc
    d["discounted_net_profit_ls"] = d["net_profits_ls"] / disc
    horizon = end_year - start_year
    tv = d[d["year"] == end_year].copy()
    tv["year"] = end_year + 1
    for c in ("net_profits_baseline", "net_profits_ls"):
        tv[c] = tv[c] * (1 + growth_rate)
    pull = (discount_rate - growth_rate) * (1 + discount_rate) ** horizon
    tv["discounted_net_profit_baseline"] = tv["net_profits_baseline"] / pull
    tv["discounted_net_profit_ls"] = tv["net_profits_ls"] / pull
    for c in ("production_plan_company_technology", "production_scenario_baseline",
              "production_scenario_target", "production_change_scenario_baseline",
              "production_change_scenario_target", "production_asset_baseline",
              "late_sudden", "scenario_price_baseline", "scenario_price_target",
              "late_sudden_price"):
        if c in tv:
            tv[c] = np.nan
    d = pd.concat([d, tv], ignore_index=True)
    return d.sort_values(KEYS + ["year"], kind="stable").reset_index(drop=True)


# --- R: calc_asset_value_at_risk.R (crispy = TRUE) -----------------------------------
def calculate_asset_value_at_risk(data, shock_year, start_year, div_netprofit_prop_coef=None):
    d = data[(data["year"] >= shock_year) & data["discounted_net_profit_ls"].notna() &
             data["discounted_net_profit_baseline"].notna()]
    by = ["company_id", "asset_id", "company_name", "asset_name", "sector", "technology"]
    out = d.groupby(by, dropna=False).agg(
        total_disc_npv_ls=("discounted_net_profit_ls", "sum"),
        total_disc_npv_baseline=("discounted_net_profit_baseline", "sum")).reset_index()
    out["duration_of_shock"] = shock_year - start_year
    out["year_of_shock"] = shock_year
    return out


# --- R: calc_pd_change_overall.R -----------------------------------------------------
def calculate_pd_change_overall(data, start_year, end_of_analysis, risk_free_interest_rate):
    d = data[data["year"] >= start_year]
    by = ["company_id", "company_name", "sector", "debt_equity_ratio", "volatility"]
    d = d.groupby(by, dropna=False).agg(
        equity_0_baseline=("discounted_net_profit_baseline", "sum"),
        equity_0_late_sudden=("discounted_net_profit_ls", "sum")).reset_index()
    d["debt"] = d["equity_0_baseline"] * d["debt_equity_ratio"]
    d["risk_free_rate"] = risk_free_interest_rate
    d = d.drop(columns="debt_equity_ratio")
    terms = pd.DataFrame({"term": np.arange(1, int(end_of_analysis - start_year) + 2)})
    d = d.merge(terms, how="cross")
    n0 = len(d)
    v0b = d["debt"] + d["equity_0_baseline"]
    v0s = d["debt"] + d["equity_0_late_sudden"]
    ok = ((d["risk_free_rate"] >= 0) & (d["debt"] > 0) & (v0b > 0) & (v0s > 0) &
          (d["volatility"] > 0) & (d["term"] > 0))
    d = d[ok].copy()
    if len(d) < n0 and len(d) == 0:
        raise ValueError("No data remain after removing rows that are not compatible with "
                         "merton model.")
    d["Survival_baseline"] = calc_survival_probability_merton(
        d["debt"], d["equity_0_baseline"] + d["debt"], d["volatility"], d["risk_free_rate"],
        d["term"])
    d["Survival_late_sudden"] = calc_survival_probability_merton(
        d["debt"], d["equity_0_late_sudden"] + d["debt"], d["volatility"],
        d["risk_free_rate"], d["term"])
    d["PD_baseline"] = 1 - d["Survival_baseline"]
    d["PD_late_sudden"] = 1 - d["Survival_late_sudden"]
    d["PD_change"] = d["PD_late_sudden"] - d["PD_baseline"]
    return d


def calc_survival_probability_merton(L, V0, sigma, r, t):
    L, V0, sigma, r, t = (np.asarray(x, dtype=float) for x in (L, V0, sigma, r, t))
    d1 = (np.log(V0 / L) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return 1 - norm.cdf(-d2)


# --- R: write_results.R prepare_* ----------------------------------------------------
def prepare_npv_results(company_technology_npv, trisk_model_input, run_id):
    countries = trisk_model_input[["asset_id", "country_iso2"]].drop_duplicates()
    d = company_technology_npv.merge(countries, on="asset_id", how="inner")
    d["run_id"] = run_id
    d = d.rename(columns={"total_disc_npv_baseline": "net_present_value_baseline",
                          "total_disc_npv_ls": "net_present_value_shock"})
    d["net_present_value_difference"] = d["net_present_value_shock"] - d["net_present_value_baseline"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["net_present_value_change"] = d["net_present_value_difference"] / d["net_present_value_baseline"]
    return d[["run_id", "company_id", "asset_id", "company_name", "asset_name", "sector",
              "technology", "country_iso2", "net_present_value_baseline",
              "net_present_value_shock", "net_present_value_difference",
              "net_present_value_change"]].reset_index(drop=True)


def prepare_pd_results(company_pd_changes_overall, run_id):
    d = company_pd_changes_overall.copy()
    d["run_id"] = run_id
    d = d.rename(columns={"PD_baseline": "pd_baseline", "PD_late_sudden": "pd_shock"})
    return d[["run_id", "company_id", "company_name", "sector", "term", "pd_baseline",
              "pd_shock"]].reset_index(drop=True)


def prepare_company_trajectories(company_trajectories, run_id):
    d = company_trajectories.copy()
    d["run_id"] = run_id
    d = d.rename(columns={
        "production_asset_baseline": "production_baseline_scenario",
        "production_scenario_target": "production_target_scenario",
        "late_sudden": "production_shock_scenario",
        "late_sudden_price": "price_shock_scenario",
        "net_profits_baseline": "net_profits_baseline_scenario",
        "net_profits_ls": "net_profits_shock_scenario",
        "discounted_net_profit_baseline": "discounted_net_profits_baseline_scenario",
        "discounted_net_profit_ls": "discounted_net_profits_shock_scenario"})
    return d[["run_id", "asset_id", "asset_name", "company_id", "company_name",
              "country_iso2", "sector", "technology", "year",
              "production_plan_company_technology", "production_baseline_scenario",
              "production_target_scenario", "production_shock_scenario", "pd",
              "net_profit_margin", "debt_equity_ratio", "volatility",
              "scenario_price_baseline", "price_shock_scenario",
              "net_profits_baseline_scenario", "net_profits_shock_scenario",
              "discounted_net_profits_baseline_scenario",
              "discounted_net_profits_shock_scenario"]].reset_index(drop=True)
