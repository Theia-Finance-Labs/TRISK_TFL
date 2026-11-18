"""portfolio.py — loan-book analysis on TRISK results: Python implementation of trisk.analysis 1.2.3.

Created with the support of AI tools.

Reference: github.com/Theia-Finance-Labs/trisk.analysis 1.2.3, commit 1f5a1e2. Each function
names the R function it mirrors and gives the same numbers.

    match_loans(loans, assets, matching, threshold)   -> (loans with company_id, match report)
    analyse(loans, npv_results, pd_results, shape)    -> (tech_detail, portfolio_results)
    integrate_pd / integrate_el                       -> adjusted PD / EL per row + aggregate
    apply_lgd(loans, method, addon, cap)              -> loans with the LGD to use

Licence: MIT (see LICENSE). Theia Finance Labs, the copyright holder of trisk.analysis,
publishes this translation under the MIT licence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRISK_ANALYSIS = {"repo": "Theia-Finance-Labs/trisk.analysis", "commit": "1f5a1e2",
                  "version": "1.2.3"}
REQUIRED = ["company_id", "company_name", "exposure_value_usd", "term", "loss_given_default"]
OPTIONAL = ["internal_pd", "internal_el", "country_iso2", "sector", "technology"]
ZSCORE_FLOOR, ZSCORE_CAP = 1e-4, 1 - 1e-4          # R/imports.R ZSCORE_*_DEFAULT
BPS = 1e4


class PortfolioError(ValueError):
    pass


def _sum_or_na(s):
    """R `if (all(is.na(x))) NA else sum(x, na.rm = TRUE)`."""
    return s.sum(min_count=1)


def _mean_or_na(s):
    return np.nan if s.isna().all() else s.mean()


# ---- input -----------------------------------------------------------------------------
def check_loans(loans: pd.DataFrame, shape: str) -> pd.DataFrame:
    """check_portfolio_simple (+ the row-level rules of run_trisk_on_simple_portfolio)."""
    miss = [c for c in REQUIRED if c not in loans.columns]
    if miss:
        raise PortfolioError(f"loan book is missing column(s) {miss} (needs {REQUIRED})")
    d = loans.copy()
    d["company_id"] = d["company_id"].astype("string").str.strip().replace({"": pd.NA})
    d["company_name"] = d["company_name"].astype(str).str.strip()
    for c in ("exposure_value_usd", "term", "loss_given_default"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = d[d["exposure_value_usd"].isna() | (d["exposure_value_usd"] < 0)]
    if len(bad):
        raise PortfolioError(f"loan book: {len(bad)} row(s) have no or a negative "
                             "exposure_value_usd")
    bad = d[d["loss_given_default"].isna() | (d["loss_given_default"] < 0) |
            (d["loss_given_default"] > 1)]
    if len(bad):
        raise PortfolioError(f"loan book: {len(bad)} row(s) have a loss_given_default outside 0..1")
    bad = d[d["term"].isna() | (d["term"] <= 0)]
    if len(bad):
        raise PortfolioError(f"loan book: {len(bad)} row(s) have no positive term (years)")
    if shape == "technology":
        for c in ("sector", "technology", "country_iso2"):
            if c not in d.columns or d[c].isna().any() or (d[c].astype(str).str.strip() == "").any():
                raise PortfolioError("a technology-level loan book needs sector, technology and "
                                     f"country_iso2 on every row ({c} is missing or blank)")
        d["country_iso2"] = d["country_iso2"].astype(str).str.strip().str.upper()
    else:
        # company-level: sector/technology come from the model (R drops stray columns)
        d = d.drop(columns=[c for c in ("sector", "technology") if c in d.columns])
    d[".portfolio_index"] = np.arange(1, len(d) + 1)
    return d


def apply_lgd(loans: pd.DataFrame, method: str, addon: float, cap: float) -> pd.DataFrame:
    """LGD vignette (bank_6) Method A: policy downturn add-on, capped."""
    if method != "downturn":
        return loans
    d = loans.copy()
    d["loss_given_default_ttc"] = d["loss_given_default"]
    d["loss_given_default"] = np.minimum(d["loss_given_default"] + addon, cap)
    return d


# ---- matching --------------------------------------------------------------------------
def _lcs_len(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def lcs_distance(a: str, b: str) -> float:
    """stringdist method "lcs": insertions + deletions = len(a) + len(b) - 2 * LCS, normalised
    by max(nchar) as in trisk.analysis::fuzzy_match_company_ids."""
    if not a and not b:
        return 0.0
    d = len(a) + len(b) - 2 * _lcs_len(a, b)
    return d / max(len(a), len(b))


def match_loans(loans: pd.DataFrame, assets: pd.DataFrame, matching: str = "id",
                threshold: float = 0.5) -> tuple:
    """Exact company_id match; optionally fuzzy_match_company_ids (lcs, normalised, ties ->
    first) for loans whose id is blank or not in the asset data. Returns the loans with the
    company_id to use and a match report (one row per loan; never dropped silently)."""
    known = assets[["company_id", "company_name"]].dropna(subset=["company_id"]).copy()
    known["company_id"] = known["company_id"].astype(str)
    known = known.drop_duplicates(["company_id", "company_name"])
    ids = set(known["company_id"])
    d = loans.copy()
    rep = []
    names = list(known.itertuples(index=False))
    for i, r in d.iterrows():
        cid = None if pd.isna(r["company_id"]) else str(r["company_id"])
        if cid is not None and cid in ids:
            rep.append((r[".portfolio_index"], cid, "id", "", 0.0))
            continue
        if matching == "fuzzy" and names and str(r["company_name"]):
            best = min(((lcs_distance(str(r["company_name"]), str(k.company_name)), k.company_id)
                        for k in names), key=lambda x: x[0])   # min keeps the first on ties
            if best[0] <= threshold:
                d.at[i, "company_id"] = best[1]
                rep.append((r[".portfolio_index"], best[1], "name (fuzzy)",
                            f"name matched to {best[1]} at distance {best[0]:.2f}; check it",
                            best[0]))
                continue
        why = ("company_id not in the company data" if cid is not None else "no company_id")
        if matching == "fuzzy":
            why += "; no company name within the fuzzy threshold"
        rep.append((r[".portfolio_index"], cid, "none", why, math.nan))
    report = pd.DataFrame(rep, columns=[".portfolio_index", "matched_company_id", "matched_by",
                                        "note", "name_distance"])
    report = d[[".portfolio_index", "company_id", "company_name", "exposure_value_usd", "term"]] \
        .rename(columns={"company_id": "company_id_used"}).merge(report, on=".portfolio_index")
    d.loc[report["matched_by"].eq("none").values, "company_id"] = pd.NA
    return d, report


# ---- TRISK outputs onto the loans --------------------------------------------------------
def aggregate_trisk_outputs_simple(npv: pd.DataFrame, pdr: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::aggregate_trisk_outputs_simple."""
    npv = npv.assign(company_id=npv["company_id"].astype(str))
    pdr = pdr.assign(company_id=pdr["company_id"].astype(str))
    npv_agg = (npv.groupby(["run_id", "company_id", "sector", "technology"], dropna=False)
               .agg(net_present_value_baseline=("net_present_value_baseline", _sum_or_na),
                    net_present_value_shock=("net_present_value_shock", _sum_or_na))
               .reset_index())
    pd_agg = (pdr.groupby(["run_id", "company_id", "sector", "term"], dropna=False)
              .agg(pd_baseline=("pd_baseline", "median"), pd_shock=("pd_shock", "median"))
              .reset_index())
    return npv_agg.merge(pd_agg, on=["run_id", "company_id", "sector"], how="inner")


def join_simple_portfolio_to_trisk(loans: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::join_simple_portfolio_to_trisk."""
    keys = ["company_id", "term"] + [k for k in ("sector", "technology")
                                     if k in loans.columns and k in agg.columns]
    left = loans.assign(company_id=loans["company_id"].astype("string"))
    right = agg.assign(company_id=agg["company_id"].astype("string"))
    return left.merge(right, on=keys, how="left")


def _share_alloc(df, run_keys, exposure_col):
    g = df.groupby(run_keys, dropna=False)
    has = g["net_present_value_baseline"].transform(lambda s: s.notna().any())
    npv_pos = df["net_present_value_baseline"].clip(lower=0)
    total = npv_pos.groupby([df[k] for k in run_keys], dropna=False).transform("sum")
    n = g["net_present_value_baseline"].transform("size")
    share = np.where(~has, np.nan, np.where(total > 0, npv_pos / total.where(total > 0), 1 / n))
    return df[exposure_col] * share


def _rescale(df, grouping_keys, anchor_keys):
    grouped = (df.groupby(grouping_keys, dropna=False)["exposure_value_usd_share_run"]
               .agg(_mean_or_na).rename("avg_exposure_value_usd_share").reset_index())
    orig = (df.groupby(anchor_keys, dropna=False)["exposure_value_usd"].mean()
            .rename("exposure_value_usd").reset_index())
    grouped = grouped.merge(orig, on=anchor_keys, how="left")
    tot = grouped.groupby(anchor_keys, dropna=False)["avg_exposure_value_usd_share"] \
        .transform(_sum_or_na)
    scale = np.where(tot.isna(), np.nan, np.where(tot == 0, 0.0,
                                                   grouped["exposure_value_usd"] / tot.where(tot != 0)))
    grouped["exposure_value_usd_share"] = grouped["avg_exposure_value_usd_share"] * scale
    return grouped[grouping_keys + ["exposure_value_usd_share"]]


def add_exposure_share_from_npv(df: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::add_exposure_share_from_npv (company-level)."""
    gk = [k for k in (".portfolio_index", "company_id", "company_name", "term",
                      "loss_given_default", "sector", "technology") if k in df.columns]
    ak = [".portfolio_index", "company_id", "company_name", "term", "loss_given_default"]
    w = df.copy()
    w["exposure_value_usd_share_run"] = _share_alloc(w, [".portfolio_index", "run_id"],
                                                     "exposure_value_usd")
    return df.merge(_rescale(w, gk, ak), on=gk, how="left")


def add_exposure_share_from_npv_row_level(df: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::add_exposure_share_from_npv_row_level (technology-level)."""
    ek = [k for k in ("company_id", "country_iso2", "term", "loss_given_default") if k in df.columns]
    dk = [c for c in ("sector", "technology") if c in df.columns and not df[c].isna().all()]
    w = df.copy()
    w["group_exposure_value_usd"] = w.groupby(["run_id"] + ek, dropna=False)["exposure_value_usd"] \
        .transform("mean")
    w["exposure_value_usd_share_run"] = _share_alloc(w, ["run_id"] + ek, "group_exposure_value_usd")
    return df.merge(_rescale(w, ek + dk, ek), on=ek + dk, how="left")


def compute_simple_portfolio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::compute_simple_portfolio_metrics."""
    d = df.copy()
    b, s = d["net_present_value_baseline"], d["net_present_value_shock"]
    d["net_present_value_difference"] = s - b
    d["net_present_value_change"] = np.where(b.isna() | (b == 0), 0.0,
                                             d["net_present_value_difference"] / b.where(b != 0))
    d["exposure_share_loss_usd"] = d["net_present_value_change"] * d["exposure_value_usd_share"]
    d["exposure_at_default"] = d["exposure_value_usd_share"]
    d["lgd_weighted_exposure"] = d["exposure_at_default"] * d["loss_given_default"]
    d["pd_difference"] = d["pd_shock"] - d["pd_baseline"]
    lw = d["exposure_at_default"] * d["loss_given_default"]
    d["expected_loss_baseline"] = lw * d["pd_baseline"]
    d["expected_loss_shock"] = lw * d["pd_shock"]
    d["expected_loss_difference"] = lw * d["pd_difference"]
    return d


def _aggregate(detail, id_cols):
    out = (detail.groupby(".portfolio_index", dropna=False)
           .agg(**{c: (c, "first") for c in id_cols},
                exposure_value_usd=("exposure_value_usd", "first"),
                exposure_loss_shock_usd=("exposure_share_loss_usd", _sum_or_na),
                expected_loss_baseline=("expected_loss_baseline", _sum_or_na),
                expected_loss_shock=("expected_loss_shock", _sum_or_na),
                expected_loss_difference=("expected_loss_difference", _sum_or_na))
           .reset_index())
    ev = out["exposure_value_usd"]
    out.insert(out.columns.get_loc("exposure_loss_shock_usd") + 1, "exposure_change_percent",
               np.where(ev.isna() | (ev == 0), np.nan, out["exposure_loss_shock_usd"] / ev.where(ev != 0)))
    return out


def aggregate_simple_portfolio_results(detail: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::aggregate_simple_portfolio_results."""
    return _aggregate(detail, ["company_id", "company_name", "term", "loss_given_default"])


def aggregate_simple_portfolio_results_row_level(detail: pd.DataFrame) -> pd.DataFrame:
    """trisk.analysis::aggregate_simple_portfolio_results_row_level."""
    ids = [c for c in ("company_id", "company_name", "country_iso2", "sector", "technology",
                       "term", "loss_given_default") if c in detail.columns]
    return _aggregate(detail, ids)


def analyse(loans: pd.DataFrame, npv: pd.DataFrame, pdr: pd.DataFrame, shape: str) -> tuple:
    """The post-run steps of run_trisk_on_simple_portfolio for ONE TRISK run (one scenario
    combination): (tech_detail, portfolio_results)."""
    agg = aggregate_trisk_outputs_simple(npv, pdr)
    j = join_simple_portfolio_to_trisk(loans, agg)
    j = add_exposure_share_from_npv_row_level(j) if shape == "technology" else add_exposure_share_from_npv(j)
    j = compute_simple_portfolio_metrics(j)
    res = (aggregate_simple_portfolio_results_row_level(j) if shape == "technology"
           else aggregate_simple_portfolio_results(j))
    detail = j.drop(columns=["exposure_value_usd"])
    return detail.drop(columns=[".portfolio_index"]), res.drop(columns=[".portfolio_index"]), j


def terms_outside_grid(loans: pd.DataFrame, pdr: pd.DataFrame) -> set:
    """Loan terms with no TRISK PD (R warn_terms_outside_grid): reported, not dropped."""
    return set(loans["term"].dropna().unique()) - set(pd.to_numeric(pdr["term"]).dropna().unique())


# ---- PD / EL integration -------------------------------------------------------------------
def pd_lifetime_to_annual(p, term):
    term = np.where(np.asarray(term, float) > 0, term, np.nan)
    return 1 - (1 - np.asarray(p, float)) ** (1 / term)


def pd_annual_to_lifetime(p, term):
    term = np.where(np.asarray(term, float) >= 0, term, np.nan)
    return 1 - (1 - np.asarray(p, float)) ** term


def _norm_ppf(x):
    from scipy.stats import norm
    return norm.ppf(x)


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def apply_pd_method(internal, baseline, shock, method, floor=ZSCORE_FLOOR, cap=ZSCORE_CAP):
    """trisk.analysis apply_pd_method (internal)."""
    internal, baseline, shock = (np.asarray(x, float) for x in (internal, baseline, shock))
    if method == "absolute":
        return internal + (shock - baseline)
    if method == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(baseline != 0, (shock - baseline) / np.where(baseline != 0, baseline, 1), 0)
        return internal * (1 + pct)
    clip = lambda x: np.minimum(np.maximum(x, floor), cap)  # noqa: E731
    return _norm_cdf(_norm_ppf(clip(internal)) + _norm_ppf(clip(shock)) - _norm_ppf(clip(baseline)))


def _notional(df):
    return df["exposure_at_default"] if "exposure_at_default" in df.columns else df["exposure_value_usd"]


def aggregate_pd_integration(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """trisk.analysis::aggregate_pd_integration (EAD-weighted)."""
    d = df.assign(_n=_notional(df), _g=1)
    keys = group_cols or ["_g"]
    rows = []
    for k, g in d.groupby(keys, dropna=False):
        tot = g["_n"].sum()
        w = {c: (g[c] * g["_n"]).sum() / tot if tot else np.nan
             for c in ("internal_pd", "pd_baseline", "pd_shock", "trisk_adjusted_pd")}
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r.update(total_exposure_usd=tot, weighted_pd_internal=w["internal_pd"],
                 weighted_pd_baseline=w["pd_baseline"], weighted_pd_shock=w["pd_shock"],
                 weighted_pd_adjusted=w["trisk_adjusted_pd"])
        r["weighted_pd_adjustment"] = r["weighted_pd_adjusted"] - r["weighted_pd_internal"]
        wi = r["weighted_pd_internal"]
        r["weighted_pd_adjustment_pct"] = r["weighted_pd_adjustment"] / wi if wi and not np.isnan(wi) else np.nan
        rows.append(r)
    return pd.DataFrame(rows).drop(columns=["_g"], errors="ignore")


def integrate_pd(detail: pd.DataFrame, internal, method: str = "zscore") -> tuple:
    """trisk.analysis::integrate_pd: (rows with trisk_adjusted_pd, EAD-weighted aggregate)."""
    internal = np.asarray(internal, float)
    if np.isnan(internal).all():
        raise PortfolioError("integrate PD: every internal PD is blank")
    b, s = detail["pd_baseline"].to_numpy(float), detail["pd_shock"].to_numpy(float)
    adj = np.clip(apply_pd_method(internal, b, s, method), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = detail.assign(internal_pd=internal, pd_change=s - b,
                            pd_change_pct=np.where(b != 0, (s - b) / np.where(b != 0, b, 1), np.nan),
                            trisk_adjusted_pd=adj, pd_adjustment=adj - internal)
    return out, aggregate_pd_integration(out)


def aggregate_el_integration(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """trisk.analysis::aggregate_el_integration (bps over notional EAD)."""
    d = df.assign(_n=_notional(df), _g=1)
    keys = group_cols or ["_g"]
    rows = []
    for k, g in d.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r.update(total_exposure_usd=g["_n"].sum(), total_el_internal=g["internal_el"].sum(),
                 total_el_baseline=g["expected_loss_baseline"].sum(),
                 total_el_shock=g["expected_loss_shock"].sum(),
                 total_el_adjusted=g["trisk_adjusted_el"].sum())
        r["total_el_adjustment"] = r["total_el_adjusted"] - r["total_el_internal"]
        ead = r["total_exposure_usd"]
        bps = (lambda x: abs(x) / ead * BPS) if ead else (lambda x: np.nan)
        r["el_adjusted_bps"] = bps(r["total_el_adjusted"])
        r["el_adjustment_bps"] = np.sign(r["total_el_adjustment"]) * bps(r["total_el_adjustment"])
        rows.append(r)
    return pd.DataFrame(rows).drop(columns=["_g"], errors="ignore")


def integrate_el(detail: pd.DataFrame, internal, method: str = "zscore") -> tuple:
    """trisk.analysis::integrate_el."""
    internal = np.asarray(internal, float)
    if np.isnan(internal).all():
        raise PortfolioError("integrate EL: every internal EL is blank")
    b = detail["expected_loss_baseline"].to_numpy(float)
    s = detail["expected_loss_shock"].to_numpy(float)
    if method == "zscore":
        lw = detail["lgd_weighted_exposure"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = lambda x: np.where(lw > 0, np.abs(x) / np.where(lw > 0, lw, 1), 0)  # noqa: E731
        adj = lw * apply_pd_method(eff(internal), eff(b), eff(s), "zscore")
    else:
        adj = apply_pd_method(internal, b, s, method)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = detail.assign(internal_el=internal, el_change=s - b,
                            el_change_pct=np.where(b != 0, (s - b) / np.where(b != 0, b, 1), np.nan),
                            trisk_adjusted_el=adj, el_adjustment=adj - internal)
    return out, aggregate_el_integration(out)


# ---- totals ------------------------------------------------------------------------------
def totals(detail: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    """Portfolio totals (exposure, value loss, expected loss baseline/shock, bps)."""
    d = detail.assign(_g=1)
    keys = list(group_cols or []) or ["_g"]
    out = (d.groupby(keys, dropna=False)
           .agg(exposure_at_default=("exposure_at_default", _sum_or_na),
                exposure_loss_shock_usd=("exposure_share_loss_usd", _sum_or_na),
                expected_loss_baseline=("expected_loss_baseline", _sum_or_na),
                expected_loss_shock=("expected_loss_shock", _sum_or_na))
           .reset_index())
    ead = out["exposure_at_default"].where(out["exposure_at_default"] != 0)
    out["value_change_pct"] = out["exposure_loss_shock_usd"] / ead
    out["expected_loss_change_bps"] = (out["expected_loss_shock"] - out["expected_loss_baseline"]) / ead * BPS
    return out.drop(columns=["_g"], errors="ignore")
