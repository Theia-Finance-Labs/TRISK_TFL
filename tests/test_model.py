"""Tests for trisk_tfl.run_trisk_model.

Created with the support of AI tools.

    pytest -v

Every case in tests/cases/ runs once without R (the tables are complete and plausible). Where
tests/r/run_reference.R has written R trisk.model 2.7.0's outputs to <case>/expected/, every
output table is compared with them (the parity workflow does this on GitHub).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from trisk_tfl import run_trisk_model

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = sorted(d for d in os.listdir(os.path.join(HERE, "cases"))
               if os.path.isfile(os.path.join(HERE, "cases", d, "case.json")))
FILES = {"assets": "assets.csv", "scenarios": "scenarios.csv",
         "financial": "financial_features.csv", "carbon": "ngfs_carbon_price.csv"}
TABLES = {"npv": ("npv_results", ["asset_id", "technology"]),
          "pd": ("pd_results", ["company_id", "sector", "term"]),
          "trajectories": ("company_trajectories", ["asset_id", "technology", "year"])}
TOL = dict(rtol=1e-6, atol=1e-6)


def _read(path):
    return pd.read_csv(path, comment="#", dtype={"asset_id": str, "company_id": str})


def _runs(name):
    cdir = os.path.join(HERE, "cases", name)
    case = json.load(open(os.path.join(cdir, "case.json"), encoding="utf-8"))
    data = {k: _read(os.path.join(cdir, "inputs", f)) for k, f in FILES.items()}
    for run in case["runs"]:
        scen = data["scenarios"]
        if run["sectors"]:
            scen = scen[scen["sector"].isin(run["sectors"])]
        assets = data["assets"]
        if case.get("countries"):
            assets = assets[assets["country_iso2"].isin(case["countries"])]
        got = run_trisk_model(assets, scen, data["financial"], data["carbon"],
                              run_id=run["run_id"], **run["args"])
        yield cdir, run, got


@pytest.mark.parametrize("name", CASES)
def test_runs_and_is_plausible(name):
    for _, run, got in _runs(name):
        for key, cols in TABLES.values():
            assert key in got and len(got[key]) > 0, f"{run['run_id']}: {key} is empty"
            assert set(cols) <= set(got[key].columns)
        pdr = got["pd_results"]
        for c in [c for c in pdr.columns if c.startswith("pd_")]:
            v = pdr[c].dropna()
            assert ((v >= 0) & (v <= 1)).all(), f"{run['run_id']}: {c} outside 0..1"


@pytest.mark.parametrize("name", CASES)
def test_matches_r_trisk_model(name):
    if not os.path.isdir(os.path.join(HERE, "cases", name, "expected")):
        pytest.skip("no R outputs: run tests/r/run_reference.R (the parity workflow does)")
    for cdir, run, got in _runs(name):
        for part, (key, keys) in TABLES.items():
            want = _read(os.path.join(cdir, "expected", f"{run['run_id']}_{part}.csv"))
            g = got[key].sort_values(keys).reset_index(drop=True)
            w = want.sort_values(keys).reset_index(drop=True)
            assert len(g) == len(w), f"{run['run_id']} {part}: {len(g)} rows vs R {len(w)}"
            for c in w.columns:
                if c in g.columns and pd.api.types.is_numeric_dtype(w[c]):
                    assert np.allclose(g[c].astype(float), w[c].astype(float),
                                       equal_nan=True, **TOL), f"{run['run_id']} {part}: {c} differs"
