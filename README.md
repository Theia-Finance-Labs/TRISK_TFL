# TRISK in Python (trisk_tfl)

TRISK is a transition risk stress test for companies and loan books. It asks what a climate
transition scenario does to each company's production, profits, value and probability of
default, compared with a baseline scenario.

This repository is Theia Finance Labs' Python implementation of TRISK. It follows the R
package [trisk.model](https://github.com/Theia-Finance-Labs/trisk.model) 2.7.0 (commit
`748ef51`) step by step and gives the same results. It also holds a Python implementation of
the loan-book step of [trisk.analysis](https://github.com/Theia-Finance-Labs/trisk.analysis)
1.2.3 (commit `1f5a1e2`).

Created with the support of AI tools.

## Install

```bash
pip install git+https://github.com/Theia-Finance-Labs/TRISK_TFL
```

Needs Python 3.10 or later, pandas (2.2 or 3.x), NumPy and SciPy.

## Use

```python
import pandas as pd
from trisk_tfl import run_trisk_model

read = lambda f: pd.read_csv(f, dtype={"asset_id": str, "company_id": str})
res = run_trisk_model(
    read("assets.csv"), read("scenarios.csv"),
    read("financial_features.csv"), read("ngfs_carbon_price.csv"),
    baseline_scenario="SYNTH_CP", target_scenario="SYNTH_NZ",
    scenario_geography="Global", shock_year=2030,
)
res["npv_results"], res["pd_results"], res["company_trajectories"]
```

The four inputs, the arguments and their defaults are the same as in
`trisk.model::run_trisk_model()`; see trisk.model's documentation for the column definitions.
`tests/cases/*/inputs/` holds made-up examples of all four files.

The loan-book functions are in `trisk_tfl.portfolio`: `match_loans`, `analyse`,
`integrate_pd`, `integrate_el` and `apply_lgd`, each named after the trisk.analysis function
it mirrors.

## Same results as R

`.github/workflows/parity.yml` installs R trisk.model 2.7.0, runs it on every case in
`tests/cases/` and compares each output table (net present values, probabilities of default,
company trajectories) with this package, to a tolerance of 1e-6. It runs on every change and
weekly.

Two deliberate differences, both in edge cases, are documented at the top of
`src/trisk_tfl/model.py`: a fill-down that R does across assets is done per asset here, and
dividend years are numbered by year rather than by row order. Neither changes the results of
the test cases.

## Data

The test cases contain made-up companies, assets, scenarios and prices only. This repository
holds no company data and no third-party scenario data.

## Licence

MIT, see [LICENSE](LICENSE).

This package translates trisk.model and part of trisk.analysis into Python. Theia Finance
Labs, the copyright holder of both, publishes the translation under the MIT licence. The R
packages themselves remain available under their own licences in their own repositories;
this repository contains none of their code, and the parity workflow downloads trisk.model
only to run it.

The TRISK method was developed by Theia Finance Labs and the authors of trisk.model.
