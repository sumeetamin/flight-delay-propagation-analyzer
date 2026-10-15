# Flight Delay Propagation Analyzer

A time-ordered arrival-delay regressor with route/calendar predictors, leakage exclusions, and holdout error in minutes.

> **Portfolio prototype.** Code is provided; no benchmark results are claimed until you run it on the cited data and record the output.

## Use case

A time-ordered arrival-delay regressor with route/calendar predictors, leakage exclusions, and holdout error in minutes. The command below reads a local data export, applies a transparent baseline, writes holdout or decision-support outputs, and saves the fitted model where applicable.

## Data and provenance

[U.S. DOT BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/ONTIME/Index.aspx). A BTS on-time CSV with `FL_DATE` and `ARR_DELAY`. Use a route/month slice to fit on a workstation; the model excludes direct post-arrival measurements. Dataset files are not bundled. Respect the source terms and cite the dataset when publishing results.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Download/prepare the source data locally, then run:
python main.py --input data/On_Time.csv --date FL_DATE --target ARR_DELAY --out outputs
```

## Outputs

The `outputs/` directory contains the model-specific results (metrics and/or ranked CSV) and a serialized estimator where applicable. Outputs are reproducible with the same input, package versions, and random seed. The repository intentionally excludes downloaded data and generated artifacts.

## Evaluation and operating notes

Delay is influenced by weather, network effects, and airline operations. This baseline is not a real-time predictor unless those sources are joined with correct event-time semantics.

For a production review, preserve a time-based, untouched holdout; compare against a simple business baseline; inspect calibration and subgroup error; document data lineage; and monitor drift. The included baseline is not a production service.

## License

Code: MIT. Third-party datasets remain subject to their original terms.
