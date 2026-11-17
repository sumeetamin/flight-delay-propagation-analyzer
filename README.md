# Flight Delay Propagation Analyzer

A reproducible scheduled-time arrival-delay regression baseline on official BTS domestic flight records. The benchmark compares a regularized model with a simple training-median baseline and preserves a chronological month holdout.

## Data and run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python fetch_data.py
python benchmark.py
```

The fetcher downloads BTS monthly files for January–March 2023 and keeps a seeded sample of up to 100,000 completed flights per month. Training uses January–February; March is untouched test data. Source archives and row-level data stay local and are ignored by Git.

## Result

The test sample contains 200,000 train and 100,000 March test flights. Ridge regression achieved **23.49 minutes MAE**, slightly worse than the training-median baseline (**22.50 minutes**); the model therefore does not improve this baseline on the March sample. Alpha was selected using February validation only, and still underperformed the validation median. Full fingerprints and results are in `reports/benchmark.json`.

The negative result is useful: route and schedule features alone were insufficient here. The model excludes airport weather, network state, and operational updates that would be needed for a stronger decision-time forecast.

## Scope and leakage controls

Target is arrival delay in minutes, clipped to [-60, 240]. Predictors are carrier, origin/destination, calendar, scheduled departure/arrival time, scheduled elapsed time, and distance. Actual departure/arrival times, departure delay, taxi and wheels-off/on measures, delay attribution fields, tail number, and flight number are excluded.

Cancelled and diverted flights are not modeled. This three-month offline baseline is not a live operations forecast and does not establish performance across seasons or years.

## Source

U.S. Department of Transportation, Bureau of Transportation Statistics, [Reporting Carrier On-Time Performance (1987-present)](https://transtats.bts.gov/ONTIME/Index.aspx). Monthly files are available from the [BTS PREZIP directory](https://transtats.bts.gov/PREZIP/). Data are not redistributed in this repository.

No code license file is included yet. Third-party data remain subject to source terms.

