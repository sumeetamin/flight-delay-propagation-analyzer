"""Fetch a bounded 2023 BTS on-time sample with chronological months."""
import io
import random
import time
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

BASE = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
FIELDS = ["Month", "DayofMonth", "DayOfWeek", "FlightDate", "Reporting_Airline", "Origin", "Dest",
          "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance", "ArrDelay", "Cancelled", "Diverted"]
DATA = Path("data")
OUTPUT = DATA / "flights.csv"
SAMPLE_PER_MONTH = 100_000


def load_month(month):
    archive = DATA / f"bts_2023_{month:02d}.zip"
    if not archive.exists():
        urlretrieve(BASE.format(year=2023, month=month), archive)
    with ZipFile(archive) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as stream:
            frame = pd.read_csv(stream, usecols=FIELDS, low_memory=False)
    frame["ArrDelay"] = pd.to_numeric(frame["ArrDelay"], errors="coerce")
    frame = frame[(frame["Cancelled"] == 0) & (frame["Diverted"] == 0)].dropna(
        subset=["ArrDelay", "FlightDate", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "Distance"])
    frame = frame.sample(n=min(SAMPLE_PER_MONTH, len(frame)), random_state=2023 + month)
    return frame


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    frames = [load_month(m) for m in (1, 2, 3)]
    combined = pd.concat(frames, ignore_index=True).sort_values("FlightDate")
    combined.to_csv(OUTPUT, index=False)
    counts = combined.groupby("Month").size().to_dict()
    print(f"Wrote {len(combined)} sampled completed flights: {counts}")


if __name__ == "__main__":
    main()

