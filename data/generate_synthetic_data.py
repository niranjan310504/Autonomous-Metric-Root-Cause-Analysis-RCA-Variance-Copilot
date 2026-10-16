"""Generate a reproducible dimensional conversion dataset."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def generate_dataset(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return baseline and anomaly aggregates with a known device/channel driver."""

    del seed
    rows: list[dict[str, object]] = []
    for period, rates in [
        ("baseline", {"mobile": 0.085, "desktop": 0.095, "partner": 0.09}),
        ("current", {"mobile": 0.052, "desktop": 0.092, "partner": 0.088}),
    ]:
        for device, device_share in [("mobile", 0.55), ("desktop", 0.35), ("tablet", 0.10)]:
            for channel, channel_share in [("direct", 0.6), ("partner", 0.4)]:
                visits = int(10000 * device_share * channel_share)
                rate = rates["mobile" if device == "mobile" else "desktop"]
                if device == "tablet":
                    rate = 0.07
                if period == "current" and device == "mobile" and channel == "partner":
                    rate -= 0.01
                rows.append(
                    {
                        "period": period,
                        "device": device,
                        "channel": channel,
                        "region": "global",
                        "visits": visits,
                        "conversions": round(visits * rate),
                    }
                )
    data = pd.DataFrame(rows)
    baseline = data[data["period"] == "baseline"].drop(columns="period").reset_index(drop=True)
    current = data[data["period"] == "current"].drop(columns="period").reset_index(drop=True)
    return baseline, current


def generate_timeseries_dataset(days: int = 30, seed: int = 7) -> pd.DataFrame:
    """Return daily aggregates with a documented final-weekend anomaly.

    The final Saturday and Sunday shift traffic toward mobile, especially the
    mobile-partner segment, and that segment also has a lower conversion rate.
    Earlier weekends retain the normal mix, so the default 30-day dataset
    produces a deterministic WoW mix and rate comparison. Rows remain
    aggregated by day and dimension; no individual visits are expanded.
    """

    if days <= 0:
        raise ValueError("days must be positive")

    generator = random.Random(seed)
    end_date = pd.Timestamp("2026-02-01")
    start_date = end_date - pd.Timedelta(days=days - 1)
    rows: list[dict[str, object]] = []
    normal_device_shares = {
        "mobile": 0.55,
        "desktop": 0.35,
        "tablet": 0.10,
    }
    anomaly_device_shares = {
        "mobile": 0.70,
        "desktop": 0.20,
        "tablet": 0.10,
    }
    channel_shares = {"direct": 0.60, "partner": 0.40}
    base_rates = {"mobile": 0.085, "desktop": 0.095, "tablet": 0.070}

    for offset in range(days):
        timestamp = start_date + pd.Timedelta(days=offset)
        is_final_weekend = timestamp >= end_date - pd.Timedelta(days=1)
        device_shares = anomaly_device_shares if is_final_weekend else normal_device_shares
        daily_visits = 10_000 + generator.randint(-250, 250)
        for device, device_share in device_shares.items():
            for channel, channel_share in channel_shares.items():
                visits = int(daily_visits * device_share * channel_share)
                rate = base_rates[device]
                if is_final_weekend and device == "mobile":
                    rate -= 0.015
                if is_final_weekend and device == "mobile" and channel == "partner":
                    rate -= 0.010
                rows.append(
                    {
                        "timestamp": timestamp,
                        "device": device,
                        "channel": channel,
                        "region": "global",
                        "visits": visits,
                        "conversions": round(visits * rate),
                    }
                )
    return pd.DataFrame(rows)


def write_csv(output_dir: str | Path = "data") -> None:
    """Write the generated baseline and current datasets to CSV files."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    baseline, current = generate_dataset()
    baseline.to_csv(target / "baseline.csv", index=False)
    current.to_csv(target / "current.csv", index=False)


if __name__ == "__main__":
    write_csv()
