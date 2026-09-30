"""Generate a reproducible dimensional conversion dataset."""

from __future__ import annotations

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


def write_csv(output_dir: str | Path = "data") -> None:
    """Write the generated baseline and current datasets to CSV files."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    baseline, current = generate_dataset()
    baseline.to_csv(target / "baseline.csv", index=False)
    current.to_csv(target / "current.csv", index=False)


if __name__ == "__main__":
    write_csv()
