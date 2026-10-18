"""Download a reproducible Eurostat reference snapshot (network only on request).

Run from the repository root: python scripts/refresh_reference_data.py
The app and tests use the checked-in files and never contact Eurostat.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
FILTERS = {"lang": "EN", "geo": "ES", "sex": "M", "time": "2021"}


def observations(payload: dict) -> list[dict]:
    """Decode sparse JSON-stat values using the published dimension order."""
    dimensions = payload["id"]
    codes = {
        name: {
            position: code
            for code, position in payload["dimension"][name]["category"]["index"].items()
        }
        for name in dimensions
    }
    rows = []
    for flat_index, value in payload["value"].items():
        index = int(flat_index)
        row = {"value": value, "flag": payload.get("status", {}).get(str(flat_index), "")}
        for name, size in reversed(list(zip(dimensions, payload["size"], strict=True))):
            index, position = divmod(index, size)
            row[name] = codes[name][position]
        rows.append(row)
    return rows


def fetch(dataset: str, **filters: str) -> tuple[dict, dict]:
    url = API + dataset + "?" + urlencode(FILTERS | filters)
    with urlopen(url, timeout=60) as response:
        body = response.read()
    payload = json.loads(body)
    return payload, {
        "dataset": dataset,
        "url": url,
        "updated": payload.get("updated"),
        "response_sha256": hashlib.sha256(body).hexdigest(),
    }


def write_csv(filename: str, fields: list[str], rows: list[dict]) -> dict:
    path = ROOT / "data" / filename
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "file": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
    }


def main() -> None:
    life, life_source = fetch("demo_mlifetable")
    life_rows = [
        {
            "age": int(r["age"][1:]),
            "indicator": r["indic_de"],
            "value": r["value"],
            "flag": r["flag"],
        }
        for r in observations(life)
        if r["age"][1:].isdigit()
        and 35 <= int(r["age"][1:]) <= 94
        and r["indic_de"] in {"DEATHRATE", "LIFEXP"}
    ]
    # The open 95+ group is a separate observation, never an exact-age 95 rate.
    life_rows += [
        {"age": "95+", "indicator": r["indic_de"], "value": r["value"], "flag": r["flag"]}
        for r in observations(life)
        if r["age"] == "Y_GE95" and r["indic_de"] in {"DEATHRATE", "LIFEXP"}
    ]
    if len(life_rows) != 122:
        raise ValueError("Incomplete life-table reference; refusing to replace snapshot")
    totals, totals_source = fetch("hlth_cd_acdr2", icd10="TOTAL")
    if not totals["value"]:
        totals, totals_source = fetch("hlth_cd_acdr2", icd10="A-R_V-Y")
    age_codes = {f"Y{age}-{age + 4}" for age in range(35, 95, 5)} | {"Y_GE95"}
    total_rows = [
        {"age_code": r["age"], "rate": r["value"], "flag": r["flag"]}
        for r in observations(totals)
        if r["age"] in age_codes
    ]
    if len(total_rows) != 13:
        raise ValueError("Incomplete all-cause reference; refusing to replace snapshot")
    (ROOT / "data").mkdir(exist_ok=True)
    files = [
        write_csv(
            "spain_male_2021_life_table.csv", ["age", "indicator", "value", "flag"], life_rows
        ),
        write_csv("spain_male_2021_all_causes.csv", ["age_code", "rate", "flag"], total_rows),
    ]
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "population": "Spain, males, 2021",
        "sources": [life_source, totals_source],
        "files": files,
        "units": {
            "DEATHRATE": "Deaths per person-year (Mx), not probability qx",
            "LIFEXP": "Remaining years at exact age (ex)",
            "all_causes_rate": "Deaths per 100,000 population",
        },
        "notes": [
            "95+ is an open age group, not a single-year observation.",
            "Cause-of-death rates and life-table rates use different statistical production methods; equality is not assumed.",
            "mortality.csv predates this snapshot; its original extraction metadata are unavailable.",
        ],
    }
    (ROOT / "data" / "reference_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved reference tables:", ", ".join(f["file"] for f in files))


if __name__ == "__main__":
    main()
