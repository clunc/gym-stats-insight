#!/usr/bin/env python3
"""Fetch gym history JSON from the local API and print it or a summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

DEFAULT_URL = "http://thinkcentre-janik.fritz.box/api/history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch gym history JSON from local API")
    parser.add_argument("--url", default=DEFAULT_URL, help="API endpoint URL")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary instead of the raw JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (ignored with --summary)",
    )
    parser.add_argument(
        "--output",
        default="data/mock_history.json",
        help="Write the JSON response to a file",
    )
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Response was not valid JSON") from exc


def iso_to_date(ts: str) -> str:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()


def summarize(payload: dict) -> dict:
    history = payload.get("history", [])
    workouts = [h for h in history if h.get("type") == "workout"]

    dates = {iso_to_date(h["timestamp"]) for h in workouts if h.get("timestamp")}
    total_sets = len(workouts)
    total_reps = sum(int(h.get("reps", 0)) for h in workouts)
    per_day_volume = defaultdict(int)

    for h in workouts:
        ts = h.get("timestamp")
        if not ts:
            continue
        date = iso_to_date(ts)
        weight = float(h.get("weight", 0))
        reps = int(h.get("reps", 0))
        per_day_volume[date] += weight * reps

    return {
        "total_sessions": len(dates),
        "total_sets": total_sets,
        "total_reps": total_reps,
        "per_day_volume": dict(sorted(per_day_volume.items())),
    }


def main() -> int:
    args = parse_args()
    payload = fetch_json(args.url)

    if args.summary:
        print(json.dumps(summarize(payload), indent=2))
        return 0

    if args.output:
        output_path = args.output
        output_dir = output_path.rsplit("/", 1)[0]
        if output_dir and output_dir != output_path:
            import os

            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2 if args.pretty else None, sort_keys=True)

    if args.pretty:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
