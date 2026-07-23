#!/usr/bin/env python3
"""Refresh assets/data/photos.json from unsplash.com/@anishlk.

Incremental: photos already in the JSON keep their entry; only new photo ids
cost a detail request (needed for location). Photos whose location can't be
formatted as "City, ST" are kept with place=null so they aren't refetched;
the landing page skips them. Fails soft: on any fetch problem the committed
file is left untouched and we exit 0 so CI builds proceed.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

USERNAME = "anishlk"
OUT = Path(__file__).resolve().parent.parent / "assets" / "data" / "photos.json"

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def get(url, retries=2):
    for attempt in range(retries + 1):
        r = subprocess.run(
            ["curl", "-s", "--max-time", "20", url],
            capture_output=True, text=True,
        )
        try:
            return json.loads(r.stdout)
        except (json.JSONDecodeError, ValueError):
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return None


def format_place(loc):
    """Mirror of formatPlace in _layouts/landing.html: "City, ST" (US) or
    "City, Country" elsewhere; None when there's not enough to format."""
    if not loc or not loc.get("name"):
        return None
    parts = [p.strip() for p in loc["name"].split(",") if p.strip()]
    if not parts:
        return None
    if parts[-1].upper() not in ("USA", "UNITED STATES"):
        if loc.get("city") and loc.get("country"):
            return f"{loc['city']}, {loc['country']}"
        return None
    parts.pop()
    state = parts.pop() if parts else ""
    state = STATES.get(state, state)
    if len(state) != 2 or not state.isalpha() or not state.isupper():
        return None
    city = loc.get("city") or (parts.pop() if parts else None)
    return f"{city}, {state}" if city else None


def main():
    existing = {}
    if OUT.exists():
        existing = {p["id"]: p for p in json.loads(OUT.read_text())}

    listing = []
    page = 1
    while True:
        batch = get(f"https://unsplash.com/napi/users/{USERNAME}/photos?per_page=30&page={page}")
        if not isinstance(batch, list):
            print(f"warning: photo list fetch broke on page {page}; keeping committed file")
            return
        if not batch:
            break
        listing.extend(batch)
        if len(batch) < 30:
            break
        page += 1
        time.sleep(0.1)

    if not listing:
        print("warning: empty photo list; keeping committed file")
        return

    out, new = [], 0
    for p in listing:
        pid = p["id"]
        if pid in existing:
            out.append(existing[pid])
            continue
        detail = get(f"https://unsplash.com/napi/photos/{pid}")
        place = format_place((detail or {}).get("location"))
        out.append({"id": pid, "url": p["urls"]["raw"].split("?")[0], "place": place})
        new += 1
        time.sleep(0.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    located = sum(1 for p in out if p["place"])
    print(f"wrote {len(out)} photos ({new} new, {located} with locations) to {OUT}")


if __name__ == "__main__":
    main()
