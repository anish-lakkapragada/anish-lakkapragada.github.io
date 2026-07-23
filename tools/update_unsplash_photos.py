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
# Photos hand-rejected via the /choser review; excluded from photos.json even
# when the Unsplash listing still returns them.
REJECTS = Path(__file__).resolve().parent / "photo_rejects.json"

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


# Descriptions read like "A [breeding male] dark-eyed junco resting on a
# branch." — the species is the noun phrase between the article/modifiers
# and the first verb/preposition. Species casing in the source text varies,
# so the result is re-cased per bird-name convention ("White-crowned
# Sparrow"). A miss returns None: hand-curated values in photos.json
# survive refreshes, and the caption falls back to place-only.
_LEAD_PATTERNS = (
    "a close-up photograph of ", "a close-up photo of ", "a close-up shot of ",
    "a close up of ", "a close-up of ", "a photograph of ", "a photo of ",
    "an upward shot of ", "a side profile of ", "side profile of ",
    "the face of ", "a ", "an ", "the ",
)
_MODIFIERS = {
    "male", "female", "breeding", "nonbreeding", "immature", "juvenile",
    "young", "adult", "banded", "dark-lored", "wintering", "molting",
}
_STOPWORDS = {
    "rest", "rests", "resting", "perch", "perches", "perched", "perching",
    "stand", "stands", "standing", "sit", "sits", "sitting", "stare",
    "stares", "staring", "look", "looks", "looking", "fly", "flies",
    "flying", "chirp", "chirps", "chirping", "sing", "sings", "singing",
    "walk", "walks", "walking", "swim", "swims", "swimming", "eat", "eats",
    "eating", "prepare", "prepares", "preparing", "turn", "turns",
    "turning", "jump", "jumps", "jumping", "hungrily", "directly",
    "with", "on", "in", "at", "near", "by", "from", "while", "is", "are",
    "was", "and", "as", "that", "up", "down", "against",
}


def _title_case_species(words):
    out = []
    for w in words:
        parts = w.split("-")
        # "white-crowned" -> "White-crowned" (only the first part capitalized)
        parts = [parts[0].capitalize()] + [p.lower() for p in parts[1:]]
        out.append("-".join(parts))
    return " ".join(out)


def extract_species(desc):
    if not desc:
        return None
    text = desc.strip().split(".")[0].lower()
    changed = True
    while changed:
        changed = False
        for lead in _LEAD_PATTERNS:
            if text.startswith(lead):
                text = text[len(lead):]
                changed = True
    words = text.replace(",", " ").split()
    while words and words[0] in _MODIFIERS:
        words.pop(0)
    species = []
    for w in words:
        w = w.rstrip("!?:;)")
        if w in _STOPWORDS:
            break
        if w.endswith("'s"):  # "...a Mallard's beak" -> Mallard
            species.append(w[:-2])
            break
        species.append(w)
    if not species or len(species) > 5:
        return None
    return _title_case_species(species)


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

    rejects = set()
    if REJECTS.exists():
        rejects = set(json.loads(REJECTS.read_text()))

    out, new = [], 0
    for p in listing:
        pid = p["id"]
        if pid in rejects:
            continue
        entry = existing.get(pid)
        if entry is not None and "species" in entry:
            out.append(entry)
            continue
        detail = get(f"https://unsplash.com/napi/photos/{pid}")
        if entry is not None:
            # backfill species onto an entry from before the field existed;
            # keep its place (may have been hand-fixed) and url.
            entry = dict(entry)
            entry["species"] = extract_species((detail or {}).get("description"))
            out.append(entry)
        else:
            out.append({
                "id": pid,
                "url": p["urls"]["raw"].split("?")[0],
                "place": format_place((detail or {}).get("location")),
                "species": extract_species((detail or {}).get("description")),
            })
        new += 1
        time.sleep(0.15)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    located = sum(1 for p in out if p["place"])
    print(f"wrote {len(out)} photos ({new} new, {located} with locations) to {OUT}")


if __name__ == "__main__":
    main()
