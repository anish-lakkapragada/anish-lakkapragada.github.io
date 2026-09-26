"""Export only the data needed to reproduce v16-B's two training figures.

Usage: python3 tools/export_swe2_video_figures.py [path/to/v16-B/assets/data.json]
The film already averages seeds within problems, then weights problems equally.
Keep its recorded precision and every checkpoint; smoothing happens in the viewer
using the same endpoint-preserving rule as the film.
"""

import json
import sys
from pathlib import Path


def main():
    site = Path(__file__).resolve().parents[1]
    source = (Path(sys.argv[1]) if len(sys.argv) > 1 else
              site.parent / "swe2-extended/videos/full-film/v16B-v15edit/assets/data.json")
    film = json.loads(source.read_text())
    data = {
        "version": 1,
        "source": film["source"],
        "steps": film["steps"],
        "methods": film["methods"],
        "tasks": film["tasks"],
        "mean_abs": film["mean_abs"],
        "medium_threads": [method[1] for method in film["threads"]],
    }
    assert data["steps"][0] == 0 and data["steps"][-1] == 1000
    assert len(data["methods"]) == 6 and len(data["tasks"]) == 100
    for method in data["medium_threads"]:
        assert len(method) == len(data["tasks"])
        assert all(len(path) == len(data["steps"]) for path in method)
    target = site / "assets/swe-2-extended/data/video-figures.json"
    target.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Exported {target.stat().st_size:,} bytes from {source}")


if __name__ == "__main__":
    main()
