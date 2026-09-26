"""Export the recorded SWE-2 toy study for the article's interactive figures.

Run with Python 3 (standard library only):
    python3 tools/export_swe2_data.py [path/to/instance_sweep_derivative_init]

The overview contains all 100 problems and paired seed endpoints. Replay files
are fetched one problem at a time; they retain step 0, 1, and every 20 updates.
No training, interpolation, or simulated observations are used in this export.
"""

import gzip
import json
import math
import statistics
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
SOURCE = (Path(sys.argv[1]) if len(sys.argv) > 1 else
          SITE.parent / "swe2-extended/runs/instance_sweep_derivative_init")
DEST = SITE / "assets/swe-2-extended/data"
METHODS = ["fixed", "adapt_0", "adapt_0.25", "adapt_0.5", "adapt_0.75", "adapt_1"]


def compact(value):
    if isinstance(value, float):
        assert math.isfinite(value)
        return float(f"{value:.8g}")
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compact(data), separators=(",", ":")) + "\n")


def main():
    instances = json.loads((SOURCE / "instance_summaries.json").read_text())
    aggregate = json.loads((SOURCE / "improvement_vectors.json").read_text())
    curves = {c["method"]: c for c in aggregate["learning_curves"]}
    overview = {
        "version": 1,
        "description": "100 problems × 3 paired seeds × 6 methods; 1,000 updates",
        "normalization": "Normalize against each problem's exact initial policy, average seeds within problem, then weight problems equally.",
        "methods": METHODS,
        "steps": curves["fixed"]["steps"],
        "curves": [{
            "cost": curves[m]["mean_cost"], "success": curves[m]["success_pct"],
            "saving": [[-v for v in row] for row in curves[m]["relative_cost_change_pct"]],
            "gain": curves[m]["relative_success_gain_pct"],
        } for m in METHODS],
        "problems": [],
    }
    checks = 0
    for instance in instances:
        problem = {
            "id": instance["task"], "row": instance["good_index"],
            "column": instance["ratio_index"], "pg": instance["p_good"],
            "pb": instance["p_bad"], "ratio": instance["p_bad_ratio"],
            "seeds": instance["training_seeds"],
            "baseCost": instance["base_mean_cost"],
            "baseSuccess": instance["base_success_pct"], "results": [],
        }
        replay = {"id": problem["id"], "seeds": [], "steps": None, "rate": None}
        endpoints = [[] for _ in METHODS]
        for seed in problem["seeds"]:
            filename = SOURCE / "raw" / f"{problem['id']}_seed{seed}.json.gz"
            with gzip.open(filename, "rt") as stream:
                saved = json.load(stream)
            runs = {r["method"]: r for r in saved["runs"]}
            seed_data = {"id": seed, "runs": []}
            for mi, method in enumerate(METHODS):
                run = runs[method]
                history = [h for h in run["history"] if h["step"] in (0, 1) or h["step"] % 20 == 0]
                steps = [h["step"] for h in history]
                assert steps[-1] == 1000
                if replay["steps"] is not None:
                    assert replay["steps"] == steps
                replay["steps"] = steps
                replay["rate"] = [h.get("controller_learning_rate", 0) for h in history]
                record = {k: [] for k in ["q", "c", "s", "l", "used", "u", "v"]}
                for h in history:
                    record["q"].append(h["policy"]["q"])
                    record["c"].append(h["policy"]["mean_cost"])
                    record["s"].append([100 * s for s in h["success"]])
                    record["l"].append(h["penalties"])
                    record["used"].append(h.get("penalties_used", h["penalties"]))
                    batch = h.get("batch")
                    u = [(s / ref - 1) for s, ref in zip(batch["effort_success"], run["references"]["success"])] if batch else [0] * 3
                    v = [(1 - c / ref) for c, ref in zip(batch["effort_cost"], run["references"]["cost"])] if batch else [0] * 3
                    record["u"].append(u)
                    record["v"].append(v)
                    if batch and method != "fixed":
                        for e in range(3):
                            delta = h["controller_learning_rate"] * ((1-run["alpha"]) * u[e] - run["alpha"] * v[e])
                            assert math.isclose(h["penalties"][e], h["penalties_used"][e] * math.exp(delta), rel_tol=1e-11)
                            checks += 1
                seed_data["runs"].append(record)
                end = history[-1]
                endpoints[mi].append({
                    "q": end["policy"]["q"], "cost": end["policy"]["mean_cost"],
                    "success": [100*s for s in end["success"]],
                    "saving": [100*(1-c/c0) for c,c0 in zip(end["policy"]["mean_cost"], problem["baseCost"])],
                    "gain": [100*(100*s/s0-1) for s,s0 in zip(end["success"], problem["baseSuccess"])],
                })
            replay["seeds"].append(seed_data)
        for mi, method in enumerate(METHODS):
            result = {k: [statistics.mean(s[k][e] for s in endpoints[mi]) for e in range(3)] for k in ["cost", "success", "saving", "gain"]}
            result["q"] = statistics.mean(s["q"] for s in endpoints[mi])
            result["seeds"] = endpoints[mi]
            original = next(m for m in instance["methods"] if m["method"] == method)
            for e in range(3):
                assert math.isclose(result["cost"][e], original["mean_cost"][-1][e], rel_tol=1e-12)
                assert math.isclose(result["success"][e], original["success_pct"][-1][e], rel_tol=1e-12)
            problem["results"].append(result)
        overview["problems"].append(problem)
        write(DEST / "replays" / f"{problem['id']}.json", replay)
    # Check the plotted aggregate endpoints independently against all seed data.
    for mi in range(6):
        for key in ["cost", "success", "saving", "gain"]:
            for e in range(3):
                actual = statistics.mean(p["results"][mi][key][e] for p in overview["problems"])
                assert math.isclose(actual, overview["curves"][mi][key][-1][e], abs_tol=1e-9)
    write(DEST / "overview.json", overview)
    size = sum(f.stat().st_size for f in DEST.rglob("*.json"))
    print(f"Exported {len(instances)} problems. Verified {checks:,} controller updates and all endpoints.")
    print(f"Overview: {(DEST / 'overview.json').stat().st_size / 1024:.0f} KB; total: {size / 1024**2:.1f} MB. Replays load on demand.")


if __name__ == "__main__":
    main()
