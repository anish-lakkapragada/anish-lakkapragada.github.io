# Interactive toy-study results

The article embeds the figures with one Liquid include:

```liquid
{% include swe2-results.html %}
```

- `_includes/swe2-results.html`: semantic markup, explanations, and fallback text.
- `assets/swe-2-extended/results.css`: styles scoped to `.swe-results`; text inherits
  the article's font. Mobile margin and equation-overflow rules affect only this
  article, which is the only page loading this stylesheet.
- `assets/swe-2-extended/results.js`: native controls and responsive SVG charts,
  with no runtime package or chart-library dependency.
- `data/overview.json`: aggregate histories and all paired-seed endpoints.
- `data/replays/`: one file per problem, fetched when needed. Only the selected
  problem's file is loaded; it contains all three seeds and six methods.

To regenerate from the recorded study (Python 3, standard library only):

```sh
python3 tools/export_swe2_data.py
# Or supply the experiment directory explicitly:
python3 tools/export_swe2_data.py /path/to/instance_sweep_derivative_init
```

The exporter validates the aggregate and per-problem endpoints and checks each
exported adaptive penalty update against the sampled controller equation. It
preserves eight significant digits. The overview uses every saved trajectory
point; individual replays retain steps 0, 1, and every 20 updates. There is no
interpolation between trained alpha values and no simulated data.

Relative changes use each problem's exact initial policy, average its three
seeds, and weight the 100 problems equally. The replay's controller equation uses
the original sampled batches and Monte Carlo reference values, while displayed
policy success/cost uses exact post-update evaluation. The color map shows
`(1 - alpha) * relative_success_gain - alpha * relative_cost_reduction`, expressed
in percentage points of relative change, with a fixed ±5 saturation range.

The three figures share alpha, effort, and selected problem. The problem grid
supports arrow keys and has an equivalent native select. Replay playback starts
only on request, pauses when the tab is hidden, and slows for reduced-motion
preferences. Loading failures offer a retry or the exported data.
