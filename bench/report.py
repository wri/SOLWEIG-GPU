# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown>=3.5"]
# ///
"""Build static HTML pages from benchmark results, for browsing on S3.

    uv run bench/report.py SRC_DIR OUT_DIR

SRC_DIR holds `results/*.json` (bench.py output) and optionally `findings/<topic>/*.md`.
OUT_DIR gets:

    index.html                        comparison of all runs, links to each run and finding
    runs/<results file stem>.html     one page per run
    findings/<topic>/<name>.html      each finding rendered from Markdown

Pages are self-contained (inline CSS and SVG, no scripts) and use relative links, so they
work from any S3 prefix: index.html links to results/<file>.json next to it.
"""
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import markdown

STAGES = ["wind_coeff", "preprocess", "walls_aspect", "svf", "utci"]

# On-demand Linux prices, us-east-1, USD per hour (checked Oct 2026). Used for the
# compute cost of a run's wall time only: VM start-up and data download are excluded.
PRICES = {
    "g4dn.xlarge": 0.526, "g4dn.2xlarge": 0.752,
    "g5.xlarge": 1.006, "g5.2xlarge": 1.212, "g5.4xlarge": 1.624,
    "g6.xlarge": 0.8048, "g6.2xlarge": 0.9776, "g6.4xlarge": 1.3232,
    "c7i.2xlarge": 0.357, "c7i.4xlarge": 0.714, "c7i.8xlarge": 1.428,
}

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e8; --card:#f6f6f8; --link:#0b5cad;
  --s1:#4e79a7; --s2:#a0cbe8; --s3:#f28e2b; --s4:#59a14f; --s5:#b07aa1; --cpu:#4e79a7; --gpu:#e15759; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececf0; --muted:#9a9aa2; --line:#2e2e33;
  --card:#202024; --link:#7fb3f0; --s1:#6f9bd1; --s2:#8cb8d8; --s3:#f5a65b; --s4:#7cc06f; --s5:#c99bbc;
  --cpu:#6f9bd1; --gpu:#ff7b7d; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1280px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.6rem; margin:0 0 4px; } h2 { font-size:1.2rem; margin:32px 0 8px; }
a { color:var(--link); } .muted { color:var(--muted); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; font-size:0.85rem; font-variant-numeric:tabular-nums; }
th, td { padding:5px 8px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { font-weight:600; background:var(--card); }
td.l, th.l { text-align:left; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font-size:0.85rem; margin:8px 0; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:6px; background:var(--c); }
dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 16px; font-size:0.9rem; }
dt { color:var(--muted); } dd { margin:0; overflow-wrap:anywhere; }
svg text { fill:var(--muted); font-size:11px; }
svg.chart { display:block; width:100%; min-width:640px; height:auto; }
main > * { min-width:0; } p, li { overflow-wrap:anywhere; }
.md { max-width:880px; } .md table { margin:12px 0; display:block; overflow-x:auto; }
.md th, .md td { text-align:left; white-space:normal; vertical-align:top; } .md pre { background:var(--card); padding:12px; overflow-x:auto; }
.md code { font-size:0.9em; }
"""


def page(title, body, depth=0):
    home = "../" * depth + "index.html"
    nav = "" if depth == 0 else f'<p><a href="{home}">&larr; All benchmark runs</a></p>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style></head>'
            f'<body><main>{nav}{body}</main></body></html>')


def esc(v):
    return html.escape("" if v is None else str(v))


def fmt(v, nd=1):
    return "–" if v is None else f"{v:,.{nd}f}"


def stage_map(run):
    return {s["stage"]: s for s in run.get("stages", [])}


def machine_name(m):
    return m.get("ec2_instance_type") or m.get("cpu_model") or "unknown"


def cost(run):
    price = PRICES.get(run["machine"].get("ec2_instance_type") or "")
    return None if price is None or run.get("total_seconds") is None else run["total_seconds"] * price / 3600


def legend():
    items = "".join(f'<span style="--c:var(--s{i + 1})">{s}</span>' for i, s in enumerate(STAGES))
    return f'<div class="legend">{items}</div>'


def stacked_bars(runs):
    """Horizontal stacked bars of seconds per stage, one bar per run."""
    label_w, bar_h, gap, width = 260, 18, 8, 1100
    vmax = max((r.get("total_seconds") or 0) for r in runs) or 1
    scale = (width - label_w - 60) / vmax
    out = []
    for i, r in enumerate(runs):
        y = i * (bar_h + gap)
        sm = stage_map(r)
        out.append(f'<text x="{label_w - 8}" y="{y + 13}" text-anchor="end">{esc(r["_name"])}</text>')
        x = label_w
        for j, st in enumerate(STAGES):
            w = (sm.get(st, {}).get("seconds") or 0) * scale
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="var(--s{j + 1})">'
                       f'<title>{st}: {fmt(sm.get(st, {}).get("seconds"))} s</title></rect>')
            x += w
        out.append(f'<text x="{x + 6:.1f}" y="{y + 13}">{fmt(r.get("total_seconds"), 0)} s</text>')
    h = len(runs) * (bar_h + gap)
    return (f'<div class="scroll"><svg class="chart" viewBox="0 0 {width} {h}" role="img" '
            f'aria-label="Seconds per stage for each run">{"".join(out)}</svg></div>')


def timeseries(run):
    """CPU % and GPU utilisation % over the run, with stage boundaries."""
    samples = run.get("samples") or []
    if len(samples) < 2:
        return "<p class='muted'>No resource samples recorded.</p>"
    width, height, pad_l, pad_b = 1000, 220, 40, 22
    t0, t1 = samples[0]["t"], samples[-1]["t"]
    span = (t1 - t0) or 1
    sx = lambda t: pad_l + (t - t0) / span * (width - pad_l - 10)
    sy = lambda v: 8 + (1 - min(max(v or 0, 0), 100) / 100) * (height - pad_b - 8)
    out = []
    for v in (0, 50, 100):
        out.append(f'<line x1="{pad_l}" x2="{width - 10}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--line)"/>'
                   f'<text x="{pad_l - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}%</text>')
    prev, last_label_x = None, -1e9
    for s in samples:
        if s.get("stage") != prev:
            x = sx(s["t"])
            out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{height - pad_b}" stroke="var(--line)"/>')
            if x - last_label_x > 80:   # skip labels of stages too short to fit one
                out.append(f'<text x="{x + 3:.1f}" y="{height - 6}">{esc(s.get("stage"))}</text>')
                last_label_x = x
            prev = s.get("stage")
    for key, color in (("cpu_pct", "--cpu"), ("gpu_util", "--gpu")):
        pts = [(sx(s["t"]), sy(s.get(key))) for s in samples if s.get(key) is not None]
        if pts:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            out.append(f'<polyline points="{path}" fill="none" stroke="var({color})" stroke-width="1.5"/>')
    leg = ('<div class="legend"><span style="--c:var(--cpu)">CPU (% of all cores)</span>'
           '<span style="--c:var(--gpu)">GPU utilisation</span></div>')
    return (leg + f'<div class="scroll"><svg class="chart" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="CPU and GPU use over time">{"".join(out)}</svg></div>')


def index_page(runs, findings):
    rows = []
    for r in runs:
        m, c, sm = r["machine"], r["config"], stage_map(r)
        cells = "".join(f"<td>{fmt(sm.get(s, {}).get('seconds'))}</td>" for s in STAGES)
        gpu = lambda st: fmt(sm.get(st, {}).get("resources", {}).get("gpu_util_mean"), 0)
        rows.append(
            f'<tr><td class="l"><a href="runs/{esc(Path(r["_file"]).stem)}.html" title="run {esc(r.get("run_id"))}">'
            f'{esc(r.get("label"))}</a></td><td class="l">{esc(r["_date"])}</td><td class="l">{esc(machine_name(m))}</td>'
            f'<td class="l">{esc(m.get("gpu_name") or "–")}</td>'
            f'<td>{esc(c.get("domain_cols"))}×{esc(c.get("domain_rows"))}</td><td>{esc(c.get("n_tiles"))}</td>'
            f'<td>{esc(c.get("n_timesteps"))}</td>{cells}<td><b>{fmt(r.get("total_seconds"))}</b></td>'
            f'<td>{"–" if cost(r) is None else "$" + format(cost(r), ".3f")}</td>'
            f'<td>{gpu("svf")} / {gpu("utci")}</td>'
            f'<td class="l">{"<b>error</b>" if r.get("error") else ""}</td></tr>')
    head = "".join(f"<th>{s}</th>" for s in STAGES)
    table = (f'<div class="scroll"><table><thead><tr><th class="l">Run</th><th class="l">Date</th>'
             f'<th class="l">Machine</th><th class="l">GPU</th><th>Domain (px)</th><th>Tiles</th><th>Hours</th>'
             f'{head}<th>Total (s)</th><th>Compute cost</th><th>GPU % svf / utci</th><th></th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    flist = "".join(f'<li><a href="{esc(href)}">{esc(title)}</a></li>' for href, title in findings)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (f"<h1>SOLWEIG-GPU benchmarks</h1><p class='muted'>{len(runs)} runs · generated {generated}. "
            f"Times are wall-clock seconds per pipeline stage, from <code>bench/bench.py</code>.</p>"
            + (f"<h2>Findings</h2><ul>{flist}</ul>" if findings else "")
            + f"<h2>Time per stage</h2>{legend()}{stacked_bars(runs)}"
            + f"<h2>All runs</h2>{table}"
            + "<p class='muted'>Compute cost = total wall time × on-demand us-east-1 price for the instance "
              "type; it excludes VM start-up and data download. GPU % is the mean utilisation during the "
              "stage. Each run's raw JSON is linked from its page.</p>")
    return page("SOLWEIG-GPU benchmarks", body)


def run_page(r):
    m, c, sm = r["machine"], r["config"], stage_map(r)
    total = r.get("total_seconds") or 0
    rows = []
    for st in STAGES:
        s = sm.get(st)
        if not s:
            continue
        res = s.get("resources", {})
        pct = 100 * (s.get("seconds") or 0) / total if total else None
        rows.append(f'<tr><td class="l">{st}</td><td>{fmt(s.get("seconds"))}</td><td>{fmt(pct, 0)}%</td>'
                    f'<td>{fmt(res.get("cpu_busy_cores_mean"))} / {fmt(res.get("cpu_busy_cores_max"))}</td>'
                    f'<td>{fmt(res.get("rss_mb_max"), 0)}</td>'
                    f'<td>{fmt(res.get("gpu_util_mean"), 0)} / {fmt(res.get("gpu_util_max"), 0)}</td>'
                    f'<td>{fmt(res.get("gpu_mem_mb_max"), 0)}</td><td>{fmt(s.get("torch_peak_gpu_mem_mb"), 0)}</td></tr>')
    table = ('<div class="scroll"><table><thead><tr><th class="l">Stage</th><th>Seconds</th><th>Share</th>'
             '<th>CPU busy cores mean / max</th><th>Peak RSS (MB)</th><th>GPU % mean / max</th>'
             '<th>GPU mem (MB, nvidia-smi)</th><th>Torch peak (MB)</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')

    def details(d):
        items = "".join(f"<dt>{esc(k)}</dt><dd>{esc(json.dumps(v) if isinstance(v, (dict, list)) else v)}</dd>"
                        for k, v in d.items())
        return f"<dl>{items}</dl>"

    dl = r.get("data_download") or {}
    dl_html = (f"<h2>Data download</h2><p>{fmt(dl.get('download_seconds'))} s download, "
               f"{fmt(dl.get('unzip_seconds'))} s unzip (not included in the total).</p>") if dl else ""
    err = f"<h2>Error</h2><pre>{esc(r['error'])}</pre>" if r.get("error") else ""
    c_cost = cost(r)
    body = (f"<h1>{esc(r.get('label'))}</h1>"
            f"<p class='muted'>Run {esc(r.get('run_id'))} · {esc(machine_name(m))}"
            f"{' · ' + esc(m.get('gpu_name')) if m.get('gpu_name') else ''} · total {fmt(total)} s"
            f"{' · compute cost $' + format(c_cost, '.3f') if c_cost is not None else ''} · "
            f"<a href='../results/{esc(r['_file'])}'>raw JSON</a></p>{err}"
            f"<h2>Stages</h2>{table}<h2>CPU and GPU use</h2>{timeseries(r)}"
            f"<h2>Configuration</h2>{details(c)}<h2>Machine</h2>{details(m)}{dl_html}")
    return page(f"{r.get('label')} – SOLWEIG-GPU benchmark", body, depth=1)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = []
    for f in sorted((src / "results").glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            print(f"skipping {f.name}: {e}", file=sys.stderr)
            continue
        r["_file"] = f.name
        rid = str(r.get("run_id") or "")
        r["_date"] = f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}" if len(rid) >= 8 else ""
        r["_name"] = f"{r.get('label')} · {r['config'].get('domain_cols')} px · {r['_date']}"
        r.setdefault("machine", {})
        runs.append(r)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for r in runs:
        (out / "runs" / f"{Path(r['_file']).stem}.html").write_text(run_page(r))

    findings = []
    for md_file in sorted((src / "findings").glob("*/*.md")):
        text = md_file.read_text()
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), md_file.stem)
        rel = md_file.relative_to(src).with_suffix(".html")
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(page(title, f'<div class="md">{body}</div>', depth=len(rel.parts) - 1))
        findings.append((rel.as_posix(), title))

    (out / "index.html").write_text(index_page(runs, findings))
    print(f"wrote {len(runs)} run pages, {len(findings)} findings and index.html to {out}")


if __name__ == "__main__":
    main()
