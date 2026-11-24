"""Consistent strict JSON, tidy CSV and offline escaped HTML quality reports."""
from __future__ import annotations

import csv
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
import re
import uuid
from zoneinfo import ZoneInfo


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def new_report(source):
    return dict(schema="stage_g.evaluation/1", created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        status="partial", data_source=source, metrics=[], comparisons={}, failures=[], stages=[],
        limitations=["Visual resemblance is not geometric accuracy.", "Unobserved/invalid truth produces unavailable metrics, never fabricated zero errors.",
                     "Depth-derived coverage describes observed surfaces only, not unobserved space or complete object volumes."])


def add_metric(report, group, name, value, *, unit="ratio", reason="", provenance="predicted", scope=""):
    report["metrics"].append(dict(group=group, metric=name, value=value, unit=unit,
        status="measured" if value is not None else "unavailable", reason=reason if value is None else "",
        prediction_provenance=provenance, scope=scope))


def add_comparison(report, group, comparison, *, provenance="predicted", scope=""):
    report["comparisons"][group] = comparison
    for name, value in comparison["values"].items():
        unit = "ratio"
        for marker, candidate in [("deg", "deg"), ("m2", "m2"), ("m", "m"), ("s", "s"), ("ppm", "ppm")]:
            if re.search(r"_"+marker+r"(?:_|$)", name):
                unit = candidate
                break
        if any(s in name for s in ("count", "frames", "positives", "negatives", "switches", "fragmentations")):
            unit = "count"
        add_metric(report, group, name, value, unit=unit, reason="No valid samples/denominator for this metric", provenance=provenance, scope=scope)


def attach_stages(report, stages):
    report["stages"] = stages
    report["metrics"] = [row for row in report["metrics"] if not row["group"].startswith("runtime.")]
    for stage in stages:
        group = "runtime."+stage["id"]
        add_metric(report, group, "current_run_wall_seconds", stage.get("current_run_wall_seconds", 0.), unit="s", scope=stage["status"])
        for attempt in stage.get("attempts", []):
            prefix = f"attempt_{attempt['number']}_"
            resource = attempt.get("resources", {})
            for key, unit in [("wall_seconds", "s"), ("cpu_seconds", "s"), ("process_tree_rss_peak_bytes", "bytes"), ("cpu_peak_percent_one_core", "%")]:
                add_metric(report, group, prefix+key, resource.get(key), unit=unit, reason="Resource sample unavailable", scope=attempt["status"])
            for key, unit in [("memory_peak_mb", "MiB"), ("utilization_peak_percent", "%")]:
                add_metric(report, group, prefix+"gpu_device_"+key, resource.get("gpu", {}).get(key), unit=unit, reason=resource.get("gpu", {}).get("reason") or "No GPU samples", scope="device-wide, not attributable to this stage")


def _chart(rows):
    values = [r["translation_m"] for r in rows if "translation_m" in r]
    if not values:
        return ""
    maximum = max(max(values), 1e-12)
    coordinates = " ".join(f"{30+i*740/max(len(values)-1,1):.2f},{150-120*v/maximum:.2f}" for i, v in enumerate(values))
    return f'<h2>Camera translation error (m)</h2><svg role="img" aria-label="Camera translation error over matched frames" viewBox="0 0 800 180"><path d="M30 20V150H780" fill="none" stroke="#888"/><polyline points="{coordinates}" fill="none" stroke="#326dcc" stroke-width="2"/><text x="32" y="17">max {maximum:.5g} m; matched-frame order</text></svg>'


def write_reports(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"evaluation_report.json", report)
    fields = ["group", "metric", "value", "unit", "status", "reason", "prediction_provenance", "scope"]
    with (output/"metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(report["metrics"])
    e = lambda value: escape(str(value), quote=True)
    rows = "".join("<tr>"+"".join(f"<td>{e(row.get(k, '')) if row.get(k) is not None else 'N/A'}</td>" for k in fields)+"</tr>" for row in report["metrics"])
    stage_rows = "".join(f"<tr><td>{e(s['id'])}</td><td>{e(s['status'])}</td><td>{e(s.get('current_run_wall_seconds', 0))}</td><td>{e(s.get('error', ''))}</td></tr>" for s in report.get("stages", []))
    camera_rows = report.get("comparisons", {}).get("camera", {}).get("details", {}).get("per_frame", [])
    artifact_links = []
    for name, target in report.get("artifacts", {}).items():
        target = Path(target).resolve()
        if target.is_relative_to(output.resolve()):
            artifact_links.append(f'<li><a href="{e(target.relative_to(output.resolve()).as_posix())}">{e(name)}</a></li>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Stage G Quality Evaluation</title><style>body{font:15px system-ui;margin:30px;color:#202b3a;background:#f6f8fb}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}td,th{border:1px solid #d7dee8;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#e8eef6}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-width:900px;width:100%;background:white}h1{font-size:26px}</style>'
        f'<h1>Stage G Quality Evaluation — {e(report["status"])}</h1><p>{e(report["data_source"])}</p><p>{e(report["created_at"])}</p>'
        '<p>Ground Truth / Predicted / Inferred / Unobserved are distinct. N/A is not a zero error. Visual similarity does not certify geometry.</p>'
        '<p><a href="evaluation_report.json">JSON report</a> · <a href="metrics.csv">Metrics CSV</a></p>'
        +('<h2>Artifacts</h2><ul>'+"".join(artifact_links)+'</ul>' if artifact_links else '')
        +'<h2>Metrics</h2><table><thead><tr>'+"".join(f"<th>{e(k)}</th>" for k in fields)+f'</tr></thead><tbody>{rows}</tbody></table>'
        +_chart(camera_rows)+'<h2>Stages / cache / retries</h2><table><tr><th>Stage</th><th>Status</th><th>Current run seconds</th><th>Error</th></tr>'+stage_rows+'</table>'
        +'<h2>Failures and limitations</h2><pre>'+e(json.dumps(dict(failures=report["failures"], limitations=report["limitations"]), ensure_ascii=False, indent=2))+'</pre>'
        +'<h2>Evidence and resource details</h2><details><summary>Complete machine report</summary><pre>'+e(json.dumps(report, ensure_ascii=False, indent=2))+'</pre></details></html>')
    (output/"evaluation_report.html").write_text(html, encoding="utf-8")

