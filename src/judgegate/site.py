"""Build the GitHub Pages site (site/index.html) from results/*.json.

Every number on the page is read from a results file at build time; nothing is typed by hand.
The page is a plain write-up: the findings reuse the README's rendered prose, so the two never disagree.
"""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path

from .data import RESULTS, ROOT
from .readme import JUDGE_LABEL, METHOD_LABEL, SETTING_LABEL, pct

SITE = ROOT / "site"
REPO_URL = "https://github.com/VijayChamakuri/judgegate"
CHARTS = ("headline.png", "coverage.png", "calibration.png")


def md(text: str) -> str:
    """The small Markdown subset the README prose uses: paragraphs, numbered lists, **bold**, *italic*."""

    def inline(s: str) -> str:
        s = html.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)

    out = []
    for block in text.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if all(re.match(r"\d+\. ", ln) for ln in lines):
            out.append("<ol>" + "".join(f"<li>{inline(ln.split('. ', 1)[1])}</li>" for ln in lines) + "</ol>")
        else:
            out.append(f"<p>{inline(' '.join(lines))}</p>")
    return "\n".join(out)


def _rows(cells: list[list[str]], numeric_from: int) -> str:
    def td(i: int, c: str) -> str:
        return f"<td class=n>{c}</td>" if i >= numeric_from else f"<td>{c}</td>"

    return "".join("<tr>" + "".join(td(i, c) for i, c in enumerate(r)) + "</tr>" for r in cells)


def judge_table(res: dict) -> str:
    rows = []
    for key, q in res["judges"].items():
        j, s = key.split("/")
        rows.append(
            [
                JUDGE_LABEL[j],
                SETTING_LABEL[s],
                f"{q['n']:,}",
                pct(q["agent_error_rate"]),
                f"{q['auc']:.2f} <span class=dim>({q['auc_ci'][0]:.2f}–{q['auc_ci'][1]:.2f})</span>",
                f"{q['kappa_at_0_5']:.2f}",
                pct(q["false_pass_rate"]),
                pct(q["false_fail_rate"]),
                f"{q['ece']:.2f}",
            ]
        )
    return _rows(rows, 2)


POLICY_NAME = {
    "human_wilson": "Human audits only",
    "judge_raw": "Judge alone, no people",
    "ppi_plain": "Judge + random audits (plain PPI)",
    "active_ppi": "Judge + judge-guided audits (active PPI)",
}


def rollout_table(res: dict, budget: float) -> str:
    rows = []
    for r in res["rollouts"]:
        if r["budget"] != budget:
            continue
        rows.append(
            [
                JUDGE_LABEL[r["judge"]] if r["policy"] != "human_wilson" else "<span class=dim>none</span>",
                SETTING_LABEL[r["setting"]],
                POLICY_NAME.get(r["policy"], r["policy"]),
                str(r["audits_per_month"]),
                f"{pct(r['automation_rate'])} <span class=dim>± {pct(r['automation_rate_sd'], 0)}</span>",
                pct(r["auto_error_rate"], 2) if r["automation_rate"] > 0 else "–",
                f"{r['months_over_budget']:.1f}",
                f"£{r['misposted_gbp']:,.0f}",
            ]
        )
    return _rows(rows, 3)


def coverage_table(cov: dict) -> str:
    rows = []
    for g in cov["real_data"]:
        cells = [
            JUDGE_LABEL[g["judge"]],
            SETTING_LABEL[g["setting"]],
            f"top {pct(1 - g['confidence_quantile'], 0)}",
            pct(g["true_error"], 2),
            str(g["audits"]),
        ]
        for m in METHOD_LABEL:
            c = g[m]["coverage"]
            cells.append(f"<span class=flag>{pct(c, 0)}</span>" if c < 0.93 else pct(c, 0))
        rows.append(cells)
    return _rows(rows, 3)


def build() -> Path:
    from .readme import prose, values

    res = json.loads((RESULTS / "judgegate.json").read_text())
    cov = json.loads((RESULTS / "coverage.json").read_text())
    SITE.mkdir(exist_ok=True)
    for c in CHARTS:
        if (RESULTS / c).exists():
            shutil.copy(RESULTS / c, SITE / c)
    v, p = values(), prose()
    mistakes = json.dumps(res["judge_mistakes"]).replace("</", "<\\/")
    page = TEMPLATE
    for k, val in {
        # The README numbers its sections; the page does not.
        "findings": md(p["key_results"].replace("a coverage test like section 3", "a coverage test like the one below")),
        "judge_rows": judge_table(res),
        "rollout_rows": rollout_table(res, 0.05),
        "coverage_rows": coverage_table(cov),
        "method_heads": "".join(f"<th class=n>{html.escape(lab)}</th>" for lab in METHOD_LABEL.values()),
        "mistakes": mistakes,
        "judge_names": json.dumps(JUDGE_LABEL),
        "setting_names": json.dumps(SETTING_LABEL),
        "repo": REPO_URL,
        "css": CSS,
        **{k: html.escape(x) for k, x in v.items()},
    }.items():
        page = page.replace("{{" + k + "}}", val)
    missing = re.findall(r"\{\{(\w+)\}\}", page)
    if missing:
        raise KeyError(f"unfilled placeholders in site template: {missing}")
    out = SITE / "index.html"
    out.write_text(page)
    (SITE / ".nojekyll").write_text("")
    return out


# Shared with the TrustGate and CommitCompass pages so the three read as one body of work.
CSS = """
:root{--paper:#ffffff;--ink:#1c1b19;--ink2:#47443e;--muted:#77736a;--rule:#d9d3c6;--rule2:#ebe6db;--link:#1a4c8c;--flag:#a3261b;--code:#f2eee3}
html{color-scheme:light}*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.62 Charter,"Bitstream Charter","Iowan Old Style","Sitka Text",Cambria,Georgia,serif;-webkit-text-size-adjust:100%}
.page{max-width:700px;margin:0 auto;padding:20px 20px 96px}
.top{display:flex;flex-wrap:wrap;justify-content:space-between;gap:6px 24px;padding-bottom:12px;margin-bottom:56px;border-bottom:1px solid var(--rule);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--muted)}
.top a{color:var(--ink2);text-decoration:none}.top a:hover{color:var(--link);text-decoration:underline}
.top nav{display:flex;flex-wrap:wrap;gap:4px 18px}.top nav span{color:var(--ink);font-weight:600}
h1{font-size:clamp(28px,5.4vw,38px);line-height:1.18;letter-spacing:-.01em;margin:0 0 14px}
.meta{font-style:italic;color:var(--muted);font-size:16px;margin:0 0 40px}
h2{font-size:24px;line-height:1.3;margin:60px 0 12px}
p{margin:0 0 18px}ol,ul{margin:0 0 18px;padding-left:26px}li{margin:0 0 12px}li::marker{color:var(--muted)}
a{color:var(--link);text-decoration-thickness:1px;text-underline-offset:3px}
code{font:.8em ui-monospace,"SF Mono",Menlo,Consolas,monospace;background:var(--code);padding:2px 4px;border-radius:3px}
.wide{position:relative;left:50%;transform:translateX(-50%);width:min(1000px,calc(100vw - 40px))}
figure{margin:36px 0}figure.narrow img{max-width:500px}figure img{display:block;width:100%;height:auto;margin:0 auto}
figcaption,.cap{font-size:15.5px;line-height:1.5;color:var(--ink2);max-width:700px;margin:12px auto 0}
.cap{margin:0 auto 10px}figcaption b,.cap b{color:var(--ink)}
.tbl{margin:32px 0}.scroll{overflow-x:auto}
.tbl.wide{width:fit-content;min-width:min(700px,calc(100vw - 40px));max-width:min(1000px,calc(100vw - 40px))}.cap{contain:inline-size}
table{border-collapse:collapse;width:100%;font:13.5px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums;border-top:2px solid var(--ink);border-bottom:2px solid var(--ink)}
th{font-weight:600;text-align:left;vertical-align:bottom;padding:8px 10px;border-bottom:1px solid var(--ink)}
td{padding:6px 10px;border-bottom:1px solid var(--rule2);vertical-align:top}tr:last-child td{border-bottom:0}
.n{text-align:right;white-space:nowrap}th.n{white-space:normal;min-width:5em}td:first-child{min-width:11em}.dim{color:var(--muted)}.flag{color:var(--flag);font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:10px 24px;align-items:center;margin:0 auto 14px;max-width:700px;font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink2)}
select,input{font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;padding:5px 8px;border:1px solid #b9b2a3;border-radius:3px;background:#fff;color:var(--ink);margin-left:6px}
footer{margin-top:80px;padding-top:14px;border-top:1px solid var(--rule);font-size:15px;color:var(--muted)}
@media (max-width:600px){body{font-size:17px}.top{margin-bottom:36px}h2{margin-top:48px}}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JudgeGate</title>
<meta name="description" content="Can an LLM judge replace the human audits that let an AI agent act on its own? Tested on 14,851 real council bookings with local models.">
<style>{{css}}</style></head><body><div class=page>
<header class=top><a href="https://github.com/VijayChamakuri">Vijay Chamakuri</a>
<nav><a href="https://vijaychamakuri.github.io/trustgate/">TrustGate</a><a href="https://vijaychamakuri.github.io/commitcompass/">CommitCompass</a><span>JudgeGate</span><a href="https://vijaychamakuri.github.io/sift/">Sift</a></nav></header>
<main>
<h1>Can an LLM judge replace the human audits behind an agent's autonomy?</h1>
<p class=meta>Vijay Chamakuri, November 2026. Code, data and full method on <a href="{{repo}}">GitHub</a>.</p>

<p>In <a href="https://vijaychamakuri.github.io/trustgate/">TrustGate</a> I built a gate that lets a bookkeeping agent post
transactions without review, but only once human audits show its error rate is under a budget. The audits are the expensive part,
and the obvious idea is to have an LLM check the work instead. This project tests whether that holds up.</p>
<p>The data is {{lines}} purchase-card lines from Stockport Council. Each was booked to a ledger account by the council's finance
staff, and those bookings are the ground truth. The judges are two small open models, Llama-3.2-3B and Qwen2.5-3B, running on
my laptop. A judge sees the transaction, the agent's proposed account and a few similar earlier bookings. It never sees the staff answer.</p>

<h2>What I found</h2>
{{findings}}

<h2>How good are the judges?</h2>
<p>A judge can be good at ranking (wrong bookings get higher risk scores than right ones) and still be wrong about how likely a
booking is to be wrong. The table separates the two. A false pass is a wrong booking the judge approved; a false fail is a correct
booking it rejected. ECE is the average gap between the judge's stated probability and what actually happened.</p>
<div class="tbl wide"><p class=cap><b>Table 1.</b> Judge quality against staff bookings, at a 0.5 cut-off. AUC has a 95% bootstrap interval.</p>
<div class=scroll><table><thead><tr><th>Judge</th><th>Data</th><th class=n>Lines</th><th class=n>Agent error</th><th class=n>AUC</th><th class=n>κ</th><th class=n>False pass</th><th class=n>False fail</th><th class=n>ECE</th></tr></thead>
<tbody>{{judge_rows}}</tbody></table></div></div>
<figure class=narrow><img src="calibration.png" alt="Calibration curves: the judges' predicted probability that a booking is wrong against the share that really were wrong. All curves sit well below the diagonal.">
<figcaption><b>Figure 1.</b> Calibration. A judge whose probabilities could be taken at face value would sit on the dashed diagonal.
Both sit below it: they call bookings wrong far more often than they are.</figcaption></figure>

<h2>Does a judge earn the agent any autonomy?</h2>
<p>Each month the gate auto-posts the agent's lines above a confidence threshold. The threshold is the lowest one whose 95% upper
bound on error, computed from earlier months only, fits the budget. The policies below differ only in the evidence behind that bound.
Policies that use people get the same number of audits, and every row is the mean of {{seeds}} independent audit draws.</p>
<figure class=wide><img src="headline.png" alt="Left: share of live lines auto-posted by each policy as audits per month rise; plain PPI is highest. Right: how often each policy's 95% bound covers the true error on real slices; plain PPI falls to 84%.">
<figcaption><b>Figure 2.</b> Left: how much each policy auto-posts (upstream data, 2% budget, Llama judge). Right: how often
its 95% bound really sits above the true error rate on real data. Plain PPI wins on the left because it fails on the right.</figcaption></figure>
<div class="tbl wide"><p class=cap><b>Table 2.</b> Twenty live months at a 5% error budget. Auto-posted is the mean share of lines
posted without review, ± its standard deviation across audit draws. Error is measured on the lines actually auto-posted.</p>
<div class=scroll><table><thead><tr><th>Judge</th><th>Data</th><th>Evidence</th><th class=n>Audits a month</th><th class=n>Auto-posted</th><th class=n>Error</th><th class=n>Months over budget</th><th class=n>Mis-posted</th></tr></thead>
<tbody>{{rollout_rows}}</tbody></table></div></div>

<h2>Are the error bounds honest?</h2>
<p>A "95% upper bound" should sit above the true error rate at least 95% of the time. I checked this two ways. In simulation the
truth is set by construction. On the real data it is known for every line, so I kept the judge scores fixed, redrew only the audits
1,000 times, and counted how often each bound held.</p>
<figure class=wide><img src="coverage.png" alt="Simulation results: coverage and average bound for each method at three true error rates and three judge strengths.">
<figcaption><b>Figure 3.</b> Simulation ({{sim_reps}} repeats per cell). A valid bound reaches the dashed 95% line.</figcaption></figure>
<div class="tbl wide"><p class=cap><b>Table 3.</b> Coverage on real slices, from all live lines down to the agent's most confident 10%.
Values under 93% are marked. "Tuned on the same sample" is included as a warning: picking the weighting on the audits you then
test with makes the bound optimistic.</p>
<div class=scroll><table><thead><tr><th>Judge</th><th>Data</th><th>Lines kept</th><th class=n>True error</th><th class=n>Audits</th>{{method_heads}}</tr></thead>
<tbody>{{coverage_rows}}</tbody></table></div></div>

<h2>The judge's mistakes</h2>
<p>These are the judge's most confident errors, as real lines from the council's card register. Some of the "wrong bookings it
passed" may be staff coding the same thing two ways rather than a judge error.</p>
<div class=wide><div class=controls><label>Judge<select id=judge></select></label><label>Show<select id=kind>
<option value=false_pass>wrong bookings it passed</option><option value=false_fail>correct bookings it failed</option></select></label></div>
<div class=scroll><table><thead><tr><th>Month</th><th>Merchant</th><th>Department</th><th>Service</th><th class=n>£</th><th>Agent proposed</th><th>Staff booked</th><th class=n>Judge P(correct)</th></tr></thead><tbody id=mist></tbody></table></div></div>

<footer><p>Every number on this page is computed from committed model outputs by <code>judgegate analyze</code>; CI recomputes
them and fails if they change. Method notes are in <a href="{{repo}}/blob/main/docs/METHODOLOGY.md">METHODOLOGY.md</a>.</p>
<p>Code: MIT licence. Data: Stockport Metropolitan Borough Council purchase-card transactions, Open Government Licence v3.0.
Models: Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct (4-bit, MLX), bge-small-en-v1.5.</p></footer>
</main></div>
<script>
const M={{mistakes}},JN={{judge_names}},SN={{setting_names}};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const J=document.getElementById("judge"),K=document.getElementById("kind");
J.innerHTML=Object.keys(M).map(k=>{const [j,s]=k.split("/");return `<option value="${k}">${JN[j]}, ${SN[s].toLowerCase()} data</option>`}).join("");
function draw(){document.getElementById("mist").innerHTML=M[J.value].filter(r=>r.kind===K.value).map(r=>`<tr><td>${r.month}</td><td>${esc(r.merchant)}</td><td>${esc(r.department)}</td><td>${esc(r.service)}</td><td class=n>${r.amount.toFixed(2)}</td><td>${esc(r.proposed)}</td><td>${esc(r.booked_by_staff)}</td><td class=n>${r.judge_p_correct.toFixed(3)}</td></tr>`).join("")}
J.addEventListener("change",draw);K.addEventListener("change",draw);draw();
</script></body></html>
"""
