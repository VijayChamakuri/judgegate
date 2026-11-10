"""Build the GitHub Pages site (site/index.html) from results/*.json.

Every number on the page is read from a results file at build time; nothing is typed by hand.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from .charts import POLICY_STYLE
from .data import RESULTS, ROOT
from .readme import JUDGE_LABEL, METHOD_LABEL, SETTING_LABEL, pct

SITE = ROOT / "site"
REPO_URL = "https://github.com/VijayChamakuri/judgegate"
CHARTS = ("headline.png", "coverage.png", "calibration.png")


def _rows(cells: list[list[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells)


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
                f"{q['auc']:.2f} <span class=ci>({q['auc_ci'][0]:.2f}–{q['auc_ci'][1]:.2f})</span>",
                f"{q['kappa_at_0_5']:.2f}",
                pct(q["false_pass_rate"]),
                pct(q["false_fail_rate"]),
                f"{q['ece']:.3f}",
            ]
        )
    return _rows(rows)


def rollout_table(res: dict, budget: float) -> str:
    rows = []
    for r in res["rollouts"]:
        if r["budget"] != budget:
            continue
        colour, _, label = POLICY_STYLE[r["policy"]]
        rows.append(
            [
                JUDGE_LABEL[r["judge"]],
                SETTING_LABEL[r["setting"]],
                f"<span class=dot style='background:{colour}'></span>{label}",
                str(r["audits_per_month"]),
                f"{r['human_labels']:,.0f}",
                f"{pct(r['automation_rate'])} <span class=ci>± {pct(r['automation_rate_sd'])}</span>",
                pct(r["auto_error_rate"], 2) if r["automation_rate"] > 0 else "–",
                f"{r['months_over_budget']:.1f}/{r['live_months']}",
                f"£{r['misposted_gbp']:,.0f}",
            ]
        )
    return _rows(rows)


def coverage_table(cov: dict) -> str:
    rows = []
    for g in cov["real_data"]:
        for m, lab in METHOD_LABEL.items():
            v = g[m]
            flag = "" if v["coverage"] >= 0.93 else " <b class=bad>under-covers</b>"
            rows.append(
                [
                    JUDGE_LABEL[g["judge"]],
                    SETTING_LABEL[g["setting"]],
                    f"top {pct(1 - g['confidence_quantile'], 0)}",
                    pct(g["true_error"], 2),
                    str(g["audits"]),
                    lab,
                    pct(v["coverage"]) + flag,
                    pct(v["mean_upper"], 2),
                ]
            )
    return _rows(rows)


def build() -> Path:
    res = json.loads((RESULTS / "judgegate.json").read_text())
    cov = json.loads((RESULTS / "coverage.json").read_text())
    SITE.mkdir(exist_ok=True)
    for c in CHARTS:
        if (RESULTS / c).exists():
            shutil.copy(RESULTS / c, SITE / c)
    from .readme import values

    v = values()
    mistakes = json.dumps(res["judge_mistakes"]).replace("</", "<\\/")
    page = TEMPLATE
    for k, val in {
        "judge_rows": judge_table(res),
        "rollout_rows": rollout_table(res, 0.05),
        "coverage_rows": coverage_table(cov),
        "mistakes": mistakes,
        "repo": REPO_URL,
        **{k: html.escape(x) for k, x in v.items()},
    }.items():
        page = page.replace("{{" + k + "}}", val)
    out = SITE / "index.html"
    out.write_text(page)
    (SITE / ".nojekyll").write_text("")
    return out


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JudgeGate</title>
<meta name="description" content="Can an LLM judge replace the human audits that grant an AI agent autonomy? Tested on real council card data with local models.">
<style>
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;--line:#e4e3df;--accent:#1baf7a;--bad:#c4302f;--good:#0f7a52}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#3a3a37;--accent:#199e70;--bad:#e66767;--good:#4cc79a}}
:root[data-theme=dark]{--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a998f;--line:#3a3a37;--accent:#199e70;--bad:#e66767;--good:#4cc79a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}h1{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:20px;margin:40px 0 8px}p{color:var(--ink2);max-width:760px}a{color:var(--accent)}
.lede{font-size:17px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:24px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.tile b{display:block;font-size:26px;letter-spacing:-.02em}
.tile span{color:var(--ink2);font-size:13px}.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--panel)}tr:last-child td{border-bottom:0}
.ci{color:var(--muted);font-size:12px}.bad{color:var(--bad)}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
figure{margin:16px 0;background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{color:#52514e;font-size:13px;padding:6px 4px 0}.controls{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
button,select{font:inherit;font-size:13px;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--ink);cursor:pointer}
button[aria-pressed=true]{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}footer{margin-top:48px;color:var(--muted);font-size:13px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media (max-width:760px){.two{grid-template-columns:1fr}h1{font-size:24px}}
</style></head><body><main>
<h1>JudgeGate</h1>
<p class=lede>Can an LLM judge replace the human audits that decide when an AI agent may act on its own? Tested on
{{lines}} real purchase-card bookings that council finance staff coded by hand, with free local models (no API).
Short answer: small local judges rank well but are badly calibrated, the textbook statistical fix gives over-optimistic bounds
when errors are rare, and human audits stay the evidence. The judge earns a smaller job: ordering the review queue.</p>
<p><a href="{{repo}}">Code and full method on GitHub</a> · every number below is computed from committed model outputs by <code>judgegate analyze</code>.</p>
<div class=tiles>
<div class=tile><b>{{judge_auc}}</b><span>AUC: the {{headline_judge}} judge ranks wrong bookings above right ones</span></div>
<div class=tile><b>{{judge_false_fail}}</b><span>of correct bookings it rejects at 0.5, so alone it unlocks no autonomy</span></div>
<div class=tile><b>{{plain_cov_min}}</b><span>worst real-data coverage of plain PPI's "95%" bound when errors are rare</span></div>
<div class=tile><b>{{active_cov_min}}</b><span>worst real-data coverage of active PPI's bound: honest, but it only pays with a well-calibrated judge</span></div>
</div>
<h2>1 · How good are the judges?</h2>
<p>Each judge sees the transaction, the agent's proposed account and similar past bookings, never the staff answer.
"False pass" = a wrong booking the judge approved; "false fail" = a correct booking it rejected.</p>
<div class=wrap><table><thead><tr><th>Judge</th><th>Setting</th><th>Lines</th><th>Agent error</th><th>AUC (95% CI)</th><th>κ at 0.5</th><th>False pass</th><th>False fail</th><th>ECE</th></tr></thead>
<tbody>{{judge_rows}}</tbody></table></div>
<div class=two>
<figure><img src="calibration.png" alt="Calibration curve of each judge: predicted probability of a wrong booking against the observed share wrong"><figcaption>Calibration: points on the dashed line mean the judge's probabilities can be taken at face value.</figcaption></figure>
<figure><img src="coverage.png" alt="Simulation: coverage and tightness of each upper-bound method at three error rates"><figcaption>Simulation: a valid 95% bound must reach the dashed line. Lower right = tighter.</figcaption></figure>
</div>
<h2>2 · Does the judge earn autonomy safely?</h2>
<p>Every month the gate auto-posts lines above a confidence threshold whose 95% upper bound on error, from earlier months only, fits a 5% budget. Policies differ only in the evidence behind that bound.</p>
<figure><img src="headline.png" alt="Left: share of live upstream lines auto-posted per policy and audit level. Right: card-feed months over budget per policy and audit level"><figcaption>Left: autonomy each policy appears to earn. Right: what happens on harder data. Plain PPI is dashed.</figcaption></figure>
<div class=wrap><table><thead><tr><th>Judge</th><th>Setting</th><th>Policy</th><th>Audits / month</th><th>Human labels</th><th>Auto-posted (mean ± sd, 20 draws)</th><th>Error when auto-posted</th><th>Months over budget (mean)</th><th>Mis-posted £</th></tr></thead>
<tbody>{{rollout_rows}}</tbody></table></div>
<h2>3 · Are the error bounds honest on real data?</h2>
<p>On the agent's confident half (where autonomy is decided) the true error rate is known for every line, so we redraw only the audits 1,000 times and count how often each "95% upper bound" really sits above the truth.</p>
<div class=wrap><table><thead><tr><th>Judge</th><th>Setting</th><th>Lines kept</th><th>True error</th><th>Audits</th><th>Method</th><th>Coverage</th><th>Mean bound</th></tr></thead>
<tbody>{{coverage_rows}}</tbody></table></div>
<h2>4 · Judge mistakes, line by line</h2>
<p>The judge's most confident errors. Real lines from the Stockport MBC purchase-card register (Open Government Licence v3.0).</p>
<div class=controls id=judgepick></div>
<div class=controls><button data-kind=false_pass aria-pressed=true>Wrong bookings it passed</button><button data-kind=false_fail aria-pressed=false>Correct bookings it failed</button></div>
<div class=wrap><table><thead><tr><th>Month</th><th>Merchant</th><th>Department</th><th>Service</th><th>£</th><th>Agent proposed</th><th>Staff booked</th><th>Judge P(correct)</th></tr></thead><tbody id=mist></tbody></table></div>
<footer>JudgeGate · MIT licence · data: Stockport Metropolitan Borough Council, Open Government Licence v3.0 · models: Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct (4-bit, MLX), bge-small-en-v1.5.</footer>
</main>
<script>
const M={{mistakes}};let judge=Object.keys(M)[0],kind="false_pass";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pick=document.getElementById("judgepick");
for(const k of Object.keys(M)){const b=document.createElement("button");b.textContent=k.replace("/"," judge · ").replace("_"," ");b.dataset.judge=k;pick.appendChild(b)}
function draw(){document.querySelectorAll("[data-judge]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.judge===judge));
document.querySelectorAll("[data-kind]").forEach(b=>b.setAttribute("aria-pressed",b.dataset.kind===kind));
document.getElementById("mist").innerHTML=M[judge].filter(r=>r.kind===kind).map(r=>`<tr><td>${r.month}</td><td>${esc(r.merchant)}</td><td>${esc(r.department)}</td><td>${esc(r.service)}</td><td>${r.amount.toFixed(2)}</td><td>${esc(r.proposed)}</td><td>${esc(r.booked_by_staff)}</td><td>${r.judge_p_correct.toFixed(3)}</td></tr>`).join("")}
document.addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;if(b.dataset.judge)judge=b.dataset.judge;if(b.dataset.kind)kind=b.dataset.kind;draw()});draw();
</script></body></html>
"""
