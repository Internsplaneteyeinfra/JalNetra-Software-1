#!/usr/bin/env bash
# Generates HTML + JSON report from check results
RESULTS_FILE="$1"
PROJECT_NAME="$2"
REPORT_DIR="$3"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HTML_FILE="${REPORT_DIR}/report_${TIMESTAMP}.html"
LATEST_FILE="${REPORT_DIR}/report_latest.html"

[ ! -f "$RESULTS_FILE" ] && echo "No results file found." && exit 1

# Parse JSON results
PASS=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['summary']['pass'])" 2>/dev/null || echo 0)
FAIL=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['summary']['fail'])" 2>/dev/null || echo 0)
WARN=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['summary']['warn'])" 2>/dev/null || echo 0)
SKIP=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['summary']['skip'])" 2>/dev/null || echo 0)
TOTAL=$(python3 -c "import json; d=json.load(open('$RESULTS_FILE')); print(d['summary']['total'])" 2>/dev/null || echo 0)
SCORE=$(python3 -c "print(round($PASS / max($PASS+$FAIL+$WARN,1) * 100))" 2>/dev/null || echo 0)

TIMESTAMP_DISPLAY=$(date)

# Build rule rows from JSON
RULE_ROWS=$(python3 3<<'PYEOF'
import json, sys

RULE_META = {
    "R03": ("Rule 3",  "Clean Code / Formatting",       "Part 1 — Core",    "3"),
    "R04": ("Rule 4",  "DRY & Complexity",               "Part 1 — Core",    "4"),
    "R05": ("Rule 5",  "Security / OWASP SAST",          "Part 1 — Core",    "5"),
    "R06": ("Rule 6",  "Secrets Detection",              "Part 1 — Core",    "6"),
    "R08": ("Rule 8",  "Database / SQL Safety",          "Part 1 — Core",    "8"),
    "R10": ("Rule 10", "API Design",                     "Part 1 — Core",    "10"),
    "R11": ("Rule 11", "Error Handling",                 "Part 1 — Core",    "11"),
    "R12": ("Rule 12", "Logging Safety",                 "Part 1 — Core",    "12"),
    "R13": ("Rule 13", "Test Coverage",                  "Part 1 — Core",    "13"),
    "R15": ("Rule 15", "Git Workflow",                   "Part 1 — Core",    "15"),
    "R16": ("Rule 16", "Dependency Security",            "Part 1 — Core",    "16"),
    "R18": ("Rule 18", "Config Separation",              "Part 1 — Core",    "18"),
    "R20": ("Rule 20", "AI Hallucination / Dep Verify",  "Part 2 — AI",      "20"),
    "R21": ("Rule 21", "SAST for AI Code",               "Part 2 — AI",      "21"),
    "R25": ("Rule 25", "AI Attribution in PRs",          "Part 2 — AI",      "25"),
}

STATUS_LABELS = {
    "PASS": ('<span class="badge pass">✔ PASS</span>', "pass"),
    "FAIL": ('<span class="badge fail">✘ FAIL</span>', "fail"),
    "WARN": ('<span class="badge warn">⚠ WARN</span>', "warn"),
    "SKIP": ('<span class="badge skip">– SKIP</span>', "skip"),
}

import os
results_file = os.environ.get('RESULTS_FILE', sys.argv[1] if len(sys.argv) > 1 else '')

try:
    data = json.load(open(results_file))
    rules = data.get("rules", {})
except:
    rules = {}

rows = []
for rid, meta in sorted(RULE_META.items()):
    label, name, part, num = meta
    rule_data = rules.get(rid, {"status": "SKIP", "message": "Not run"})
    status = rule_data.get("status", "SKIP")
    message = rule_data.get("message", "")
    badge, css = STATUS_LABELS.get(status, STATUS_LABELS["SKIP"])
    rows.append(f'<tr class="row-{css}"><td>{label}</td><td>{name}</td><td>{part}</td><td>{badge}</td><td>{message}</td></tr>')

print("\n".join(rows))
PYEOF
RESULTS_FILE="$RESULTS_FILE" python3 - "$RESULTS_FILE" <<'PYEOF'
import json, sys, os

RULE_META = {
    "R03": ("Rule 3",  "Clean Code / Formatting",       "Part 1 — Core",    "3"),
    "R04": ("Rule 4",  "DRY & Complexity",               "Part 1 — Core",    "4"),
    "R05": ("Rule 5",  "Security / OWASP SAST",          "Part 1 — Core",    "5"),
    "R06": ("Rule 6",  "Secrets Detection",              "Part 1 — Core",    "6"),
    "R08": ("Rule 8",  "Database / SQL Safety",          "Part 1 — Core",    "8"),
    "R10": ("Rule 10", "API Design",                     "Part 1 — Core",    "10"),
    "R11": ("Rule 11", "Error Handling",                 "Part 1 — Core",    "11"),
    "R12": ("Rule 12", "Logging Safety",                 "Part 1 — Core",    "12"),
    "R13": ("Rule 13", "Test Coverage",                  "Part 1 — Core",    "13"),
    "R15": ("Rule 15", "Git Workflow",                   "Part 1 — Core",    "15"),
    "R16": ("Rule 16", "Dependency Security",            "Part 1 — Core",    "16"),
    "R18": ("Rule 18", "Config Separation",              "Part 1 — Core",    "18"),
    "R20": ("Rule 20", "AI Hallucination / Dep Verify",  "Part 2 — AI",      "20"),
    "R21": ("Rule 21", "SAST for AI Code",               "Part 2 — AI",      "21"),
    "R25": ("Rule 25", "AI Attribution in PRs",          "Part 2 — AI",      "25"),
}
STATUS_LABELS = {
    "PASS": ('<span class="badge pass">✔ PASS</span>', "pass"),
    "FAIL": ('<span class="badge fail">✘ FAIL</span>', "fail"),
    "WARN": ('<span class="badge warn">⚠ WARN</span>', "warn"),
    "SKIP": ('<span class="badge skip">– SKIP</span>', "skip"),
}
results_file = sys.argv[1]
try:
    data = json.load(open(results_file))
    rules = data.get("rules", {})
except:
    rules = {}
for rid, meta in sorted(RULE_META.items()):
    label, name, part, num = meta
    rule_data = rules.get(rid, {"status": "SKIP", "message": "Not run"})
    status = rule_data.get("status", "SKIP")
    message = rule_data.get("message", "")
    badge, css = STATUS_LABELS.get(status, STATUS_LABELS["SKIP"])
    print(f'<tr class="row-{css}"><td>{label}</td><td>{name}</td><td>{part}</td><td>{badge}</td><td>{message}</td></tr>')
PYEOF
)

# Determine score color
if [ "$SCORE" -ge 80 ]; then SCORE_COLOR="#22c55e"
elif [ "$SCORE" -ge 60 ]; then SCORE_COLOR="#f59e0b"
else SCORE_COLOR="#ef4444"; fi

cat > "$HTML_FILE" <<HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy Report — ${PROJECT_NAME}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
            padding: 40px 48px; border-bottom: 1px solid #334155; }
  .header h1 { font-size: 28px; font-weight: 700; color: #f8fafc; }
  .header .sub { font-size: 14px; color: #94a3b8; margin-top: 6px; }
  .header .meta { font-size: 13px; color: #64748b; margin-top: 4px; }
  .container { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }

  /* Score card */
  .score-section { display: flex; gap: 24px; margin-bottom: 40px; flex-wrap: wrap; }
  .score-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
                padding: 28px 36px; text-align: center; min-width: 160px; }
  .score-number { font-size: 48px; font-weight: 800; color: ${SCORE_COLOR}; }
  .score-label { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  .stat-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px;
               padding: 20px 28px; text-align: center; min-width: 120px; flex: 1; }
  .stat-number { font-size: 36px; font-weight: 700; }
  .stat-label { font-size: 12px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .pass-num { color: #22c55e; }
  .fail-num { color: #ef4444; }
  .warn-num { color: #f59e0b; }
  .skip-num { color: #64748b; }

  /* Table */
  .section-title { font-size: 18px; font-weight: 600; color: #f1f5f9;
                   margin-bottom: 16px; padding-bottom: 8px;
                   border-bottom: 1px solid #334155; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 40px;
          background: #1e293b; border-radius: 12px; overflow: hidden;
          border: 1px solid #334155; }
  th { background: #0f172a; padding: 12px 16px; text-align: left;
       font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
       color: #94a3b8; font-weight: 600; }
  td { padding: 13px 16px; font-size: 14px; border-top: 1px solid #1e293b; }
  tr.row-pass { background: #1e293b; }
  tr.row-fail { background: #1e1a1a; }
  tr.row-warn { background: #1e1c14; }
  tr.row-skip { background: #161e2e; }
  tr:hover { background: #263148 !important; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 12px;
           font-weight: 600; white-space: nowrap; }
  .badge.pass { background: #14532d; color: #4ade80; }
  .badge.fail { background: #450a0a; color: #f87171; }
  .badge.warn { background: #422006; color: #fbbf24; }
  .badge.skip { background: #1e293b; color: #64748b; }

  /* Non-automated rules */
  .manual-section { background: #1e293b; border: 1px solid #334155;
                    border-radius: 12px; padding: 24px 28px; margin-bottom: 40px; }
  .manual-section h3 { font-size: 15px; font-weight: 600; color: #f1f5f9;
                       margin-bottom: 12px; }
  .manual-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                 gap: 10px; }
  .manual-item { background: #0f172a; border: 1px solid #334155; border-radius: 8px;
                 padding: 10px 14px; font-size: 13px; color: #94a3b8; }
  .manual-item strong { color: #e2e8f0; display: block; margin-bottom: 2px; }

  /* Footer */
  .footer { text-align: center; font-size: 12px; color: #475569;
            padding: 24px; border-top: 1px solid #1e293b; margin-top: 20px; }
</style>
</head>
<body>
<div class="header">
  <h1>Software Development Policy Report</h1>
  <div class="sub">Project: <strong>${PROJECT_NAME}</strong></div>
  <div class="meta">Generated: ${TIMESTAMP_DISPLAY} &nbsp;|&nbsp; Policy Version 1.2 — 25 Rules</div>
</div>

<div class="container">

  <div class="score-section">
    <div class="score-card">
      <div class="score-number">${SCORE}%</div>
      <div class="score-label">Compliance Score</div>
    </div>
    <div class="stat-card"><div class="stat-number pass-num">${PASS}</div><div class="stat-label">Pass</div></div>
    <div class="stat-card"><div class="stat-number fail-num">${FAIL}</div><div class="stat-label">Fail</div></div>
    <div class="stat-card"><div class="stat-number warn-num">${WARN}</div><div class="stat-label">Warn</div></div>
    <div class="stat-card"><div class="stat-number skip-num">${SKIP}</div><div class="stat-label">Skip</div></div>
    <div class="stat-card"><div class="stat-number" style="color:#94a3b8">${TOTAL}</div><div class="stat-label">Total Checks</div></div>
  </div>

  <div class="section-title">Automated Rule Results (15 of 25 Rules)</div>
  <table>
    <thead>
      <tr>
        <th>Rule</th>
        <th>Check</th>
        <th>Policy Part</th>
        <th>Status</th>
        <th>Detail</th>
      </tr>
    </thead>
    <tbody>
      ${RULE_ROWS}
    </tbody>
  </table>

  <div class="manual-section">
    <h3>Rules Requiring Human Review (cannot be automated)</h3>
    <div class="manual-grid">
      <div class="manual-item"><strong>Rule 1</strong>Requirements defined before coding</div>
      <div class="manual-item"><strong>Rule 2</strong>Architecture layering followed</div>
      <div class="manual-item"><strong>Rule 7</strong>Auth enforced server-side</div>
      <div class="manual-item"><strong>Rule 9</strong>Performance & N+1 review</div>
      <div class="manual-item"><strong>Rule 14</strong>Code review completed</div>
      <div class="manual-item"><strong>Rule 17</strong>Documentation maintained</div>
      <div class="manual-item"><strong>Rule 19</strong>AI code read & understood</div>
      <div class="manual-item"><strong>Rule 22</strong>No secrets in AI prompts</div>
      <div class="manual-item"><strong>Rule 23</strong>Architecture not bypassed by AI</div>
      <div class="manual-item"><strong>Rule 24</strong>AI-generated tests verified</div>
    </div>
  </div>

</div>
<div class="footer">
  Software Development Policy v1.2 — Internal Use Only &nbsp;|&nbsp; ${PROJECT_NAME} &nbsp;|&nbsp; ${TIMESTAMP_DISPLAY}
</div>
</body>
</html>
HTMLEOF

cp "$HTML_FILE" "$LATEST_FILE"
echo "  HTML report: $HTML_FILE"
echo "  Latest:      $LATEST_FILE"
