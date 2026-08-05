#!/usr/bin/env bash
# One-command frozen first real-data study (macOS/Linux).
#
#   ./scripts/run_first_real_study.sh [--start 1998-01-01] \
#       [--providers "yahoo stooq fred sec"] [--skip-download]
#
# Phases: environment validation -> freeze verification -> download (resumable)
# -> checksum-verified import -> quality gate (hard stop on FAIL) -> frozen
# real-mode backtests -> robustness -> capacity -> company analysis -> reports
# (full + decision brief) -> shareable results bundle.
#
# --skip-download reuses the store already in data/import and runs everything
# from the quality gate onward. It is for re-running the study under a NEW
# FREEZE that needs no new data — the common case, since a freeze bump usually
# adds strategies rather than inputs. It refuses if the store is empty rather
# than backtesting nothing, and it does NOT skip the quality gate: the gate
# validates the store, and a store that was fine for the last freeze is not
# automatically fine for this one. Free providers serve 404s for many delisted
# tickers and the retries are slow, so re-downloading unchanged data is the
# single most expensive way to change nothing.
#
# Guarantees: never substitutes synthetic data; never tunes parameters (the
# freeze hash is verified before any backtest); API keys are redacted from the
# log; provider responses and manifests are preserved in data/export_bundle.

set -euo pipefail
cd "$(dirname "$0")/.."

START="1998-01-01"
# Provider hierarchy (§frozen study): SEC EDGAR for PIT fundamentals, FRED for
# macro (revised, labeled as such), Tiingo preferred for prices WHEN a key is
# present, Yahoo+Stooq for reconciliation/fallback. Alpha Vantage is omitted
# by default (free-tier limits); add it via --providers if licensed.
PROVIDERS="yahoo stooq fred sec"
SKIP_DOWNLOAD=0
if [[ -n "${TIINGO_API_KEY:-}" ]]; then
  PROVIDERS="tiingo $PROVIDERS"
  echo "TIINGO_API_KEY detected: tiingo added as preferred price source"
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start) START="$2"; shift 2 ;;
    --providers) PROVIDERS="$2"; shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

RUN_LOG="results/real/first_study.log"
mkdir -p results/real
# Redact anything that looks like a credential before it reaches the log.
redact() { sed -E 's/((api_?key|token|secret)[=: ]+)[A-Za-z0-9._-]{8,}/\1<redacted>/Ig'; }
exec > >(redact | tee -a "$RUN_LOG") 2>&1

step() { printf '\n=== %s ===\n' "$*"; }
die() {
  printf '\nFAILED: %s\n' "$1"
  printf 'Recovery: %s\n' "$2"
  exit 1
}

step "1/10 Environment validation"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3, 11), f"need Python>=3.11, have {sys.version}"' \
  || die "Python >= 3.11 required" "install Python 3.11+ or set PYTHON=/path/to/python3"
$PY -c 'import pandas, numpy, pyarrow, scipy, sklearn, yaml, jinja2, matplotlib, exchange_calendars, tabulate' \
  || die "missing packages" "run: $PY -m pip install -e '.[dev]'"
$PY -m pytest tests/ -q || die "test suite failed" "fix failing tests before running the study"

step "2/10 Research-freeze verification (no tuning permitted)"
$PY -c "import sys; sys.path.insert(0, 'src'); from aitb.freeze import verify_freeze; verify_freeze()" \
  || die "freeze verification failed" \
         "the study spec drifted from the current configs/research_freeze_v*.json — revert the changes, or increment FREEZE_VERSION and create a new freeze for a NEW study"

step "3/10 Environment fingerprint"
$PY - <<'EOF'
import json, platform, subprocess, sys
sys.path.insert(0, "src")
from datetime import datetime, timezone
from aitb.freeze import load_freeze
fp = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()),
    "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines(),
    "freeze_hash": load_freeze()["hash"],
}
json.dump(fp, open("results/real/run_fingerprint.json", "w"), indent=2)
print("fingerprint recorded (git_dirty=%s)" % fp["git_dirty"])
EOF

if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  step "4-5/10 Download and import SKIPPED (--skip-download)"
  # Refuse rather than silently backtest an empty store. The quality gate below
  # would catch this too, but failing here says why in one line instead of
  # sending the reader into data_quality.json.
  # data/real is the store the backtests actually read (config.REAL_DATA_DIR);
  # data/export_bundle is only the download staging area, so its presence
  # proves nothing about whether anything was imported.
  $PY - <<'EOF' || die "--skip-download was passed but there is no usable store in data/real" \
                       "nothing has been imported yet — re-run without --skip-download"
import sys
sys.path.insert(0, "src")
from aitb.data import realstore
from aitb.data.quality import store_fingerprint
prices = realstore.available("prices")
if not prices:
    print("no price datasets in the store", file=sys.stderr)
    raise SystemExit(1)
print(f"reusing existing store: {len(prices)} price series, "
      f"fingerprint {store_fingerprint()}")
EOF
else
  step "4/10 Real-data download (resumable; failures logged per symbol)"
  $PY scripts/download_real_data.py --providers $PROVIDERS --start "$START" \
      --output data/export_bundle \
    || die "download failed" "re-run this script: completed symbols are skipped, partial downloads resume"

  step "5/10 Checksum-verified import + provider reconciliation"
  $PY scripts/import_data_bundle.py --input data/export_bundle \
    || die "import failed" "inspect results/real/import_report.json; a checksum failure means the bundle was corrupted in transit — re-download"
fi

step "6/10 Data-quality gate (hard stop on FAIL)"
$PY scripts/validate_real_data.py \
  || die "quality gate returned FAIL — DO NOT BACKTEST" \
         "read results/real/data_quality.json, fix the fatal findings (or import better data), then re-run"

step "7/10 Frozen real-mode study (backtests, robustness, capacity, companies)"
$PY scripts/run_experiments.py --data-mode real \
  || die "experiments failed" "registry is append-only: re-running resumes where it stopped"
$PY scripts/run_robustness.py --data-mode real
$PY scripts/run_capacity.py --data-mode real --top 5
$PY scripts/run_company_analysis.py --data-mode real

step "8/10 Reports (research_report_full.html + decision_brief.html)"
$PY scripts/make_report.py --data-mode real
$PY scripts/make_decision_brief.py

step "9/10 Rebuild the published site on REAL results"
# Overwrites site/ with real-mode pages. The synthetic warning banner
# disappears on its own (it is keyed on the mode), and the data page lists only
# the companies that actually had price history — anything the providers could
# not serve is excluded and disclosed rather than shown as if it were tested.
$PY -c "import sys; sys.path.insert(0, 'src'); \
from aitb.platform.site import build_site; print(build_site('real'))" \
  || die "site build failed" "results exist; re-run: python scripts/research.py dashboard --data-mode real"
echo "site/ now reflects the real study — commit and push to publish"

step "10/10 Shareable results bundle"
STAMP=$(date -u +%Y%m%d_%H%M)
# Add the final data-store fingerprint to the run fingerprint.
$PY -c "
import json, sys; sys.path.insert(0, 'src')
from aitb.data.quality import store_fingerprint
fp = json.load(open('results/real/run_fingerprint.json'))
fp['data_store_fingerprint'] = store_fingerprint()
json.dump(fp, open('results/real/run_fingerprint.json', 'w'), indent=2)"
tar czf "first_real_study_${STAMP}.tar.gz" \
    reports/real \
    results/real/data_quality.json results/real/import_report.json \
    results/real/experiments.jsonl results/real/strategy_ranking.csv \
    results/real/robustness results/real/capacity.csv \
    results/real/company_analysis.csv results/real/run_fingerprint.json \
    results/real/holdout_lock.json \
    data/export_bundle/manifest.json \
    configs/research_freeze_v*.json \
    docs/prospective_testing_protocol.md 2>/dev/null || true

printf '\nDONE. Deliverables:\n'
printf '  reports/real/research_report_full.html\n'
printf '  reports/real/decision_brief.html\n'
printf '  site/                       (the published site, now on real data)\n'
printf '  first_real_study_%s.tar.gz (shareable bundle)\n' "$STAMP"
printf 'The bundle contains no API keys and no licensed raw price files.\n'
printf '\nTo publish:\n'
printf '  git add -A && git commit -m "First real-data study" && git push\n'
printf '\nThat commits the site AND the research record (registry, ranking,\n'
printf 'quality gate, holdout lock). Raw vendor prices and equity curves stay\n'
printf 'out of git deliberately — licensed data is not redistributable, and\n'
printf 'curves regenerate from the registry.\n'
