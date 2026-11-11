#!/usr/bin/env bash
# Rebuild the HTML benchmark pages from everything under $BENCH_S3_URI and upload them:
#
#   $BENCH_S3_URI/index.html                       comparison of all runs
#   $BENCH_S3_URI/runs/<run>.html                  one page per results JSON
#   $BENCH_S3_URI/findings/<topic>/<name>.html     each findings/<topic>/<name>.md, rendered
#
# upload_results.sh runs this after uploading. Needs uv and the AWS CLI.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f bench/.env ]]; then set -a; . bench/.env; set +a; fi
base="${BENCH_S3_URI:?set BENCH_S3_URI in bench/.env (see bench/.env.example)}"
base="${base%/}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
aws s3 sync --only-show-errors "$base/results/" "$tmp/src/results/" --exclude '*' --include '*.json'
aws s3 sync --only-show-errors "$base/findings/" "$tmp/src/findings/" --exclude '*' --include '*.md'
uv run --quiet bench/report.py "$tmp/src" "$tmp/site"
aws s3 cp --only-show-errors --recursive --content-type 'text/html; charset=utf-8' "$tmp/site/" "$base/"
echo "report: $base/index.html"
