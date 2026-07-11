#!/usr/bin/env bash
# Build submission.tar.gz with main.py and deck.csv at the archive root.
# Usage: tools/package.sh [deck_csv]   (default: ./deck.csv)
set -euo pipefail
cd "$(dirname "$0")/.."

DECK="${1:-deck.csv}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp main.py "$STAGE/"
cp "$DECK" "$STAGE/deck.csv"
cp -R ptcg_agent "$STAGE/ptcg_agent"
cp -R cg "$STAGE/cg"
find "$STAGE" -name __pycache__ -type d -exec rm -rf {} +

# COPYFILE_DISABLE stops macOS bsdtar from embedding AppleDouble (._*)
# metadata, which Kaggle's Python tarfile would extract as junk files that
# break directory scanners (e.g. the opponent meta-deck loader).
find "$STAGE" \( -name '._*' -o -name '.DS_Store' \) -delete
COPYFILE_DISABLE=1 tar -czf submission.tar.gz -C "$STAGE" .
echo "submission.tar.gz:"
tar -tzf submission.tar.gz | sort | head -20
du -h submission.tar.gz
