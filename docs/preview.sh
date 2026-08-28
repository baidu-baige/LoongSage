#!/usr/bin/env bash
# Local acceptance preview: build both languages, lay them out exactly like the
# future GitHub Pages site (English at the site root, Chinese under /zh/) and
# serve over HTTP.
#
#   bash docs/preview.sh            # build + serve on :8000
#   PORT=8123 bash docs/preview.sh  # other port
#
#   English  ->  http://127.0.0.1:8000/
#   Chinese  ->  http://127.0.0.1:8000/zh/
#
# HTTP (not file://) is required: full-text search loads searchindex.js
# asynchronously and would be blocked by the same-origin policy on file://.
set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8000}

# Must be a clean slate: the `mv` below would otherwise nest build/en/zh/zh on
# a second run. CI never hits this (fresh checkout every time), local runs do.
rm -rf ./build

bash ./build.sh en
bash ./build.sh zh
mv ./build/zh ./build/en/

echo
echo "  English  http://127.0.0.1:${PORT}/"
echo "  中文     http://127.0.0.1:${PORT}/zh/"
echo
exec python3 -m http.server "$PORT" -d ./build/en
