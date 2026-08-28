#!/usr/bin/env bash
# Build one language tree.
#
#   bash docs/build.sh en   ->  docs/build/en
#   bash docs/build.sh zh   ->  docs/build/zh
#
# docs/en and docs/zh are two independent Sphinx source trees that share a
# single configuration directory (docs/), hence --conf-dir. Paths such as
# html_static_path resolve relative to the conf dir, so docs/_static/ is
# shared by both languages.
#
# DOC_LANG (not LANG: overriding the shell locale would change the
# encoding behaviour of child processes) tells conf.py which tree is being
# built; conf.py maps it to the Sphinx language code (zh -> zh_CN).
#
# Extra sphinx-build flags can be passed through SPHINX_OPTS, e.g.
#   SPHINX_OPTS="-W --keep-going" bash docs/build.sh zh
set -euo pipefail
cd "$(dirname "$0")"

LANG_CODE=${1:?usage: build.sh <en|zh>}
case "$LANG_CODE" in
  en | zh) ;;
  *)
    echo "build.sh: unknown language '$LANG_CODE' (expected en or zh)" >&2
    exit 1
    ;;
esac

# Word-split SPHINX_OPTS into an array so that an empty value adds no argument.
EXTRA_OPTS=()
if [ -n "${SPHINX_OPTS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_OPTS=(${SPHINX_OPTS})
fi

DOC_LANG=$LANG_CODE sphinx-build -b html \
  --conf-dir ./ \
  ${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"} \
  "./$LANG_CODE" "./build/$LANG_CODE"
