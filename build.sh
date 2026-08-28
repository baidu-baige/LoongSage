#!/usr/bin/env bash

set -e
set -x

# remove previous
rm -rf output/

mkdir -p output

# make
# cp path/to/artifact output/

# ---------------------------------------------------------------------------
# Unit tests
#
# Run all unit tests under tests/ut/ with coverage report:
#   bash build.sh test
#
# Run a specific subdirectory or file under tests/ut/:
#   bash build.sh test tests/ut/utils/
#   bash build.sh test tests/ut/utils/test_registry.py
#   bash build.sh test tests/ut/data_factory/test_data_source.py
#
# Coverage notes:
#   - Line and branch coverage are reported via pytest-cov / coverage.py.
#   - Test dependencies come from the "test" extra in pyproject.toml.
#   - HTML report is written to output/coverage_html/.
#   - coverage.py does not provide a dedicated "function coverage" metric;
#     a function is considered covered when its definition line is executed.
#
# Viewing the HTML coverage report:
#   Option 1 - open directly (macOS):
#     open output/coverage_html/index.html
#
#   Option 2 - serve via HTTP and open in browser:
#     python3 -m http.server 8080 --directory output/coverage_html
#     # then visit http://localhost:8080
# ---------------------------------------------------------------------------

if [[ $# -gt 0 ]] && [[ "${1}" == "test" ]]; then
    TEST_TARGET="${2:-tests/ut/}"
    python3 -m pip install -q -e ".[ci]"
    python3 -m pytest "${TEST_TARGET}" -v \
        --cov=coda \
        --cov-branch \
        --cov-report=term-missing \
        --cov-report=html:output/coverage_html
fi
