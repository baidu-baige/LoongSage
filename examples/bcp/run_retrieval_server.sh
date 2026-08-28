#!/usr/bin/env bash
# run_retrieval_server.sh — Stop existing processes, start the BCP retrieval server.
#
# Usage (run from repo root):
#   ./examples/bcp/run_retrieval_server.sh \
#       --data_dir /path/to/corpus \
#       --model /path/to/Qwen3-Embedding-8B \
#       --dense_cache /path/to/browsecomp_dense_cache.pkl
#
# Once the server is up, launch training separately, e.g.:
#   bash examples/start.sh qwen3_30b_a3b/bcp_h20_1node

set -euo pipefail
cd "$(dirname "$0")/../.."   # always run from repo root

SERVER_ARGS=("$@")
RETRIEVAL_PORT=9000

for ((i = 0; i < ${#SERVER_ARGS[@]}; i++)); do
    case "${SERVER_ARGS[i]}" in
        -h|--help)
            exec python3 "examples/bcp/retrieval_server.py" "${SERVER_ARGS[@]}"
            ;;
        --port)
            if ((i + 1 >= ${#SERVER_ARGS[@]})); then
                echo "ERROR: --port requires a value." >&2
                exit 2
            fi
            RETRIEVAL_PORT="${SERVER_ARGS[i + 1]}"
            ;;
        --port=*)
            RETRIEVAL_PORT="${SERVER_ARGS[i]#--port=}"
            ;;
    esac
done

mkdir -p log

# ---------------------------------------------------------------------------
# [1/2] Stop existing retrieval server
# ---------------------------------------------------------------------------
echo "=== [1/2] Stopping existing retrieval server ==="
pkill -f "retrieval_server.py" 2>/dev/null || true
sleep 2

# ---------------------------------------------------------------------------
# [2/2] Start retrieval server
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/2] Starting retrieval server (port ${RETRIEVAL_PORT}) ==="

if lsof -i :"${RETRIEVAL_PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port ${RETRIEVAL_PORT} already in use — assuming server is running."
    exit 0
fi

python3 "examples/bcp/retrieval_server.py" "${SERVER_ARGS[@]}" \
    > log/retrieval_server.log 2>&1 &
RETRIEVAL_PID=$!
echo "Retrieval server PID=${RETRIEVAL_PID}, waiting for ready..."

for i in $(seq 1 1200); do
    if curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/health" > /dev/null 2>&1; then
        echo "Ready (${i}s)."
        break
    fi
    if ! kill -0 "${RETRIEVAL_PID}" 2>/dev/null; then
        echo "ERROR: server died. Check log/retrieval_server.log."
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:${RETRIEVAL_PORT}/health" > /dev/null 2>&1; then
    echo "ERROR: server did not start within 1200s."
    exit 1
fi
