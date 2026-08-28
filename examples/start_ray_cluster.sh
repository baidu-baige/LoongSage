#!/bin/bash
# start_ray_cluster.sh - Start Ray cluster manually for disaggregated mode
#
# Usage: bash examples/start_ray_cluster.sh <master-ip>
# Run the SAME command on the master and each worker pod; the script
# auto-detects the role from the local node IP.
# Example: bash examples/start_ray_cluster.sh <head-ip>
# Nodes default to 8 GPUs; override with NUM_GPUS=4 bash examples/start_ray_cluster.sh <head-ip>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <master-ip>" >&2
    exit 2
fi
MASTER_IP="$1"

RAY_PORT="6385"
DASHBOARD_PORT=6386
RUNTIME_ENV_AGENT_PORT=20002
CLIENT_SERVER_PORT=10001
# GPUs per node; override for nodes with a different GPU count, e.g. NUM_GPUS=4
NUM_GPUS="${NUM_GPUS:-8}"

echo "=================================="
echo "Ray Cluster Setup for CODA Training"
echo "Master: ${MASTER_IP}"
echo "=================================="

# Detect if running on Master or Worker
MY_IP=$(python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('${MASTER_IP}',80)); print(s.getsockname()[0]); s.close()")

echo "Current node IP: ${MY_IP}"

if [ "${MY_IP}" = "${MASTER_IP}" ]; then
    echo "[Master] Starting Ray head..."
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --head --port=${RAY_PORT} --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT \
	--runtime-env-agent-port $RUNTIME_ENV_AGENT_PORT \
        --ray-client-server-port $CLIENT_SERVER_PORT \
        --disable-usage-stats \
        --num-gpus=${NUM_GPUS} --num-cpus=100 \
        --object-store-memory=32000000000
    echo "[Master] Ray head started at ${MASTER_IP}:${RAY_PORT}"
    echo "[Master] Run the following on each Worker to join:"
    echo "  bash examples/start_ray_cluster.sh ${MASTER_IP}"
else
   echo "[Worker] Waiting for Ray head at ${MASTER_IP}:${RAY_PORT}..."
   for i in $(seq 1 60); do
       if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(('${MASTER_IP}',${RAY_PORT})); s.close(); sys.exit(0)" 2>/dev/null; then
           echo "[Worker] Ray head is up!"
           break
       fi
       echo "[Worker] Not ready yet ($i/60)..."
       sleep 5
   done

   echo "[Worker] Joining Ray cluster..."
   ray stop --force 2>/dev/null || true
   sleep 2
   ray start --address="${MASTER_IP}:${RAY_PORT}" \
       --num-gpus=${NUM_GPUS} --num-cpus=100 \
       --object-store-memory=32000000000
   echo "[Worker] Joined Ray cluster at ${MASTER_IP}:${RAY_PORT}"
   echo "[Worker] Run 'ray status' to verify cluster"
fi

sleep 2
echo "=================================="
echo "Ray cluster status:"
ray status
