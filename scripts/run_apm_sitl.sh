#!/bin/bash
# ArduPilot SITL + Gazebo (Harmonic) launcher for drone-vla.
#
#   scripts/run_apm_sitl.sh                 # iris on the runway world
#   APM_WORLD=baylands_apm.sdf scripts/run_apm_sitl.sh
#
# The pilot (scripts/apm_flight_logger.py) then connects to
# udpin://0.0.0.0:$APM_OUT_PORT (default 14600).
#
# Notes
# - ArduPilot talks to Gazebo over JSON FDM (udp 9002), not gz-transport.
# - No `offboard` mode exists: the logger uses GUIDED + velocity setpoints.
# - `--no-mavproxy` keeps MAVProxy from occupying 14550/14551, so the pilot
#   owns the telemetry port.

set -uo pipefail

APM_DIR=${APM_DIR:-/home/drone/ardupilot}
APM_GZ_DIR=${APM_GZ_DIR:-/home/drone/ardupilot_gazebo}
APM_WORLD=${APM_WORLD:-iris_runway.sdf}
APM_VEHICLE=${APM_VEHICLE:-ArduCopter}
APM_FRAME=${APM_FRAME:-gazebo-iris}
APM_OUT_PORT=${APM_OUT_PORT:-14600}
APM_SYSID=${APM_SYSID:-1}

BIN=$APM_DIR/build/sitl/bin/arducopter
[ -x "$BIN" ] || { echo "ArduPilot SITL not built: $BIN"; exit 1; }

export GZ_SIM_SYSTEM_PLUGIN_PATH="${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH:-}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$APM_GZ_DIR/build:$GZ_SIM_SYSTEM_PLUGIN_PATH"
export GZ_SIM_RESOURCE_PATH="$APM_GZ_DIR/models:$APM_GZ_DIR/worlds:$GZ_SIM_RESOURCE_PATH"

echo "world=$APM_WORLD vehicle=$APM_VEHICLE frame=$APM_FRAME port=$APM_OUT_PORT"
echo "GZ_SIM_RESOURCE_PATH=$GZ_SIM_RESOURCE_PATH"

pkill -9 -f 'gz sim' 2>/dev/null
pkill -9 -f 'sim_vehicle.py' 2>/dev/null
pkill -9 -f 'arducopter' 2>/dev/null
sleep 2

gz sim -v2 -r -s "$APM_WORLD" > /tmp/apm_gz.log 2>&1 &
GZ_PID=$!

WORLD=${APM_WORLD%.sdf}
for i in $(seq 1 120); do
  if timeout 10 gz topic -l 2>/dev/null | grep -q "^/world/$WORLD/clock"; then
    echo "gazebo ready after ${i}s"
    break
  fi
  sleep 1
done

cd "$APM_DIR" ||	exit 1
python3 Tools/autotest/sim_vehicle.py \
  -v "$APM_VEHICLE" \
  -f "$APM_FRAME" \
  --model JSON \
  --no-mavproxy \
  --out="udpout:127.0.0.1:$APM_OUT_PORT" \
  --sysid "$APM_SYSID" \
  --no-rebuild \
  > /tmp/apm_sitl.log 2>&1 &
SITL_PID=$!

echo "gazebo pid=$GZ_PID  sitl pid=$SITL_PID"
echo "waiting for SITL heartbeat on udp $APM_OUT_PORT ..."
for i in $(seq 1 90); do
  if grep -q "ArduPilot Ready\|Ready to FLY" /tmp/apm_sitl.log 2>/dev/null; then
    echo "SITL ready after ${i}s"
    break
  fi
  sleep 1
done
tail -5 /tmp/apm_sitl.log
echo "--- ready ---"
