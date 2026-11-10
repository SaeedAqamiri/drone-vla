#!/usr/bin/env bash
# شروع PX4 SITL + Gazebo Harmonic
#
#   ./scripts/run_sitl.sh              # بی‌پنجره (HEADLESS) - برای تست و CI
#   ./scripts/run_sitl.sh gui          # با پنجره Gazebo روی ویندوز (WSLg)
#   ./scripts/run_sitl.sh gui x500_mono_cam_down baylands
#
# پیش‌نیاز: PX4-Autopilot بیلد شده (باینری در build/px4_sitl_default/bin/px4)
# برای فرود تمیز: Ctrl+C یا بستن پنجره Gazebo

set -uo pipefail

MODE="${1:-headless}"
MODEL="${2:-x500_mono_cam_down}"
WORLD="${3:-baylands}"

PX4_DIR="${PX4_DIR:-}"
if [ -z "$PX4_DIR" ]; then
  for d in "$HOME/drone-vla/PX4-Autopilot" "$HOME/PX4-Autopilot"; do
    [ -d "$d" ] && PX4_DIR="$d" && break
  done
fi
if [ -z "$PX4_DIR" ] || [ ! -d "$PX4_DIR" ]; then
  echo "PX4-Autopilot پیدا نشد. PX4_DIR را ست کن."; exit 1
fi

if [ ! -x "$PX4_DIR/build/px4_sitl_default/bin/px4" ]; then
  echo "PX4 بیلد نشده. اول: cd \"$PX4_DIR\" && make px4_sitl_default"
  exit 1
fi

cd "$PX4_DIR" || exit 1
export PX4_GZ_WORLD="$WORLD"
# GIT_SUBMODULES_ARE_EVIL: چند submodule اختیاری (voxl2/nuttx/jsbsim) در دسترس نیستند
export GIT_SUBMODULES_ARE_EVIL=1

if [ "$MODE" = "headless" ]; then
  export HEADLESS=1
  echo "حالت headless — world=$WORLD model=$MODEL"
else
  unset HEADLESS
  echo "حالت GUI — world=$WORLD model=$MODEL"
  echo "اگر پنجره باز نشد: wsl -d Ubuntu-24.04 و DISPLAY=:0 را چک کن"
fi

exec make px4_sitl "gz_${MODEL}"
