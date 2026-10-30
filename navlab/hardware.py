from __future__ import annotations

import glob
import os
from typing import Any


def list_serial_devices() -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for by_id in sorted(glob.glob("/dev/serial/by-id/*")):
        devices.append({"by_id": by_id, "path": os.path.realpath(by_id)})
    known = {item["path"] for item in devices}
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        for path in sorted(glob.glob(pattern)):
            if path not in known:
                devices.append({"by_id": None, "path": path})
    return devices


def hitl_status() -> dict[str, Any]:
    devices = list_serial_devices()
    checks = [
        {
            "name": "اتصال سریال Pixhawk",
            "ok": bool(devices),
            "detail": ", ".join(item["path"] for item in devices)
            if devices
            else "هیچ /dev/ttyACM* یا /dev/ttyUSB* پیدا نشد",
        },
        {
            "name": "PX4 HITL firmware (pwm_out_sim)",
            "ok": False,
            "detail": "نیاز به build/flash firmware با CONFIG_MODULES_SIMULATION_PWM_OUT_SIM=y",
        },
        {
            "name": "Gazebo Classic یا SIH",
            "ok": False,
            "detail": "HITL رسمی PX4 با Gazebo Harmonic نیست؛ Gazebo Classic یا SIH لازم است",
        },
        {
            "name": "دوربین واقعی + کالیبراسیون/زمان",
            "ok": False,
            "detail": "دوربین global shutter، intrinsics/extrinsics و همگام‌سازی زمان هنوز وصل نشده",
        },
        {
            "name": "تغذیه/حرارت/لرزش",
            "ok": False,
            "detail": "bench bring-up سخت‌افزار انجام نشده",
        },
    ]
    return {
        "serial": devices,
        "checks": checks,
        "ready": all(item["ok"] for item in checks),
        "note": "HITL بدون Pixhawk واقعی قابل تست نیست؛ SITL/SIH مسیر فعلی است.",
    }
