from __future__ import annotations

import math

import numpy as np


class Fusion2D:
    """Constant-velocity Kalman filter for local north/east map-match updates."""

    def __init__(self, accel_noise: float = 0.8, gate: float = 9.21):
        self.accel_noise = accel_noise
        self.gate = gate
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 100.0
        self.last_t: float | None = None
        self.accepted = 0
        self.rejected = 0
        self.last_d2: float | None = None

    @property
    def initialized(self) -> bool:
        return self.last_t is not None

    def _predict(self, dt: float) -> None:
        dt = max(0.0, min(dt, 2.0))
        if dt <= 0:
            return
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        q = self.accel_noise
        Q = q * np.array(
            [
                [dt**3 / 3.0, 0.0, dt**2 / 2.0, 0.0],
                [0.0, dt**3 / 3.0, 0.0, dt**2 / 2.0],
                [dt**2 / 2.0, 0.0, dt, 0.0],
                [0.0, dt**2 / 2.0, 0.0, dt],
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(
        self,
        north: float,
        east: float,
        timestamp: float,
        sigma: float = 1.5,
    ) -> dict:
        if not self.initialized:
            self.x = np.array([north, east, 0.0, 0.0], dtype=float)
            self.P = np.diag([sigma**2, sigma**2, 4.0, 4.0])
            self.last_t = timestamp
            self.accepted = 1
            return {
                "accepted": True,
                "north": north,
                "east": east,
                "d2": 0.0,
                "std_north": sigma,
                "std_east": sigma,
            }

        dt = timestamp - (self.last_t or timestamp)
        self._predict(dt)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = np.eye(2) * sigma**2
        S = H @ self.P @ H.T + R
        innovation = np.array([north, east]) - H @ self.x
        d2 = float(innovation @ np.linalg.inv(S) @ innovation)
        self.last_d2 = d2
        self.last_t = timestamp
        if d2 > self.gate:
            self.rejected += 1
            return {
                "accepted": False,
                "north": float(self.x[0]),
                "east": float(self.x[1]),
                "d2": d2,
                "std_north": float(math.sqrt(max(self.P[0, 0], 0.0))),
                "std_east": float(math.sqrt(max(self.P[1, 1], 0.0))),
            }

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P
        self.accepted += 1
        return {
            "accepted": True,
            "north": float(self.x[0]),
            "east": float(self.x[1]),
            "d2": d2,
            "std_north": float(math.sqrt(max(self.P[0, 0], 0.0))),
            "std_east": float(math.sqrt(max(self.P[1, 1], 0.0))),
        }

    def predict_only(self, timestamp: float) -> dict | None:
        if not self.initialized:
            return None
        dt = timestamp - (self.last_t or timestamp)
        self._predict(dt)
        self.last_t = timestamp
        return {
            "accepted": True,
            "predicted": True,
            "north": float(self.x[0]),
            "east": float(self.x[1]),
            "std_north": float(math.sqrt(max(self.P[0, 0], 0.0))),
            "std_east": float(math.sqrt(max(self.P[1, 1], 0.0))),
        }

    def status(self) -> dict:
        return {
            "initialized": self.initialized,
            "north": float(self.x[0]),
            "east": float(self.x[1]),
            "v_north": float(self.x[2]),
            "v_east": float(self.x[3]),
            "std_north": float(math.sqrt(max(self.P[0, 0], 0.0))),
            "std_east": float(math.sqrt(max(self.P[1, 1], 0.0))),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "last_d2": self.last_d2,
        }
