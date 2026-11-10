"""Experimental SSH client for a GIM-DKM research worker; not a flight API."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import select
import shlex
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .deep import _read_rgb
from .types import MatchOutcome


class RemoteGIMMatcher:
    def __init__(self, host: str, *, remote_root="/home/ava/drone-vla-eval/gim",
                 python="/home/ava/flux3-klein/venv-sglang/bin/python", max_dim=768,
                 mps_pipe="/home/ava/drone-vla-eval/private_mps", timeout=120,
                 min_inliers=15, min_ratio=.15, min_coverage=.03):
        if max_dim != 768:
            raise ValueError("remote GIM worker currently requires --max-dim 768")
        if not host or host.startswith("-"):
            raise ValueError("invalid SSH host")
        if min_inliers < 15 or min_ratio < .15 or min_coverage < .03:
            raise ValueError("remote thresholds cannot be looser than the worker's thresholds")
        self.min_inliers, self.min_ratio, self.min_coverage = min_inliers, min_ratio, min_coverage
        self.timeout = timeout
        self._recv = bytearray()
        self._broken = False
        self._known = set()
        self._refs = {}
        self._last_query_key = None
        self._last_encoded = None
        env = ["env", "PYTHONWARNINGS=ignore::DeprecationWarning", f"PYTHONPATH={remote_root}:{remote_root}/deps",
               f"CUDA_MPS_PIPE_DIRECTORY={mps_pipe}", python, "-u", f"{remote_root}/gim_match_worker.py"]
        self.proc = subprocess.Popen(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                      host, shlex.join(env)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        try:
            ready = self._read()
            if not ready.get("ready") or ready.get("max_dim") != max_dim:
                raise RuntimeError(f"unexpected worker greeting: {ready}")
        except Exception:
            self.close()
            raise

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            newline = self._recv.find(b"\n")
            if newline >= 0:
                line = bytes(self._recv[:newline])
                del self._recv[:newline+1]
                return json.loads(line.decode("utf-8"))
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("remote GIM worker timed out")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError(f"remote GIM worker exited: {self.proc.poll()}")
            self._recv.extend(chunk)
            if len(self._recv) > 8 * 1024 * 1024:
                raise ValueError("remote GIM response exceeds size limit")

    def _encode(self, image):
        file_key = None
        query_key = None
        if isinstance(image, (str, Path)):
            stat = Path(image).stat()
            file_key = (str(image), stat.st_size, stat.st_mtime_ns)
            if file_key in self._refs:
                return self._refs[file_key]
        elif isinstance(image, np.ndarray):
            # Camera producers often reuse the same NumPy buffer for each
            # frame. Object identity is therefore not a valid cache key.
            raw_image = np.ascontiguousarray(image)
            query_key = (raw_image.shape, raw_image.dtype.str,
                         hashlib.sha256(raw_image.tobytes()).digest())
        if query_key is not None and query_key == self._last_query_key:
            return self._last_encoded
        resized = _read_rgb(image, 768)
        if resized is None:
            raise ValueError("image read failed")
        ok, encoded = cv2.imencode(".png", resized[0])
        if not ok:
            raise ValueError("PNG encoding failed")
        raw = encoded.tobytes()
        pair = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii"))
        if file_key is not None:
            self._refs[file_key] = pair
        else:
            self._last_query_key, self._last_encoded = query_key, pair
        return pair

    def match(self, ref, query):
        outcome, _, _ = self._match(ref, query, want_points=False)
        return outcome

    def match_with_points(self, ref, query):
        """Return outcome plus raw sampled points in resized input coordinates."""
        return self._match(ref, query, want_points=True)

    def _match(self, ref, query, *, want_points):
        started = time.time()
        names = tuple("<array>" if isinstance(i, np.ndarray) else str(i) for i in (ref, query))
        try:
            if self._broken:
                raise RuntimeError("remote matcher is closed after a prior execution failure")
            pairs = [self._encode(ref), self._encode(query)]
            for attempt in range(2):
                message = {"want_points": want_points}
                for name, (key, image) in zip(("ref", "query"), pairs):
                    message[f"{name}_key"] = key
                    if key not in self._known or attempt:
                        message[f"{name}_png"] = image
                self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
                self.proc.stdin.flush()
                result = self._read()
                if result.get("execution_error") and "missing cached" in result.get("error", "") and not attempt:
                    continue
                break
            if result.get("execution_error"):
                raise RuntimeError(result.get("error", "remote inference failed"))
            self._known.update(key for key, _ in pairs)
            homography = result.get("homography")
            if homography is not None:
                homography = np.asarray(homography, dtype=np.float64)
                if homography.shape != (3, 3) or not np.isfinite(homography).all():
                    raise ValueError("invalid remote homography")
            success = (bool(result["success"]) and int(result["inliers"]) >= self.min_inliers
                       and float(result["inlier_ratio"]) >= self.min_ratio
                       and float(result["coverage"]) >= self.min_coverage)
            points = (None, None)
            if want_points:
                points = (np.asarray(result["ref_points"], dtype=np.float64).reshape(-1, 2),
                          np.asarray(result["query_points"], dtype=np.float64).reshape(-1, 2))
                if points[0].shape != points[1].shape or len(points[0]) > 10000 or any(not np.isfinite(p).all() for p in points):
                    raise ValueError("invalid remote correspondences")
            outcome = MatchOutcome(*names, "gim_dkm_remote", int(result["inliers"]),
                                   float(result["inlier_ratio"]), float(result["coverage"]),
                                   result.get("reproj"), result.get("scale"), result.get("rotation"),
                                   (time.time()-started)*1000, success,
                                   error=result.get("error", ""), homography=homography)
            return outcome, *points
        except Exception as error:
            # A late response after a timeout cannot be correlated with the
            # next request. Quarantine the connection instead of risking a
            # stale homography being used for a different image pair.
            self._broken = True
            try:
                self.close()
            except Exception:
                pass
            outcome = MatchOutcome(*names, "gim_dkm_remote", 0, 0, 0, None, None, None,
                                   (time.time()-started)*1000, False,
                                   error=f"{type(error).__name__}: {error}", execution_error=True)
            return outcome, None, None

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=5)
