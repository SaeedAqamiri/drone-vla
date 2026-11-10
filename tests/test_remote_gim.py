import io
import json
import os
import unittest

import numpy as np

from navlab.lab.remote_gim import RemoteGIMMatcher


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()


def client(replies):
    matcher = RemoteGIMMatcher.__new__(RemoteGIMMatcher)
    matcher.proc = FakeProcess()
    matcher._known, matcher._refs = set(), {}
    matcher._broken = False
    matcher._last_query_key = matcher._last_encoded = None
    matcher.min_inliers, matcher.min_ratio, matcher.min_coverage = 15, .15, .03
    matcher._read = lambda: replies.pop(0)
    matcher.close = lambda: None
    return matcher


def success(h=None):
    return {"inliers": 16, "inlier_ratio": .7, "coverage": .4, "success": True,
            "reproj": 1, "scale": 1, "rotation": 0,
            "homography": h if h is not None else np.eye(3).tolist()}


class RemoteGIMTest(unittest.TestCase):
    def test_cached_inputs_and_missing_cache_retry(self):
        matcher = client([success(), {"execution_error": True, "error": "missing cached ref"}, success()])
        a, b = np.zeros((32, 32, 3), np.uint8), np.ones((32, 32, 3), np.uint8)
        self.assertTrue(matcher.match(a, b).success)
        self.assertTrue(matcher.match(a, b).success)
        records = [json.loads(line) for line in matcher.proc.stdin.getvalue().splitlines()]
        self.assertIn("ref_png", records[0])
        self.assertNotIn("ref_png", records[1])
        self.assertIn("ref_png", records[2])
        self.assertIn("query_png", records[2])

    def test_invalid_homography_fails_closed(self):
        matcher = client([success([[1, 2], [3, 4]])])
        image = np.zeros((16, 16, 3), np.uint8)
        out = matcher.match(image, image)
        self.assertFalse(out.success)
        self.assertTrue(out.execution_error)
        self.assertIn("invalid remote homography", out.error)

    def test_worker_error_fails_closed(self):
        matcher = client([{"execution_error": True, "error": "OOM"}])
        image = np.zeros((16, 16, 3), np.uint8)
        out = matcher.match(image, image)
        self.assertTrue(out.execution_error)
        self.assertIn("OOM", out.error)
        self.assertTrue(matcher.match(image, image).execution_error)

    def test_stricter_local_coverage_rejects_remote_success(self):
        reply = success()
        reply["coverage"] = .08
        matcher = client([reply])
        matcher.min_coverage = .10
        image = np.zeros((16, 16, 3), np.uint8)
        outcome = matcher.match(image, image)
        self.assertFalse(outcome.success)
        self.assertFalse(outcome.execution_error)

    def test_reused_camera_buffer_does_not_return_stale_image(self):
        matcher = client([success(), success()])
        image = np.zeros((16, 16, 3), np.uint8)
        matcher.match(image, image)
        image[:] = 123
        matcher.match(image, image)
        records = [json.loads(line) for line in matcher.proc.stdin.getvalue().splitlines()]
        self.assertNotEqual(records[0]["query_key"], records[1]["query_key"])
        self.assertIn("query_png", records[1])

    def test_returned_point_pairs_are_validated(self):
        reply = success()
        reply.update(ref_points=[[1, 2], [3, 4]], query_points=[[5, 6], [7, 8]])
        matcher = client([reply])
        image = np.zeros((16, 16, 3), np.uint8)
        outcome, ref, query = matcher.match_with_points(image, image)
        self.assertTrue(outcome.success)
        self.assertEqual(ref.shape, (2, 2))
        self.assertEqual(query.shape, (2, 2))
        self.assertTrue(json.loads(matcher.proc.stdin.getvalue())["want_points"])

    def test_partial_worker_line_still_times_out(self):
        read_fd, write_fd = os.pipe()
        try:
            matcher = RemoteGIMMatcher.__new__(RemoteGIMMatcher)
            matcher.timeout = .02
            matcher._recv = bytearray()
            matcher.proc = type("P", (), {"stdout": os.fdopen(read_fd, "rb", buffering=0)})()
            os.write(write_fd, b'{"partial":')
            with self.assertRaises(TimeoutError):
                matcher._read()
            matcher.proc.stdout.close()
        finally:
            os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
