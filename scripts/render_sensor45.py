"""Test the 4.5-native isaacsim.sensors.camera.Camera capture path.

Boots, spawns cube+lights+camera via the native Camera sensor class, steps the
world, grabs RGBA, raw-saves to disk, exits immediately.
"""

import os
import sys

sys.argv += [
    "--/rtx-transient/scenedb/maxInstancesLimit=4000000",
    "--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=true",
    "--/renderer/activeGpu=0",
    "--/renderer/multiGpu/enabled=False",
]

print("booting ...", flush=True)
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.sensors.camera import Camera
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
cube = rep.create.cube(position=(0, 0, 50), scale=100)
rep.create.light(light_type="dome", intensity=3000)
rep.create.light(light_type="distant", intensity=100000)

camera = Camera(
    prim_path="/World/CameraSensor",
    resolution=(256, 256),
    position=np.array([250.0, -250.0, 180.0]),
    orientation=np.array([0.4, 0.3, 0.3, 0.8]),
)
camera.initialize()

for _ in range(20):
    app.update()

import omni.timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(6):
    rep.orchestrator.step(rt_subframes=32)
    app.update()

rgba = camera.get_rgba()
arr = np.asarray(rgba)
print("rgba:", arr.shape, arr.dtype, flush=True)

if arr.size >= 256 * 256 * 3:
    img = arr[..., :3].astype(np.uint8)
    with open("/home/arman/drone-vla/outputs/isaac45_sensor_frame.raw", "wb") as f:
        img.tofile(f)
    print(f"SENSOR FRAME OK: mean={img.mean():.1f}", flush=True)
else:
    print("SENSOR FRAME EMPTY", flush=True)

timeline.stop()
os._exit(0)

