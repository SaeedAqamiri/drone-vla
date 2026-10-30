"""Render frames via Replicator's native writer (bypasses the python annotator
path that segfaults in libomni.syntheticdata on this machine).

Usage: python render_writer45.py
"""

import os
import sys

sys.argv += [
    "--/rtx-transient/scenedb/maxInstancesLimit=500000",
    "--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=true",
    "--/renderer/activeGpu=0",
    "--/renderer/multiGpu/enabled=False",
]

print("booting ...", flush=True)
from isaacsim import SimulationApp

OUT = "/home/arman/drone-vla/outputs/isaac45_frames"
os.makedirs(OUT, exist_ok=True)

import isaacsim
EXP = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.kit")
app = SimulationApp({"headless": True, "experience": EXP})

import numpy as np
import omni.replicator.core as rep

cube = rep.create.cube(position=(0, 0, 50), scale=100)
rep.create.light(light_type="dome", intensity=3000)
rep.create.light(light_type="distant", intensity=100000)

cam = rep.create.camera(position=(250, -250, 180), look_at=(0, 0, 50))
rp = rep.create.render_product(cam, (256, 256))

from isaacsim.replicator.writers.scripts.writers import register_writers
register_writers()  # extension startup order can wipe the registry; re-register
writer = rep.writers.get("PytorchWriter", init_params={"output_dir": OUT, "rgb": True})
writer.attach([rp])

for _ in range(10):
    app.update()

timeline = rep.orchestrator.get_timeline()
timeline.play()
for i in range(5):
    rep.orchestrator.step(rt_subframes=48)
    print(f"frame batch {i} done", flush=True)
timeline.stop()  # flush writer queue
app.update()

writer.detach()
print(f"frames written to {OUT}", flush=True)

with open(os.path.join(OUT, "DONE"), "w") as f:
    f.write("ok\n")

os._exit(0)
