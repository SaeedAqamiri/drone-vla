"""Minimal RTX camera render test for Isaac Sim 4.5 on this machine.

Boots SimulationApp headless (the same code path that segfaults on 5.1 with
cameras), creates a scene with a cube and a camera, renders one RGB frame via
Replicator, and reports whether the buffer is real.

Usage: python render_test45.py
"""

import sys

# Cap the RTX scene-db TLAS buffers: default (-1) sizes a 7.5GB buffer that
# cannot fit the 8GB GPU -> segfault in rtx.scenedb during boot.
sys.argv += [
    "--/rtx-transient/scenedb/maxInstancesLimit=500000",
    "--/rtx-transient/scenedb/forceMaxTLASInstancesLimit=true",
    "--/renderer/activeGpu=0",
    "--/renderer/multiGpu/enabled=False",
]

print("booting SimulationApp (4.5, headless, full kit) ...", flush=True)
from isaacsim import SimulationApp

import os
import isaacsim
EXP = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.base.kit")
app = SimulationApp({"headless": True, "experience": EXP})
print("KIT BOOTED", flush=True)

import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()

cube = rep.create.cube(position=(0, 0, 50), scale=100)
rep.create.light(light_type="dome", intensity=3000)
rep.create.light(light_type="distant", intensity=100000)

cam = rep.create.camera(position=(250, -250, 180), look_at=(0, 0, 50))
rp = rep.create.render_product(cam, (256, 256))
reader = rep.AnnotatorRegistry.get_annotator("rgb")
reader.attach([rp])

for _ in range(10):
    app.update()

timeline.stop()  # halt graph execution so the annotator buffer stops being recycled

for attempt in range(5):
    rep.orchestrator.step(rt_subframes=48)
    timeline.stop()
    data = reader.get_data()
    arr = np.array(data, copy=True)  # deep-copy out of replicator memory immediately
    if arr.size > 0:
        img0 = arr.reshape(-1, 256, 256, arr.shape[-1])[0] if arr.ndim == 4 else arr
        print(f"attempt {attempt}: mean={np.asarray(img0).mean():.1f}", flush=True)
        if np.asarray(img0).mean() > 1.0:
            break
    else:
        print(f"attempt {attempt}: empty", flush=True)
print("buffer:", arr.shape, arr.dtype, flush=True)
if arr.size >= 256 * 256 * 3:
    img = arr.reshape(-1, 256, 256, arr.shape[-1])[0] if arr.ndim == 4 else arr
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    out = Path("/home/arman/drone-vla/outputs/isaac45_render_proof.raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        img.tofile(f)
    print(f"RGB frame OK: {img.shape} mean={img.mean():.1f} -> {out}", flush=True)
    print("RENDER VERDICT: WORKS", flush=True)
else:
    print(f"RENDER VERDICT: EMPTY BUFFER", flush=True)

# Data is flushed; exit immediately — kit teardown has a flaky race.
import os
os._exit(0)
