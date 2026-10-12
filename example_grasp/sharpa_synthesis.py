"""Run stock BODex with explicit mesh-sampling seeds and read-only seed audits.

Only this process is patched. Neither BODex source nor the sampler distribution
is changed; mesh surface samples use a reproducible seed instead of OS entropy.
"""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy

# Match the original entry point's import order: torch loads the environment's
# C++ runtime before scipy/trimesh on this Ubuntu server.
import torch
import numpy as np
import trimesh


def deterministic_surface(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['seed'] is None:
            mesh = bound.arguments['mesh']
            digest = hashlib.sha256(b'sphere-ablation-20261008')
            for array in [mesh.vertices, mesh.faces]:
                digest.update(np.ascontiguousarray(array).tobytes())
            digest.update(str(bound.arguments['count']).encode())
            bound.arguments['seed'] = int.from_bytes(digest.digest()[:8], 'little')
        return original(*bound.args, **bound.kwargs)

    return wrapped


trimesh.sample.sample_surface = deterministic_surface(trimesh.sample.sample_surface)
trimesh.sample.sample_surface_even = deterministic_surface(trimesh.sample.sample_surface_even)

from curobo.util.sample_grasp import HeurGraspSeedGenerator

original_samples = HeurGraspSeedGenerator.get_samples
audit_path = Path(os.environ.get('SPHERE_SEED_AUDIT', 'output/sharpa_seed_audit.jsonl'))
audit_path.parent.mkdir(parents=True, exist_ok=True)
assert not audit_path.exists(), 'Do not append to an old seed audit on restart'
calls = 0


def samples(self, batch, num_samples):
    global calls
    poses = original_samples(self, batch, num_samples)
    array = poses.detach().cpu().numpy()
    spheres = self.full_robot_model.get_state(poses.reshape(-1, poses.shape[-1])).link_spheres_tensor
    assert spheres.shape[-2] == int(os.environ.get('SPHERE_EXPECTED_COUNT', '34'))
    assert np.isfinite(array).all()
    row = dict(call=calls, shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest(),
               spheres=int(spheres.shape[-2]))
    with audit_path.open('a') as file:
        file.write(json.dumps(row)+'\n')
    calls += 1
    return poses


HeurGraspSeedGenerator.get_samples = samples
runpy.run_path(str(Path(__file__).with_name('plan_batch_env.py')), run_name='__main__')
