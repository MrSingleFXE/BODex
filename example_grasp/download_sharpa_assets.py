"""Fetch the pinned official Sharpa meshes and verify each SHA-256."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / 'src/curobo/content/assets/robot/sharpa_wave'
COMMIT = '0d19cac602f46456b819e4b6a2c09a74982c9a3e'
BASE = f'https://raw.githubusercontent.com/sharpa-robotics/sharpa-urdf-usd-xml/{COMMIT}/wave_01/right_sharpa_wave/meshes/'

def main():
    manifest = json.loads((ROOT / 'mesh_manifest.json').read_text())
    (ROOT / 'meshes').mkdir(exist_ok=True)
    for name, expected in manifest.items():
        assert Path(name).name == name
        path = ROOT / 'meshes' / name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
            continue
        with urlopen(BASE + name, timeout=60) as response:
            content = response.read()
        assert hashlib.sha256(content).hexdigest() == expected, name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    print(f'Verified {len(manifest)} official mesh files at commit {COMMIT}')

if __name__ == '__main__':
    main()
