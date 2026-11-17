import hashlib
import json
import os
from pathlib import Path


def project_root():
    for path in Path(__file__).resolve().parents:
        config = path / "pyproject.toml"
        if config.is_file() and 'name = "clinical-image-fusion-lab"' in config.read_text():
            return path
    raise RuntimeError("Project root and pyproject.toml not found")


ROOT = project_root()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    with path.open("x" if exclusive else "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def config():
    return json.loads((ROOT / "protocol.json").read_text())


def fraction(salt, key):
    return int(hashlib.sha256((salt + ":" + str(key)).encode()).hexdigest()[:16], 16) / 2**64
