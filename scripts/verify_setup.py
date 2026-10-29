"""Verify framework, bundled checkpoint/data identities and installed dependency pins."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from setup_framework import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',action='store_true',help='Also require the test HTTP client')
    parser.add_argument('--files-only',action='store_true',help='Check identities without importing or requiring installed libraries')
    args=parser.parse_args()
    try:
        framework=verify(Path(os.environ.get('MICRO_TRANSFORMER_ROOT',ROOT.parent/'micro-transformer')))
        provenance=json.loads((ROOT/'checkpoints.json').read_text())
        for name,expected in provenance['bundled_files'].items():
            path=ROOT/name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Bundled artifact hash mismatch: '+name)
        installed={}
        if not args.files_only:
            if sys.version_info[:2]!=(3,12):raise ValueError('Use the supported Python 3.12 environment; other Python versions have not been qualified')
            needed=['torch','numpy','fastapi','uvicorn','pydantic','starlette']+(['httpx'] if args.tests else [])
            pins={}
            for line in (ROOT/'constraints.txt').read_text().splitlines():
                if '==' in line and not line.startswith('#'):
                    name,version=line.split('==');pins[name.lower().replace('_','-')]=version
            for name in needed:
                actual=importlib.metadata.version(name)
                if actual!=pins[name]:raise ValueError(f'{name}: expected {pins[name]}, found {actual}; use a fresh virtual environment')
                installed[name]=actual
            for name,wanted in pins.items():
                try:actual=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:continue
                if actual!=wanted:raise ValueError(f'{name}: expected constraint {wanted}, found {actual}')
        print(json.dumps(dict(status='passed',python=platform.python_version(),platform=platform.platform(),framework=framework,bundled_artifacts=len(provenance['bundled_files']),dependencies=installed),indent=2))
    except (ValueError,OSError,subprocess.CalledProcessError,importlib.metadata.PackageNotFoundError) as exc:
        print(f'Setup verification failed: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
