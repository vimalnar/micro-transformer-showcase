"""Obtain the pinned framework as a separate checkout; never reset an existing one."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE='https://github.com/vimalnar/micro-transformer.git'


def verify(directory):
    lock=json.loads((ROOT/'framework-lock.json').read_text())
    directory=Path(directory).expanduser().resolve()
    if not (directory/'.git').exists():
        raise ValueError(f'Framework checkout not found at {directory}. Run python scripts/setup_framework.py first, or set MICRO_TRANSFORMER_ROOT to a verified checkout.')
    revision=subprocess.run(['git','-C',str(directory),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if revision!=lock['revision']:
        raise ValueError(f'Framework revision mismatch: expected {lock["revision"]}, found {revision}. Use a separate checkout; existing files were not changed.')
    for name,expected in lock['files'].items():
        path=directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Framework file/checkpoint hash mismatch: '+name)
    return dict(path=str(directory),revision=revision,verified_files=len(lock['files']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=ROOT.parent/'micro-transformer')
    parser.add_argument('--source',default=SOURCE,help='Canonical repository, or an existing local Git repository for offline setup')
    args=parser.parse_args();target=args.destination.expanduser().resolve()
    try:
        if not target.exists():
            lock=json.loads((ROOT/'framework-lock.json').read_text())
            subprocess.run(['git','clone','--no-checkout','--',args.source,str(target)],check=True)
            subprocess.run(['git','-C',str(target),'checkout','--detach',lock['revision']],check=True)
        print(json.dumps(verify(target),indent=2))
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'Framework setup failed: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
