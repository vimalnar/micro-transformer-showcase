"""Standalone Ava training, prediction, world stepping and route checks."""
import argparse,json
from pathlib import Path
from .session import AvaSession
from .world import features
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path);p.add_argument('--config',type=Path,help='Developer architecture; use a new state directory')
commands=p.add_subparsers(dest='command',required=True)
for name in ('info','step','retry','predict','check','examples'):commands.add_parser(name)
train=commands.add_parser('train');train.add_argument('--epochs',type=int);train.add_argument('--rate',type=float);train.add_argument('--seed',type=int);train.add_argument('--situations',nargs='+')
a=p.parse_args();s=AvaSession(a.state,json.loads(a.config.read_text()) if a.config else None)
if a.command=='train':
    settings=dict(s.settings)
    for key in ('epochs','rate','seed'):
        if getattr(a,key) is not None:settings[key]=getattr(a,key)
    try:s.start(settings,a.situations or s.selection,False)
    except KeyboardInterrupt:s.pause.set();s.busy=False;s.save()
    if s.error:raise SystemExit(s.error)
    result=s.snapshot()
elif a.command=='step':result=s.action()
elif a.command=='retry':result=s.retry()
elif a.command=='predict':result=s.model.inspect(features(s.world),s.memory)
elif a.command=='check':result=s.rollouts()
elif a.command=='examples':
    from .world import dataset
    data,worlds=dataset(s.layout,s.selection,s.model.has_memory);result=dict(examples=len(data),worlds=len(worlds))
else:result=s.snapshot()
print(json.dumps(result,indent=2))
