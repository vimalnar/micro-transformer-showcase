"""Convert the supplied Browser Lab Ava example without training or modifying references."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from showcase.core import ROOT,LOCK
from showcase.ava.model import AvaModel

browser=Path(os.environ.get('BROWSER_LAB_ROOT',ROOT.parent/'micro-transformer-browser-lab'))
source=browser/'public/trained-example.json';r=json.loads(source.read_text())
if r['pythonSource']['sha256']!=LOCK['files']['models/transformer.py']:raise ValueError('Canonical transformer identity differs')
if r.get('optimizerState'):raise ValueError('This converter handles the supplied weights-only example, not browser optimizer migration')
model=AvaModel(r['config']);model.load_browser(r['weights'])
meta={k:r.get(k) for k in ['config','world','editorWorld','settings','trainingSituations','demoExample','epoch','optimizerSteps','pythonSource','taskAdapter']}
meta.update(checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),optimizer_available=False)
output=ROOT/'data/ava';output.mkdir(exist_ok=True)
(output/'base.json').write_text(json.dumps(meta,indent=2)+'\n');torch.save(model.state_dict(),output/'base.pt')
files=['src/python-transformer.js','src/python-transformer-config.js','src/causal-map-agent.js','src/capability-kit.js','src/map-token-adapter.js','src/editable-world.js','src/world-objects.js','src/micro-world.js','src/navigation-training.js','src/training-situations.js','src/room-qualification.js','src/agent-lab.js','public/trained-example.json']
provenance=dict(browser_revision=subprocess.check_output(['git','-C',str(browser),'rev-parse','HEAD'],text=True).strip(),browser_files={n:hashlib.sha256((browser/n).read_bytes()).hexdigest() for n in files},converted_files={n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in ('base.json','base.pt')},scope='Weights-only faithful conversion; no training; optimizer starts fresh; canonical source lock remains unchanged.')
(output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'source_sha256':meta['checkpoint_sha256'],'parameters':sum(p.numel() for p in model.parameters()),'converted_sha256':provenance['converted_files']['base.pt']},indent=2))
