"""Real local HTTP-server/restart review; exclusively temporary state, no publishing.

Checks downloaded checkpoints in a new process and compares resumed training
against a separately restored uninterrupted continuation, including all Adam state.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from showcase.core import Session,rows
from showcase.ava.session import AvaSession,state_hash
from showcase.ava import world as W


def identical(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


class Service:
    def __init__(self,folder):
        self.folder=Path(folder);self.process=None
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1]
        self.url=f'http://127.0.0.1:{self.port}'

    def start(self,directories):
        self.log=(self.folder/'server.log').open('a')
        command=[sys.executable,'-m','showcase','--state',str(directories['chatbot']),'serve','--device-state',str(directories['devices']),'--ava-state',str(directories['ava']),'--port',str(self.port)]
        self.process=subprocess.Popen(command,cwd=ROOT,stdout=self.log,stderr=subprocess.STDOUT)
        until=time.monotonic()+30
        while time.monotonic()<until:
            if self.process.poll() is not None:raise RuntimeError((self.folder/'server.log').read_text())
            try:self.request('/api/status');return
            except (urllib.error.URLError,TimeoutError):time.sleep(.05)
        raise RuntimeError('Local server startup timed out')

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait();raise
            self.process=None;self.log.close()

    def request(self,path,body=None,raw=False):
        req=urllib.request.Request(self.url+path,data=None if body is None else json.dumps(body).encode(),headers={} if body is None else {'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response:
            value=response.read()
            return value if raw else json.loads(value)

    def idle(self,prefix,timeout=60):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            value=self.request(prefix+'/status')
            if not value['busy']:
                assert value['error'] is None,value['error']
                return value
            time.sleep(.03)
        raise RuntimeError('Training failed to finish within review budget')


def review():
    report={'format':'showcase-local-end-to-end-review-v1','scope':'Real CPU training and HTTP-server restart, isolated temporary state; no generalisation claim','showcases':{}}
    with tempfile.TemporaryDirectory(prefix='micro-transformer-full-review-') as temporary:
        folder=Path(temporary);service=Service(folder)
        original={name:folder/'original'/name for name in ('chatbot','devices','ava')}
        restored={name:folder/'restored'/name for name in original}
        expected={name:folder/'expected'/name for name in original}
        prefixes={'chatbot':'/api','devices':'/api/devices','ava':'/api/ava'}
        try:
            service.start(original)
            # Check served HTML, JavaScript and styles as well as inference APIs.
            for path in ('/','/devices.html','/ava.html','/app.js','/ava.js','/style.css'):
                assert len(service.request(path,raw=True))>100,path
            initial={name:service.request(prefix+'/status') for name,prefix in prefixes.items()}
            first=service.request('/api/devices/events',{'place':'river','delivered':True})
            withheld=service.request('/api/devices/events',{'place':'dock','delivered':False})
            assert first['result']['prediction']=='river |'
            assert withheld['result']['prediction']=='river |' and withheld['actual']=='dock'
            chatbot_recipe=dict(text='\n'.join(['ava move key one to lab. where key one? => lab |']*8),rounds=5,rate=.0001)
            device_recipe=dict(text='\n'.join(service.request('/api/devices/practice')['Missing updates']),rounds=5,rate=.0001)
            recipes={'chatbot':chatbot_recipe,'devices':device_recipe,'ava':dict(settings=dict(epochs=1,seed=42,rate=.0005),selection=['layout','lamps'])}
            paused={};payloads={}
            for name,prefix in prefixes.items():
                service.request(prefix+'/train',recipes[name])
                until=time.monotonic()+15
                while time.monotonic()<until:
                    current=service.request(prefix+'/status')
                    if current['steps']>0:break
                    time.sleep(.02)
                service.request(prefix+'/pause',{})
                paused[name]=service.idle(prefix)
                assert paused[name]['steps']>0
                assert paused[name]['state_sha256']!=initial[name]['state_sha256']
                if name=='ava':assert paused[name]['epoch']==0 and 0<paused[name]['done']<paused[name]['total']
                else:assert 0<paused[name]['done']<paused[name]['total']
                downloaded=service.request(prefix+'/checkpoint',raw=True)
                payloads[name]=torch.load(io.BytesIO(downloaded),weights_only=True)
                assert payloads[name]['showcase']==name
                assert payloads[name]['optimizer']['state']
                for destination in (restored[name],expected[name]):
                    destination.mkdir(parents=True,exist_ok=True);(destination/'current.pt').write_bytes(downloaded)
                report['showcases'][name]=dict(paused_steps=paused[name]['steps'],paused_offset=paused[name]['done'],paused_state_sha256=paused[name]['state_sha256'],download_sha256=hashlib.sha256(downloaded).hexdigest())
                print(name,'paused at',paused[name]['steps'],'actual updates',flush=True)
            simulation=original['devices']/'simulation.json'
            shutil.copy2(simulation,restored['devices']/'simulation.json')
            service.stop()
            # New server process, only downloaded checkpoint files plus simulation.
            service.start(restored)
            for name,prefix in prefixes.items():
                value=service.request(prefix+'/status')
                assert value['state_sha256']==paused[name]['state_sha256']
                assert value['steps']==paused[name]['steps'] and value['done']==paused[name]['done']
                report['showcases'][name]['process_restore_exact']=True
            assert len(service.request('/api/devices/simulation')['events'])==2
            # Continue each restored model independently, then compare with the
            # real HTTP trainer's continuation. Different directories/processes.
            final={};offline_models={}
            for name,prefix in prefixes.items():
                if name=='ava':
                    offline=AvaSession(expected[name]);offline.start(**recipes[name],background=False)
                    expected_hash=state_hash(offline.model)
                else:
                    offline=Session(expected[name],scope=name);offline.start(recipes[name],background=False)
                    expected_hash=offline.harness.state_sha256()
                assert offline.error is None,offline.error
                service.request(prefix+'/train',recipes[name]);final[name]=service.idle(prefix)
                assert final[name]['state_sha256']==expected_hash,(name,'resumed weights differ')
                actual=torch.load(io.BytesIO(service.request(prefix+'/checkpoint',raw=True)),weights_only=True)
                assert identical(actual['optimizer'],offline.optimizer.state_dict()),(name,'optimizer restoration differs')
                assert final[name]['steps']==offline.steps
                if name=='ava':assert final[name]['epoch']==1
                else:assert final[name]['done']==final[name]['total']
                offline_models[name]=offline
                report['showcases'][name].update(final_steps=final[name]['steps'],learned_state_sha256=expected_hash,continuation_weights_exact=True,continuation_optimizer_exact=True)
                print(name,'resumed weights and optimizer match independent continuation exactly',flush=True)
            # Playground replies must use the selected trained weights, and must
            # leave them unchanged. Validate actual raw tokens/probabilities.
            query='ava move key one to lab. where key one?'
            chat=service.request('/api/predict',{'prompt':query})
            direct=offline_models['chatbot'].predict(query)
            assert chat['token_ids']==direct['token_ids']
            assert chat['state_sha256']==final['chatbot']['state_sha256']
            devices=service.request('/api/devices/interpret',{})
            direct=offline_models['devices'].predict(devices['prompt'])
            assert devices['result']['token_ids']==direct['token_ids']
            assert devices['result']['state_sha256']==final['devices']['state_sha256'] and not devices['stale']
            ava=service.request('/api/ava/status')
            trace=offline_models['ava'].model.inspect(W.features(ava['world']),ava['memory'])
            moved=service.request('/api/ava/step',{})
            assert moved['last_decision']['trace']['probabilities']==trace['probabilities']
            assert moved['last_decision']['state_sha256']==final['ava']['state_sha256']
            for name,prefix in prefixes.items():
                assert service.request(prefix+'/status')['state_sha256']==final[name]['state_sha256']
                report['showcases'][name].update(playground_matches_restored_model=True,inference_weights_unchanged=True)
            report['showcases']['chatbot']['observed_prediction']=chat['prediction']
            report['showcases']['devices']['observed_prediction']=devices['result']['prediction']
            report['showcases']['ava']['observed_action']=moved['last_decision']['action']
            report['status']='passed'
        finally:service.stop()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path);args=parser.parse_args()
    result=review()
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
