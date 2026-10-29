import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from showcase.core import Session, rows, TOKENS, LOCK
from showcase.devices import Simulation, packet, prompt, state, assess, practice, PLACES
from showcase.server import create_app
from training.data import encode
from micro_transformer.data.generator import derive_answer

BROWSER = Path(os.environ.get('BROWSER_LAB_ROOT',Path(__file__).resolve().parents[2]/'micro-transformer-browser-lab'))
RECIPE = dict(text=practice()['Missing updates'][0],rounds=20,rate=.0001)

class DeviceWorkflow(unittest.TestCase):
    def test_adapter_browser_parity_and_canonical_labels(self):
        for lines in practice().values():
            for line in lines:
                p,a = line.split('=>')
                self.assertEqual(derive_answer([TOKENS[i] for i in encode(p+a)]),a.strip().split()[:-1])
                self.assertEqual(len(rows(line)),1)
        # Compare exact strings, labels, and packet shape to original JS adapter.
        if os.environ.get('BROWSER_LAB_ROOT') and shutil.which('node'):
            self.assertTrue((BROWSER/'src/device-messaging.js').is_file(), 'BROWSER_LAB_ROOT must contain the audited working-tree source')
            script = f"import {{devicePractice,devicePacket,devicePrompt}} from '{BROWSER}/src/device-messaging.js'; console.log(JSON.stringify({{sets:devicePractice(),packet:devicePacket('river',false),six:devicePrompt(Array.from({{length:6}},()=>devicePacket('dock')))}}));"
            original = json.loads(subprocess.check_output(['node','--input-type=module','-e',script],text=True))
            self.assertEqual(practice(),original['sets'])
            self.assertEqual(packet('river',False),original['packet'])
            self.assertEqual(prompt([packet('dock')]*6),original['six'])
        for n in range(1,7):
            self.assertLessEqual(len(encode(prompt([packet('river')]*n))),128)
        with self.assertRaises(ValueError): prompt([packet('river')]*7)
        for value in ('true',1,None):
            with self.assertRaises(ValueError): packet('river',value)

    def test_predictions_numerics_and_no_weight_updates(self):
        fixtures=json.loads((Path(__file__).parent/'fixtures/device-language.json').read_text())
        self.assertEqual(fixtures['interpreterSha256'],LOCK['files']['micro-world/micro_transformer/data/generator.py'])
        with tempfile.TemporaryDirectory() as d:
            session=Session(d,scope='devices');before=session.harness.state_sha256()
            for case in fixtures['pythonPredictions']:
                self.assertEqual(session.predict(case['prompt'])['token_ids'],case['answer'])
                with torch.inference_mode():
                    logits=session.model(torch.tensor([encode(case['prompt'])]))[0,-1]
                torch.testing.assert_close(logits,torch.tensor(case['logits']),atol=1e-5,rtol=1e-5)
            self.assertEqual(before,session.harness.state_sha256())
            sim=Simulation(session)
            first=sim.send('river',True)
            self.assertEqual(first['result']['assessment']['report'],'river')
            withheld=sim.send('dock',False)
            self.assertEqual(withheld['actual'],'dock');self.assertEqual(withheld['received'],'river')
            self.assertEqual(withheld['result']['prediction'],'river |')
            self.assertEqual(before,session.harness.state_sha256())
            sim.clear()
            self.assertEqual(sim.send('dock',False)['result']['prediction'],'unknown |')
            # Deliberately wrong model output must remain wrong, never simulator-corrected.
            wrong=dict(prediction='lab |',token_ids=[TOKENS.index('lab'),TOKENS.index('|')],state_sha256=before,steps=0)
            with patch.object(session,'predict',return_value=wrong):
                result=sim.interpret()['result']
            self.assertEqual(result['prediction'],'lab |');self.assertFalse(result['assessment']['matches'])
            self.assertFalse(assess(sim.events,[TOKENS.index('yes'),TOKENS.index('|')])['valid'])
            self.assertFalse(assess(sim.events,[TOKENS.index('dock')])['valid'])

    def test_simulation_limits_failure_persistence_and_clear(self):
        with tempfile.TemporaryDirectory() as d:
            session=Session(d,scope='devices');sim=Simulation(session)
            with patch.object(session,'predict',side_effect=RuntimeError('diagnostic inference failure')):
                with self.assertRaises(RuntimeError):sim.send('lab',False)
            restored=Simulation(Session(d,scope='devices'))
            self.assertEqual(restored.snapshot()['actual'],'lab');self.assertIn('failure',restored.error)
            self.assertIsNone(restored.result)
            restored.interpret()
            for _ in range(5): restored.send('river',True)
            before=restored.snapshot()
            with self.assertRaises(ValueError):restored.send('dock',True)
            self.assertEqual(before,restored.snapshot())
            weights=session.harness.state_sha256()
            cleared=restored.clear();self.assertEqual(cleared['events'],[]);self.assertIsNone(cleared['result'])
            self.assertEqual(weights,restored.session.harness.state_sha256())

    def test_scope_isolation_and_exact_device_optimizer_restore(self):
        with tempfile.TemporaryDirectory() as d:
            chat=Session(Path(d)/'chat');chat.save();chat_bytes=chat.path.read_bytes()
            device=Session(Path(d)/'device',scope='devices')
            device.recipe=RECIPE.copy();device.draft=RECIPE.copy()
            before=device.harness.state_sha256()
            device.update(rows(RECIPE['text'])[0]);device.save()
            self.assertNotEqual(before,device.harness.state_sha256())
            restored=Session(device.directory,scope='devices')
            device.update(rows(RECIPE['text'])[0]);restored.update(rows(RECIPE['text'])[0])
            self.assertEqual(device.harness.state_sha256(),restored.harness.state_sha256())
            self.assertEqual(device.last_loss,restored.last_loss)
            self.assertEqual(chat_bytes,chat.path.read_bytes())
            with self.assertRaises(ValueError):Session(device.directory,scope='chatbot')
            with self.assertRaises(ValueError):Session(chat.directory,scope='devices')

    def test_device_http_training_pause_resume_stale_and_download(self):
        with tempfile.TemporaryDirectory() as d:
            app=create_app(Path(d)/'chat',Path(d)/'device')
            with TestClient(app) as c:
                chat_before=c.get('/api/status').json()
                self.assertEqual(c.get('/devices.html').status_code,200)
                self.assertEqual(c.get('/api/devices/practice').json(),practice())
                self.assertEqual(c.post('/api/devices/events',json={'place':'dock','delivered':'false'}).status_code,400)
                with patch.object(app.state.sessions['devices'],'predict',side_effect=RuntimeError('diagnostic failure')):
                    failure=c.post('/api/devices/events',json={'place':'dock','delivered':False})
                self.assertEqual(failure.status_code,500)
                self.assertIn('diagnostic failure',failure.json()['detail'])
                self.assertEqual(len(c.get('/api/devices/simulation').json()['events']),1)
                c.post('/api/devices/clear')
                c.post('/api/devices/events',json={'place':'river','delivered':True})
                old=c.get('/api/devices/simulation').json()['result']['state_sha256']
                self.assertEqual(c.post('/api/devices/train',json=RECIPE).status_code,200)
                c.post('/api/devices/pause')
                until=time.monotonic()+10
                while c.get('/api/devices/status').json()['busy'] and time.monotonic()<until:time.sleep(.02)
                paused=c.get('/api/devices/status').json()
                self.assertFalse(paused['busy']);self.assertLess(paused['done'],20)
                self.assertGreater(paused['done'],0)
                self.assertTrue(c.get('/api/devices/simulation').json()['stale'])
                downloaded=c.get('/api/devices/checkpoint')
                self.assertIn('devices.pt',downloaded.headers['content-disposition'])
                self.assertEqual(downloaded.content,(Path(d)/'device/current.pt').read_bytes())
            with TestClient(create_app(Path(d)/'chat',Path(d)/'device')) as c:
                self.assertEqual(c.get('/api/devices/status').json()['done'],paused['done'])
                self.assertEqual(len(c.get('/api/devices/simulation').json()['events']),1)
                c.post('/api/devices/train',json=RECIPE)
                until=time.monotonic()+15
                while c.get('/api/devices/status').json()['busy'] and time.monotonic()<until:time.sleep(.03)
                final=c.get('/api/devices/status').json()
                self.assertEqual(final['steps'],20);self.assertIsNone(final['error'])
                self.assertNotEqual(old,final['state_sha256'])
                result=c.post('/api/devices/interpret').json()
                self.assertFalse(result['stale'])
                self.assertEqual(result['result']['state_sha256'],final['state_sha256'])
                self.assertEqual(result['result']['token_ids'],c.post('/api/devices/predict',json={'prompt':result['prompt']}).json()['token_ids'])
                self.assertEqual(c.get('/api/devices/status').json()['state_sha256'],final['state_sha256'])
                chat_after=c.get('/api/status').json()
                for key in ('state_sha256','steps','draft','recipe'):self.assertEqual(chat_before[key],chat_after[key])
                self.assertEqual(c.post('/api/devices/clear').json()['events'],[])

if __name__=='__main__':unittest.main()
