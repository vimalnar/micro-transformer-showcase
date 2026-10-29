import tempfile
import time
import unittest
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from showcase.core import Session, BASE, ARCH, rows
from showcase.server import create_app
from training.harness import InferenceHarness
from training.data import encode
from harness.talk_to_micro_transformer import VERIFIED_EXAMPLES, KNOWN_LIMITATIONS

RECIPE = dict(text='ava move key one to lab. where key one? => lab |',rounds=3,rate=.0001)

class Workflow(unittest.TestCase):
    def test_baseline_logits_predictions_and_immutable_inference(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session(d)
            reference = InferenceHarness(BASE,device='cpu',architecture=ARCH)
            before = s.harness.state_sha256()
            for _, prompt, expected in VERIFIED_EXAMPLES:
                self.assertEqual(s.predict(prompt)['prediction'], expected)
                x = torch.tensor([encode(prompt)])
                with torch.inference_mode():
                    torch.testing.assert_close(s.model(x),reference.model(x),atol=1e-7,rtol=1e-6)
            for _,prompt,_,expected in KNOWN_LIMITATIONS:
                self.assertEqual(s.predict(prompt)['prediction'],expected)
            self.assertEqual(before,s.harness.state_sha256())

    def test_exact_optimizer_resume_and_trained_playground(self):
        with tempfile.TemporaryDirectory() as d:
            s = Session(d)
            before = s.harness.state_sha256()
            s.recipe = RECIPE.copy(); s.draft = RECIPE.copy()
            s.update(rows(RECIPE['text'])[0]); s.save()
            self.assertNotEqual(before,s.harness.state_sha256())
            saved = Session(d)
            self.assertTrue(saved.optimizer.state)
            self.assertEqual(saved.offset,1)
            self.assertEqual(saved.recipe,RECIPE)
            s.update(rows(RECIPE['text'])[0])
            saved.update(rows(RECIPE['text'])[0])
            self.assertEqual(s.harness.state_sha256(),saved.harness.state_sha256())
            self.assertEqual(s.last_loss,saved.last_loss)
            saved.save()
            client = TestClient(create_app(d))
            expected = saved.predict('ava move key one to lab. where key one?')
            actual = client.post('/api/predict',json={'prompt':expected['prompt']}).json()
            self.assertEqual(expected['token_ids'],actual['token_ids'])
            self.assertEqual(expected['state_sha256'],actual['state_sha256'])
            self.assertEqual(client.get('/api/status').json()['state_sha256'],expected['state_sha256'])

    def test_http_training_pause_restart_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            with TestClient(create_app(d)) as c:
                self.assertEqual(c.get('/').status_code,200)
                self.assertIn('app.js',c.get('/').text)
                self.assertEqual(c.post('/api/predict',json={'prompt':'hello'}).status_code,400)
                self.assertEqual(c.post('/api/predict',json={'prompt':None}).status_code,400)
                self.assertEqual(c.post('/api/train',json=dict(RECIPE,text=None)).status_code,400)
                self.assertEqual(c.put('/api/draft',json=dict(RECIPE,rate=0)).status_code,400)
                self.assertEqual(c.post('/api/train',json=dict(RECIPE,text='bad')).status_code,400)
                self.assertEqual(c.post('/api/train',headers={'origin':'https://other.example'},json=RECIPE).status_code,403)
                long = dict(RECIPE,rounds=20)
                self.assertEqual(c.post('/api/train',json=long).status_code,200)
                c.post('/api/pause')
                deadline = time.monotonic()+10
                while c.get('/api/status').json()['busy'] and time.monotonic()<deadline:
                    time.sleep(.02)
                paused = c.get('/api/status').json()
                self.assertFalse(paused['busy'])
                self.assertLess(paused['done'],20)
                self.assertEqual(c.get('/api/checkpoint').status_code,200)
            with TestClient(create_app(d)) as c:
                self.assertEqual(c.get('/api/status').json()['done'],paused['done'])
                c.post('/api/train',json=long)
                deadline = time.monotonic()+20
                while c.get('/api/status').json()['busy'] and time.monotonic()<deadline:
                    time.sleep(.03)
                final = c.get('/api/status').json()
                self.assertIsNone(final['error'])
                self.assertEqual(final['done'],20)
                self.assertEqual(final['steps'],20)
                before = final['state_sha256']
                c.post('/api/predict',json={'prompt':'ava move key one to lab. where key one?'})
                self.assertEqual(c.get('/api/status').json()['state_sha256'],before)

if __name__ == '__main__': unittest.main()
