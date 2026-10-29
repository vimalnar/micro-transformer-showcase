import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from fastapi.testclient import TestClient
from showcase.core import Session
from showcase.server import create_app

class NumericalFailures(unittest.TestCase):
    def test_invalid_logits_are_not_decoded_as_answers(self):
        with tempfile.TemporaryDirectory() as d:
            app=create_app(Path(d)/'chat',Path(d)/'devices',Path(d)/'ava')
            with TestClient(app) as client:
                with torch.no_grad():app.state.sessions['chatbot'].model.token_embedding.weight[0,0]=float('nan')
                response=client.post('/api/predict',json={'prompt':'hal move flag six to river. where flag six?'})
                self.assertEqual(response.status_code,400)
                self.assertIn('Non-finite',response.json()['detail'])
                with torch.no_grad():app.state.sessions['devices'].model.token_embedding.weight[0,0]=float('nan')
                response=client.post('/api/devices/events',json={'place':'river','delivered':True})
                self.assertEqual(response.status_code,400)
                simulation=client.get('/api/devices/simulation').json()
                self.assertEqual(len(simulation['events']),1)
                self.assertIsNone(simulation['result'])
                self.assertIn('Non-finite',simulation['error'])

    def test_invalid_checkpoint_is_rejected_without_replacing_it(self):
        for scope in ('chatbot','devices'):
            with self.subTest(scope=scope),tempfile.TemporaryDirectory() as d:
                session=Session(d,scope=scope);session.save()
                payload=torch.load(session.path,weights_only=True)
                payload['model']['token_embedding.weight'][0,0]=float('nan')
                torch.save(payload,session.path)
                before=session.path.read_bytes()
                with self.assertRaisesRegex(ValueError,'Non-finite checkpoint'):
                    Session(d,scope=scope)
                self.assertEqual(before,session.path.read_bytes())

    def test_invalid_saved_optimizer_is_rejected(self):
        from showcase.core import rows
        with tempfile.TemporaryDirectory() as d:
            session=Session(d)
            session.update(rows('ava move key one to lab. where key one? => lab |')[0])
            session.save()
            payload=torch.load(session.path,weights_only=True)
            state=next(iter(payload['optimizer']['state'].values()))
            state['exp_avg'].flatten()[0]=float('nan')
            torch.save(payload,session.path)
            before=session.path.read_bytes()
            with self.assertRaisesRegex(ValueError,'Non-finite checkpoint optimizer'):
                Session(d)
            self.assertEqual(before,session.path.read_bytes())
