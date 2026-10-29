import gzip
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import torch
from fastapi.testclient import TestClient
from showcase.ava import world as W
from showcase.ava.model import AvaModel,canonical
from showcase.ava.session import AvaSession,BASE_DIR,state_hash

FIXTURE=json.loads(gzip.decompress((Path(__file__).parent/'fixtures/ava-browser.json.gz').read_bytes()))

def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()

class AvaParity(unittest.TestCase):
    def test_world_rules_encoding_teachers_and_training_histories(self):
        for item in FIXTURE['worlds']:
            w=item['world'];plan=W.teaching_plan(w)
            self.assertEqual(W.identity(w),item['identity']);self.assertEqual(W.features(w),item['features']);self.assertEqual(W.tokens(W.features(w)),item['tokens'])
            self.assertEqual(digest([[e['id'],e['label'],e['optimal'],e['cost']] for e in plan['examples']]),item['planDigest'])
            self.assertEqual(plan['trials'],item['trials']);self.assertEqual(plan['reachable'],item['reachable'])
            data,_=W.dataset(w,['layout'],True)
            self.assertEqual(len(data),item['train'])
            self.assertEqual(digest([[e['id'],W.tokens(e['features']),e['label'],e['memory']] for e in data]),item['dataDigest'])
            self.assertEqual([[e['id'],e['memory']] for e in W.shuffle(data,42+7919)[:12]],item['shuffle'])
        for item in FIXTURE.get('situations',[]):
            data,worlds=W.dataset(item['world'],item['selection'],True)
            self.assertEqual([W.identity(w) for w in worlds],item['worldIdentities'])
            self.assertEqual(len(data),item['train'])
            self.assertEqual(digest([[e['id'],W.tokens(e['features']),e['label'],e['memory']] for e in data]),item['digest'])
        for item in FIXTURE['observations']:
            self.assertEqual(W.observe(item['world']),item['observation'])
            for case in item['steps']:
                self.assertEqual(W.step(item['world'],case['action']),{k:v for k,v in case.items() if k!='action'})

    def test_browser_checkpoint_and_all_eight_stages(self):
        worst=0.
        for spec in FIXTURE['models']:
            model=AvaModel(spec['config'])
            if spec['source']=='trained-example':model.load_state_dict(torch.load(BASE_DIR/'base.pt',weights_only=True))
            before=state_hash(model)
            for case in spec['cases']:
                actual=model.inspect(case['features'],case['memory'])
                self.assertEqual(actual['tokens'],case['tokens']);self.assertEqual(actual['memoryReads'],case['memoryReads'])
                torch.testing.assert_close(torch.tensor(actual['vocabLogits']),torch.tensor(case['logits']),atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(torch.tensor(actual['probabilities']),torch.tensor(case['probabilities']),atol=2e-6,rtol=1e-5)
                for a,b in zip(actual['stages'],case['stages']):
                    torch.testing.assert_close(torch.tensor(a['state'][0]),torch.tensor(b['state']),atol=1e-5,rtol=1e-5)
                worst=max(worst,max(abs(a-b) for a,b in zip(actual['vocabLogits'],case['logits'])))
            self.assertEqual(before,state_hash(model))
        print('Ava browser/Python maximum logit error:',worst)

    def test_browser_gradient_update_and_all_stage_gradients(self):
        spec=FIXTURE['models'][1];update=spec['update'];model=AvaModel(spec['config']);examples=update['examples']
        initial={n:p.clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
        ids=torch.tensor([W.tokens(e['features']) for e in examples]);target=torch.tensor([96+e['label'] for e in examples])
        loss=torch.nn.functional.cross_entropy(model(ids,[e['memory'] for e in examples])[:,-1],target);loss.backward()
        self.assertAlmostEqual(float(loss.detach()),update['loss'],places=5)
        for name,param in model.browser_parameters().items():
            torch.testing.assert_close(param.grad,torch.tensor(update['gradients'][name]).reshape(param.shape),atol=1e-6,rtol=1e-4)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for b,stage in zip(model.config['blocks'],model.stages):
            self.assertTrue(any(not torch.equal(initial['stages.'+str(i)+'.'+n],p) for i,s in enumerate(model.stages) if s is stage for n,p in s.named_parameters()),b['id'])
        for name,p in model.browser_parameters().items():
            expected=update['weights'][name]
            torch.testing.assert_close(p,torch.tensor(expected['values']).reshape(expected['shape']),atol=3e-6,rtol=1e-4)

    def test_empty_memory_causality_and_order(self):
        spec=FIXTURE['models'][1];m=AvaModel(spec['config']);f=W.features(W.preset());t=torch.tensor([W.tokens(f)])
        a=m(t,[[]]);changed=t.clone();changed[0,-1]=5;b=m(changed,[[]])
        torch.testing.assert_close(a[:,:-1],b[:,:-1],atol=0,rtol=0)
        with_memory=m(t,[[W.record(f,0)]])
        self.assertFalse(torch.equal(a,with_memory))
        for stage in m.stages:
            if stage.spec['type']=='memory':
                x=torch.randn(1,t.shape[1],m.config['width']);bank=torch.full((1,8),128)
                y,_=stage(x,t,m.token_embedding,m.position_embedding,bank)
                self.assertTrue(torch.equal(x,y))
        records=[[32,40,48,96],[32,40,48,97],[33,40,48,98]]
        self.assertEqual(W.memory(records,'archive',2),records[0]+records[2]);self.assertEqual(W.memory(records,'recent',2),records[1]+records[2])

    def test_depth_only_matches_canonical_transformer(self):
        model=AvaModel(FIXTURE['models'][3]['config'])
        reference=canonical.Transformer(canonical.ModelConfig(n_layers=1,d_model=16,n_heads=4,d_ff=64))
        reference.load_state_dict({name:p.detach().clone() for name,p in model.browser_parameters().items()})
        ids=torch.tensor([W.tokens(W.features(W.preset()))])
        torch.testing.assert_close(model(ids),reference(ids),atol=1e-7,rtol=1e-6)

    def test_real_world_inventory_and_impossible_cycles(self):
        w=W.preset('empty',True);w['tiles']=[['wall']*16 for _ in range(16)]
        for x in range(1,9):w['tiles'][2][x]='floor'
        w.update(agent=dict(x=1,y=2,direction=1,keys=0),goal=[8,2]);w['tiles'][2][8]='lamp';w['tiles'][2][2]='key-red';w['tiles'][2][4]='door-red'
        self.assertTrue(W.teaching_plan(w)['reachable'])
        for _ in range(3):w=W.step(w,'forward')['world']
        self.assertEqual(w['agent']['keys'],1);self.assertEqual(w['agent']['x'],4)
        w['tiles'][2][5]='door-blue';self.assertEqual(W.step(w,'forward')['events'][0]['kind'],'blocked')
        initial=W.fresh(w,dict(x=1,y=2,direction=1,keys=0));initial['tiles'][2][2]='floor';initial['tiles'][2][6]='key-red'
        self.assertFalse(W.teaching_plan(initial)['reachable'])

class AvaApplication(unittest.TestCase):
    def tiny(self,d):return AvaSession(d,config=FIXTURE['models'][1]['config'])

    def test_optimizer_resume_state_isolation_and_no_teacher_play(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.tiny(d);data,_=W.dataset(s.layout,['layout'],True);s.selection=['layout'];s.examples=data;s.example_count=len(data)
            before=state_hash(s.model);s.update(data[:4]);s.save();self.assertNotEqual(before,state_hash(s.model))
            restored=AvaSession(d);self.assertTrue(restored.optimizer.state)
            s.update(data[4:8]);restored.update(data[4:8]);self.assertEqual(state_hash(s.model),state_hash(restored.model));self.assertEqual(s.last_loss,restored.last_loss)
            checkpoint=restored.path.read_bytes();before=state_hash(restored.model)
            from unittest.mock import patch
            with patch.object(W,'teaching_plan',return_value={'reachable':True,'starts':[]}),patch.object(W,'dataset',side_effect=AssertionError('teacher used to choose action')):
                direct=restored.model.inspect(W.features(restored.world),[])
                report=restored.action()
            self.assertEqual(report['last_decision']['action'],W.ACTIONS[direct['actionIndex']]);self.assertEqual(report['state_sha256'],before)
            self.assertEqual(len(restored.memory),1);restored.retry();self.assertEqual(restored.memory,[]);self.assertEqual(state_hash(restored.model),before)
            loaded=AvaSession(d);self.assertEqual(loaded.offset,8);self.assertEqual(loaded.example_count,len(data));self.assertEqual(loaded.selection,['layout'])

    def test_world_edit_reset_and_developer_configuration_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            s=self.tiny(d);before=state_hash(s.model)
            s.apply_layout(W.edit(s.layout,4,4,'wall'));self.assertEqual(state_hash(s.model),before)
            self.assertTrue(s.snapshot()['world_changed']);self.assertEqual(s.memory,[])
            s.start(dict(epochs=1,rate=.0005,seed=42),['layout'],False)
            self.assertEqual(s.epoch,1);self.assertIsNone(s.error);self.assertEqual(s.qualification['total'],len(W.teaching_plan(s.layout)['starts']))
            before=state_hash(s.model);s.apply_layout(W.expand(s.layout));self.assertNotEqual(state_hash(s.model),before);self.assertEqual(s.steps,0);self.assertEqual(len(W.tokens(W.features(s.world))),104)
            with self.assertRaises(ValueError):AvaSession(d,config=FIXTURE['models'][3]['config'])

    def test_http_world_model_checkpoint_and_training_pause(self):
        import time
        from showcase.server import create_app
        with tempfile.TemporaryDirectory() as d:
            app=create_app(Path(d)/'chat',Path(d)/'devices',Path(d)/'ava');s=app.state.sessions['ava']
            with TestClient(app) as c:
                chat=c.get('/api/status').json()['state_sha256'];devices=c.get('/api/devices/status').json()['state_sha256']
                self.assertEqual(c.get('/ava.html').status_code,200)
                before=c.get('/api/ava/status').json()['state_sha256']
                self.assertEqual(c.post('/api/ava/step').status_code,200)
                self.assertEqual(c.get('/api/ava/status').json()['state_sha256'],before)
                self.assertEqual(c.post('/api/ava/retry').json()['memory'],[])
                settings=dict(epochs=1,rate=.0005,seed=42)
                self.assertEqual(c.post('/api/ava/train',json={'settings':settings,'selection':['layout']}).status_code,200)
                c.post('/api/ava/pause')
                deadline=time.monotonic()+20
                while c.get('/api/ava/status').json()['busy'] and time.monotonic()<deadline:time.sleep(.02)
                status=c.get('/api/ava/status').json();self.assertFalse(status['busy']);self.assertIsNone(status['error']);self.assertGreater(status['steps'],0)
                saved=AvaSession(Path(d)/'ava');self.assertEqual(saved.offset,status['done']);self.assertEqual(state_hash(saved.model),status['state_sha256'])
                response=c.get('/api/ava/checkpoint');self.assertEqual(response.status_code,200);self.assertIn('ava.pt',response.headers['content-disposition'])
                self.assertEqual(response.content,s.path.read_bytes())
                self.assertEqual(chat,c.get('/api/status').json()['state_sha256']);self.assertEqual(devices,c.get('/api/devices/status').json()['state_sha256'])
                self.assertEqual(c.post('/api/ava/world',json={'preset':'locks','large':True}).status_code,200)
                self.assertEqual(c.post('/api/ava/world',json={'x':2,'y':2,'tool':'key-red'}).status_code,200)
                self.assertEqual(c.post('/api/ava/train',json={'settings':settings,'selection':[]}).status_code,400)
