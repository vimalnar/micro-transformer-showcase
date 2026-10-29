"""Local Ava application: authoritative world, model, optimizer and actual rollouts."""
import copy
import hashlib
import json
import threading
from pathlib import Path
import torch
from ..core import ROOT, LOCK
from .model import AvaModel
from . import world as W

BASE_DIR=ROOT/'data/ava'
PROVENANCE=json.loads((BASE_DIR/'provenance.json').read_text())
for filename,expected in PROVENANCE['converted_files'].items():
    if hashlib.sha256((BASE_DIR/filename).read_bytes()).hexdigest()!=expected:raise RuntimeError('Ava converted checkpoint identity changed: '+filename)
META=json.loads((BASE_DIR/'base.json').read_text())
IMPLEMENTATION={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('world.py','model.py','session.py')}
BASE_HASH=hashlib.sha256((BASE_DIR/'base.pt').read_bytes()).hexdigest()


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}\n'.encode());digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


class AvaSession:
    def __init__(self,directory=None,config=None):
        self.directory=Path(directory or ROOT/'state/ava');self.directory.mkdir(parents=True,exist_ok=True)
        self.path=self.directory/'current.pt';self.lock=threading.RLock();self.pause=threading.Event();self.busy=False;self.phase='ready';self.error=None
        self.config=copy.deepcopy(config or META['config']);self.settings=copy.deepcopy(META['settings']);self.selection=list(META['trainingSituations'])
        self.layout=W.fresh(META['editorWorld']);self.training_identity=W.identity(self.layout)
        self.epoch=self.offset=self.steps=0;self.round_loss=0.;self.last_loss=None;self.history=[];self.qualification=None;self.check_progress=None;self.origin='Supplied browser-trained weights · 50 historical rounds; fresh optimizer'
        self.model=AvaModel(self.config,self.settings['seed'])
        self.example_count=None
        if config is not None:self.origin='Fresh developer-configured weights'
        if config is None:self.model.load_state_dict(torch.load(BASE_DIR/'base.pt',weights_only=True,map_location='cpu'))
        self.optimizer=self.new_optimizer();self.examples=None;self.training_worlds=[];self._plan_key=None;self._plan=None
        self.rewind()
        if self.path.exists():
            saved=torch.load(self.path,weights_only=True,map_location='cpu')
            if saved['showcase']!='ava' or saved['framework']!=LOCK or saved['base_sha256']!=META['checkpoint_sha256'] or saved.get('implementation')!=IMPLEMENTATION or saved.get('base_tensor_sha256')!=BASE_HASH:raise ValueError('Ava checkpoint provenance mismatch')
            if config is not None and config!=saved['config']:raise ValueError('Use a new state directory for a changed architecture')
            self.config=saved['config'];self.model=AvaModel(self.config,saved['settings']['seed']);self.model.load_state_dict(saved['model']);self.optimizer=self.new_optimizer();self.optimizer.load_state_dict(saved['optimizer'])
            for key in self.saved_fields():setattr(self,key,saved[key])

    def plan(self):
        key=(W.identity(self.layout),W.pose(self.layout['agent']))
        if key!=self._plan_key:self._plan=W.teaching_plan(self.layout);self._plan_key=key
        return self._plan

    def new_optimizer(self):return torch.optim.AdamW(self.model.parameters(),lr=self.settings['rate'],betas=(.9,.999),eps=1e-8,weight_decay=.01)

    @staticmethod
    def saved_fields():return ('config','settings','selection','layout','training_identity','epoch','offset','steps','round_loss','last_loss','history','qualification','origin','world','memory','decisions','visits','repeated','play_agent','error','example_count')

    def save(self):
        payload={key:getattr(self,key) for key in self.saved_fields()}
        payload.update(showcase='ava',framework=LOCK,implementation=IMPLEMENTATION,base_tensor_sha256=BASE_HASH,base_sha256=META['checkpoint_sha256'],model=self.model.state_dict(),optimizer=self.optimizer.state_dict())
        temp=self.path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(self.path)

    def rewind(self,agent=None):
        self.play_agent=copy.deepcopy(agent or self.layout['agent']);self.world=W.fresh(self.layout,self.play_agent);self.memory=[];self.decisions=[];self.visits={};self.repeated=False

    def idle(self):
        if self.busy:raise ValueError('Pause training or journey checks before changing or playing')

    def reset(self):
        self.model=AvaModel(self.config,self.settings['seed']);self.optimizer=self.new_optimizer();self.epoch=self.offset=self.steps=0;self.round_loss=0.;self.last_loss=None;self.history=[];self.qualification=None;self.origin='Fresh weights; current selected situations';self.training_identity=W.identity(self.layout);self.examples=None;self.example_count=None;self.error=None;self.rewind()

    def snapshot(self):
        with self.lock:
            digest=state_hash(self.model);plan=self.plan()
            return dict(showcase='ava',busy=self.busy,phase=self.phase,error=self.error,config=self.config,settings=self.settings,selection=self.selection,epoch=self.epoch,done=self.offset,total=self.example_count,steps=self.steps,loss=self.last_loss,history=copy.deepcopy(self.history),qualification=copy.deepcopy(self.qualification),check_progress=self.check_progress,
                        state_sha256=digest,checkpoint_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest() if self.path.exists() else hashlib.sha256((BASE_DIR/'base.pt').read_bytes()).hexdigest(),origin=self.origin,base=META,framework=LOCK,implementation=IMPLEMENTATION,
                        parameters=sum(p.numel() for p in self.model.parameters()),layout=self.layout,world=self.world,observation=W.observe(self.world),memory=copy.deepcopy(self.memory),last_decision=copy.deepcopy(self.decisions[-1] if self.decisions else None),decisions=len(self.decisions),repeated=self.repeated,reached=W.reached(self.world),move_limit=W.limit(self.world),reachable=plan['reachable'],starts=len(plan['starts']),world_changed=W.identity(self.layout)!=self.training_identity)

    def apply_layout(self,w):
        W.validate(w)
        with self.lock:
            self.idle();encoding_changed=W.expanded(w)!=W.expanded(self.layout);changed=W.identity(w)!=W.identity(self.layout)
            self.layout=W.fresh(w)
            if encoding_changed:self.reset()
            elif changed:self.examples=None;self.qualification=None;self.rewind()
            else:self.rewind()
            self.save();return self.snapshot()

    def action(self):
        with self.lock:
            self.idle()
            if W.reached(self.world) or self.repeated or self.world['tick']>=W.limit(self.world):raise ValueError('Journey ended; retry or choose another start')
            if not self.plan()['reachable']:raise ValueError('No route; edit the world before playing')
            before_hash=state_hash(self.model);f=W.features(self.world);trace=self.model.inspect(f,self.memory)
            action=trace['actionIndex'];result=W.step(self.world,W.ACTIONS[action]);self.memory.append(W.record(f,action));self.world=result['world']
            key=W.pose(self.world['agent']);prior=self.visits.get(key,0)
            self.repeated=not W.reached(self.world) and (prior>=7 if self.model.has_memory else prior>0 or key==W.pose(self.play_agent))
            self.visits[key]=prior+1
            self.decisions.append(dict(trace=trace,events=result['events'],action=W.ACTIONS[action],tick=self.world['tick'],state_sha256=before_hash,training_steps=self.steps))
            self.save();return self.snapshot()

    def retry(self,another=False):
        with self.lock:
            self.idle();agent=None
            if another:
                trials=self.plan()['trials']
                if not trials:raise ValueError('No reachable starts')
                index=next((i for i,a in enumerate(trials) if a==self.play_agent),-1);agent=trials[(index+1)%len(trials)]
            self.rewind(agent);self.save();return self.snapshot()

    def update(self,batch):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        ids=torch.tensor([W.tokens(e['features']) for e in batch],dtype=torch.long)
        target=torch.tensor([96+e['label'] for e in batch]);logits=self.model(ids,[e.get('memory',[]) for e in batch])[:,-1]
        loss=torch.nn.functional.cross_entropy(logits,target)
        if not torch.isfinite(loss):raise ValueError('Non-finite Ava loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=True)
        for group in self.optimizer.param_groups:group['lr']=self.settings['rate']*min(1,(self.steps+1)/20)
        self.optimizer.step();self.steps+=1;self.last_loss=float(loss.detach());self.round_loss+=self.last_loss*len(batch);self.offset+=len(batch)

    def start(self,settings,selection,background=True):
        if type(settings.get('epochs')) is not int or not 1<=settings['epochs']<=240 or type(settings.get('seed')) is not int or not 0<=settings['seed']<1000000 or settings.get('rate') not in (.0003,.0005,.001,.003,.01):raise ValueError('Choose supported Ava rounds, seed and rate')
        with self.lock:
            self.idle()
            # Validate feasibility and selected data before resetting learned weights.
            examples,worlds=W.dataset(self.layout,selection,self.model.has_memory)
            changed=W.identity(self.layout)!=self.training_identity or selection!=self.selection or settings['rate']!=self.settings['rate'] or settings['seed']!=self.settings['seed'] or self.epoch>=settings['epochs']
            self.settings=copy.deepcopy(settings);self.selection=list(selection)
            if changed:self.reset()
            self.examples=examples;self.example_count=len(examples);self.training_worlds=worlds;self.qualification=None;self.rewind();self.error=None;self.pause.clear();self.busy=True;self.phase='training';self.save()
        if background:threading.Thread(target=self.run,daemon=True).start()
        else:self.run()

    def configure_practice(self,settings,selection):
        if type(settings.get('epochs')) is not int or not 1<=settings['epochs']<=240 or type(settings.get('seed')) is not int or not 0<=settings['seed']<1000000 or settings.get('rate') not in (.0003,.0005,.001,.003,.01):raise ValueError('Choose supported Ava rounds, seed and rate')
        W.situation_worlds(self.layout,selection)
        with self.lock:
            self.idle()
            changed=selection!=self.selection or settings['rate']!=self.settings['rate'] or settings['seed']!=self.settings['seed']
            self.settings=copy.deepcopy(settings);self.selection=list(selection)
            if changed:self.reset()
            self.save();return self.snapshot()

    def evaluate_examples(self):
        correct=0;loss=0.;classes=[dict(correct=0,total=0) for _ in range(3)]
        for i in range(0,len(self.examples),32):
            if self.pause.is_set():return None
            batch=self.examples[i:i+32]
            with self.lock:probabilities=self.model.predict(batch)
            for e,p in zip(batch,probabilities):
                label=e['label'];match=p.index(max(p))==label;correct+=match;classes[label]['correct']+=match;classes[label]['total']+=1
                import math
                loss-=math.log(max(1e-9,p[label]))
        return dict(correct=correct,total=len(self.examples),accuracy=correct/len(self.examples),loss=loss/len(self.examples),classes=classes,scope='Agreement on selected practice, not held-out generalisation')

    def run(self):
        try:
            while self.epoch<self.settings['epochs'] and not self.pause.is_set():
                order=W.shuffle(self.examples,self.settings['seed']+self.epoch*7919)
                while self.offset<len(order) and not self.pause.is_set():
                    with self.lock:self.update(order[self.offset:self.offset+32]);self.save()
                if self.offset<len(order):break
                self.phase='checking practice';measurement=self.evaluate_examples()
                if measurement is None:break
                with self.lock:
                    self.epoch+=1;self.history.append(dict(epoch=self.epoch,loss=self.round_loss/len(order),validation=measurement));self.offset=0;self.round_loss=0.;self.save()
                if self.epoch%10==0 or self.epoch==self.settings['epochs']:
                    self.phase='checking journeys';self.qualification=self.rollouts()
                    with self.lock:self.save()
                    if self.qualification['completed']==self.qualification['total'] and not self.qualification['interrupted']:break
                self.phase='training'
        except Exception as exc:self.error=str(exc)
        finally:
            with self.lock:self.busy=False;self.phase='ready';self.model.eval();self.save()

    def rollouts(self):
        room=W.fresh(self.layout);plan=W.teaching_plan(room);starts=plan['starts'];runs=[dict(index=i,agent=a,world=W.fresh(room,a),memory=[],actions=[],blocked=0,visits={},done=False) for i,a in enumerate(starts)]
        policy=None
        if not self.model.has_memory:
            policy={}
            for i in range(0,len(plan['examples']),32):
                if self.pause.is_set():break
                chunk=plan['examples'][i:i+32]
                with self.lock:p=self.model.predict(chunk)
                policy.update({e['id']:row.index(max(row)) for e,row in zip(chunk,p)})
        interrupted=self.pause.is_set()
        for tick in range(W.limit(room)):
            if self.pause.is_set():interrupted=True;break
            active=[r for r in runs if not r['done']]
            if not active:break
            for i in range(0,len(active),32):
                if self.pause.is_set():interrupted=True;break
                chunk=active[i:i+32];features=[W.features(r['world']) for r in chunk]
                with self.lock:p=self.model.predict([dict(features=f,memory=r['memory']) for f,r in zip(features,chunk)]) if self.model.has_memory else None
                for j,r in enumerate(chunk):
                    old=W.pose(r['world']['agent'])
                    if not self.model.has_memory and old in r['visits']:r.update(done=True,repeated=True);continue
                    if not self.model.has_memory:r['visits'][old]=1
                    action=p[j].index(max(p[j])) if p else policy[old]
                    r['memory'].append(W.record(features[j],action));r['actions'].append(W.ACTIONS[action]);result=W.step(r['world'],W.ACTIONS[action],False);r['world']=result['world'];r['blocked']+=result['events'][0]['kind']=='blocked'
                    key=W.pose(r['world']['agent'])
                    if self.model.has_memory:r['visits'][key]=r['visits'].get(key,0)+1
                    repeated=self.model.has_memory and r['visits'].get(key,0)>=8
                    r.update(repeated=repeated,done=W.reached(r['world']) or repeated or r['world']['tick']>=W.limit(room))
            self.check_progress=dict(move=tick+1,finished=sum(r['done'] for r in runs),total=len(runs))
        measured=[dict(index=r['index'],agent=r['agent'],success=W.reached(r['world']),steps=r['world']['tick'],blocked=r['blocked'],repeated=r.get('repeated',False),actions=r['actions']) for r in runs]
        return dict(interrupted=interrupted,sampled=False,completed=sum(r['success'] for r in measured),total=len(measured),runs=measured,worldIdentity=W.identity(room),state_sha256=state_hash(self.model),scope='All reachable starting poses in the displayed room; separate histories; practice exposure, not held-out evaluation')

    def check(self):
        with self.lock:
            self.idle()
            if not self.plan()['reachable']:raise ValueError('No route from Ava; edit the layout first')
            self.pause.clear();self.busy=True;self.phase='checking journeys';self.check_progress=None
        def run():
            try:self.qualification=self.rollouts()
            except Exception as exc:self.error=str(exc)
            finally:
                with self.lock:self.busy=False;self.phase='ready';self.save()
        threading.Thread(target=run,daemon=True).start()
