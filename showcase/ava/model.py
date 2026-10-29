"""Ordered Ava stages reusing the pinned canonical PyTorch transformer components."""
import copy
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from ..core import ARCH
from training.runtime import load_architecture
from .world import memory, tokens, rng

canonical=load_architecture(ARCH)
OPTIONS={'depth':{'layers':[1,2,3,5]},'context':{'window':[8,16,32,128]},'object':{'focus':['lamp','ava','both']},'recent':{'capacity':[2,4,8]},'archive':{'capacity':[4,8,16]},'workspace':{'slots':[4,8,16]},'refinement':{'multiplier':[1,2,4],'activation':['gelu','relu']},'perspectives':{'heads':[1,2,4]}}

def compile_config(config):
    c=copy.deepcopy(config);caps=c.get('capabilities');seen=set();blocks=[]
    if c.get('version')!=2 or c.get('width') not in (16,24,32,64,128) or c.get('context')!=128 or c.get('vocab')!=128 or c.get('repeat') is not False:raise ValueError('Unsupported active Ava architecture')
    if not isinstance(caps,list) or not caps:raise ValueError('At least one active capability required')
    def piece(kind,id,**kw):return dict(id=id,type=kind,residual=True,normalize=True,**kw)
    for cap in caps:
        kind=cap.get('type')
        if kind not in OPTIONS or kind in seen or any(cap.get(k) not in vals for k,vals in OPTIONS[kind].items()):raise ValueError('Invalid or duplicate capability')
        seen.add(kind)
        if kind=='depth':
            for i in range(cap['layers']):blocks.extend([piece('attention',f'depth-attention-{i}',heads=4),piece('ff',f'depth-ff-{i}',multiplier=4,activation='gelu')])
        elif kind in ('context','object','perspectives'):blocks.append(piece('attention',kind,**({'heads':cap['heads']} if kind=='perspectives' else {'heads':4,**{k:v for k,v in cap.items() if k!='type'}})))
        elif kind=='refinement':blocks.append(piece('ff',kind,multiplier=cap['multiplier'],activation=cap['activation']))
        elif kind in ('recent','archive'):blocks.append(piece('memory',kind,bank=kind,capacity=cap['capacity']))
        else:blocks.append(piece('workspace',kind,slots=cap['slots']))
    if len(blocks)>16:raise ValueError('Ava supports at most 16 processing stages')
    if 'blocks' in c and c['blocks']!=blocks:raise ValueError('Visible capabilities and compiled stages differ')
    c['blocks']=blocks;return c


class Stage(nn.Module):
    def __init__(self,b,width):
        super().__init__();self.spec=b;self.norm=nn.LayerNorm(width)
        cfg=canonical.ModelConfig(d_model=width,n_heads=b.get('heads',1),n_layers=1,d_ff=width*b.get('multiplier',4))
        if b['type'] in ('attention','memory'):
            self.operation=canonical.CausalSelfAttention(cfg)
            if b.get('focus'):self.object_bias=nn.Parameter(torch.ones(b['heads']))
        elif b['type']=='ff':self.operation=canonical.FeedForward(cfg)
        else:self.compress=nn.Linear(width,b['slots']);self.broadcast=nn.Linear(b['slots'],width)

    def forward(self,x,ids,embeddings,positions,bank=None,inspect=False):
        b=self.spec;z=self.norm(x);attention=None;op=getattr(self,'operation',None)
        if b['type']=='attention':
            if not b.get('window') and not b.get('focus'):
                y=op(z)
                if inspect:
                    batch,time,d=z.shape;h=b['heads'];hd=d//h
                    q=op.q_proj(z).reshape(batch,time,h,hd).transpose(1,2);k=op.k_proj(z).reshape(batch,time,h,hd).transpose(1,2)
                    mask=torch.arange(time)[None,:]>torch.arange(time)[:,None]
                    attention=(q@k.transpose(-2,-1)/math.sqrt(hd)+mask.to(z.dtype)*-1e9).softmax(-1)
            else:
                batch,time,d=z.shape;h=b['heads'];hd=d//h
                def heads(layer):return layer(z).reshape(batch,time,h,hd).transpose(1,2)
                scores=heads(op.q_proj)@heads(op.k_proj).transpose(-2,-1)/math.sqrt(hd)
                i=torch.arange(time);mask=(i[None,:]>i[:,None])
                if b.get('window'):mask|=(i[:,None]-i[None,:]>=b['window'])
                scores=scores+mask.to(z.dtype)*-1e9
                if b.get('focus'):
                    focus=b['focus'];selected=torch.zeros_like(ids,dtype=torch.bool)
                    if time==104:
                        if focus!='ava':selected[:,[time-4,time-3]]=True
                        if focus!='lamp':selected[:,[time-7,time-6]]=True
                    else:
                        if focus!='ava':selected|=(ids>=56)&(ids<72)
                        if focus!='lamp':selected|=(ids>=32)&(ids<48)
                    scores=scores+self.object_bias[None,:,None,None]*selected[:,None,None,:]
                attention=scores.softmax(-1);y=op.out_proj((attention@heads(op.v_proj)).transpose(1,2).reshape(batch,time,d))
        elif b['type']=='memory':
            mids=bank;values=embeddings(mids)+positions(torch.arange(mids.shape[1]))
            scores=op.q_proj(z)@op.k_proj(values).transpose(-2,-1)/math.sqrt(x.shape[-1])+(mids==128)[:,None,:].to(x.dtype)*-1e9
            attention=scores.softmax(-1)
            y=op.out_proj(attention@op.v_proj(values))*(mids!=128).any(1)[:,None,None]
        elif b['type']=='workspace':y=self.broadcast(torch.tanh(self.compress(z.cumsum(1)/torch.arange(1,z.shape[1]+1)[None,:,None])))
        elif b['activation']=='gelu':y=op(z)
        else:y=op.down_proj(F.relu(op.up_proj(z)))
        return x+y,attention


class AvaModel(nn.Module):
    def __init__(self,config,seed=42):
        super().__init__();self.config=compile_config(config);d=config['width']
        # Embeddings, attention, feed-forward and normalizations use canonical modules.
        base=canonical.Transformer(canonical.ModelConfig(n_layers=1,d_model=d,n_heads=4,d_ff=4*d))
        self.token_embedding=base.token_embedding;self.position_embedding=base.position_embedding;self.final_norm=base.final_norm
        self.stages=nn.ModuleList([Stage(b,d) for b in self.config['blocks']])
        self.names={};a=f=0
        for b in self.config['blocks']:
            if b['type'] in ('memory','workspace'):name=f"extensions.{b['id']}.ff"
            elif b['type']=='attention':name=f'blocks.{a}.attention';a+=1
            else:name=f'blocks.{f}.ff';f+=1
            self.names[b['id']]=name
        self.initialize_browser(seed)

    @property
    def has_memory(self):return any(b['type']=='memory' for b in self.config['blocks'])

    def browser_parameters(self):
        pairs={'token_embedding.weight':self.token_embedding.weight,'position_embedding.weight':self.position_embedding.weight}
        focus=[]
        for b,s in zip(self.config['blocks'],self.stages):
            name=self.names[b['id']];norm=re.sub(r'\.(attention|ff)$',lambda m:'.'+m[1]+'_norm',name)
            pairs[norm+'.weight']=s.norm.weight;pairs[norm+'.bias']=s.norm.bias
            if b['type']=='workspace':
                for k in ('compress','broadcast'):
                    layer=getattr(s,k);pairs[name+'.'+k+'.weight']=layer.weight;pairs[name+'.'+k+'.bias']=layer.bias
            else:
                for n,p in s.operation.named_parameters():pairs[name+'.'+n]=p
            if b.get('focus'):focus.append((name+'.object_bias',s.object_bias))
        pairs.update(focus);pairs['final_norm.weight']=self.final_norm.weight;pairs['final_norm.bias']=self.final_norm.bias
        return pairs

    @torch.no_grad()
    def initialize_browser(self,seed):
        r=rng(seed);layers=sum(b['type']=='attention' for b in self.config['blocks']) or 1
        for name,p in self.browser_parameters().items():
            if name.endswith('.bias'):p.zero_()
            elif '_norm.weight' in name or name=='final_norm.weight' or name.endswith('object_bias'):p.fill_(1)
            else:
                scale=.02/math.sqrt(2*layers) if '.attention.out_proj.weight' in name or '.down_proj.weight' in name else .02
                values=[math.sqrt(-2*math.log(max(1e-12,r())))*math.cos(2*math.pi*r())*scale for _ in range(p.numel())]
                p.copy_(torch.tensor(values).reshape(p.shape))
        self.token_embedding.weight[128].zero_()

    @torch.no_grad()
    def load_browser(self,weights):
        pairs=self.browser_parameters()
        if set(pairs)!=set(weights):raise ValueError('Browser checkpoint tensor keys differ')
        converted={}
        for name,p in pairs.items():
            value=weights[name]
            t=torch.tensor(value['values'],dtype=torch.float32)
            if list(p.shape)!=value['shape'] or t.numel()!=p.numel() or not torch.isfinite(t).all():raise ValueError('Invalid tensor: '+name)
            converted[name]=t.reshape(p.shape)
        for name,t in converted.items():pairs[name].copy_(t)

    def forward(self,ids,histories=None,inspect=False):
        batch,time=ids.shape
        if not 1<=time<=128:raise ValueError('Context limit exceeded')
        histories=histories or [[] for _ in range(batch)]
        if len(histories)!=batch:raise ValueError('One independent history per example is required')
        x=self.token_embedding(ids)+self.position_embedding(torch.arange(time));traces=[]
        for b,s in zip(self.config['blocks'],self.stages):
            bank=None
            if b['type']=='memory':
                record_size=5 if time==104 else 4;length=b['capacity']*record_size
                values=[memory(hist,b['bank'],b['capacity']) for hist in histories]
                if any(len(v)>length for v in values):raise ValueError('Memory adapter mismatch')
                bank=torch.tensor([v+[128]*(length-len(v)) for v in values],dtype=torch.long)
            x,attention=s(x,ids,self.token_embedding,self.position_embedding,bank,inspect)
            if inspect:
                traces.append(dict(id=b['id'],type=b['type'],state=x[:,-1].detach().tolist(),**({'attention':attention[:,:,-1].detach().tolist() if attention.ndim==4 else attention[:,-1].detach().tolist()} if attention is not None else {})))
        logits=F.linear(self.final_norm(x),self.token_embedding.weight[:128])
        return (logits,traces) if inspect else logits

    @torch.inference_mode()
    def predict(self,examples):
        self.eval();ids=torch.tensor([tokens(e['features']) for e in examples],dtype=torch.long)
        logits=self(ids,[e.get('memory',[]) for e in examples])[:,-1]
        if not torch.isfinite(logits).all():raise ValueError('Non-finite Ava prediction')
        return logits[:,96:99].softmax(-1).tolist()

    @torch.inference_mode()
    def inspect(self,features,history):
        self.eval();ids=tokens(features);logits,trace=self(torch.tensor([ids]),[history],True);last=logits[0,-1]
        if not torch.isfinite(last).all():raise ValueError('Non-finite Ava prediction')
        probs=last[96:99].softmax(-1).tolist()
        return dict(features=features,tokens=ids,vocabLogits=last.tolist(),probabilities=probs,actionIndex=probs.index(max(probs)),stages=trace,memoryReads=[dict(bank=b['bank'],records=len(memory(history,b['bank'],b['capacity']))//(5 if len(features)==520 else 4)) for b in self.config['blocks'] if b['type']=='memory'])
