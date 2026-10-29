"""Faithful Python port of Browser Lab navigation rules; teachers are training only."""
import copy
import json
import math
from collections import deque

EXPANDED='editable-keys-doors-v2'
WORLD_VERSION='gridworld-v1/browser-navigation-1'
SMALL='editable-map-lamp-v1'
ACTIONS=['forward','turn-left','turn-right']
DELTAS=[(0,-1),(1,0),(0,1),(-1,0)]
OBJECTS=['key-red','key-blue','door-red','door-blue']

def rng(seed):
    state=seed & 0xffffffff
    def draw():
        nonlocal state
        state=(state+0x6D2B79F5)&0xffffffff
        t=state
        t=((t^(t>>15))*(t|1))&0xffffffff
        t^=(t+(((t^(t>>7))*(t|61))&0xffffffff))&0xffffffff
        return ((t^(t>>14))&0xffffffff)/4294967296
    return draw

def shuffle(items,seed):
    result=list(items);r=rng(seed)
    for i in range(len(result)-1,0,-1):
        j=math.floor(r()*(i+1));result[i],result[j]=result[j],result[i]
    return result

def expanded(w):return w['mapName']==EXPANDED
def bounds(w):return (1,14) if expanded(w) else (4,11)
def limit(w):return 512 if expanded(w) else 160
def keybit(t):return 1 if t.endswith('-red') else 2 if t.endswith('-blue') else 0
def pose(a):return ','.join(str(a[k]) for k in ('x','y','direction','keys') if k in a)
def inside(x,y):return 0<=x<16 and 0<=y<16

def enterable(w,x,y,keys=None):
    if not inside(x,y):return False
    t=w['tiles'][y][x];keys=w['agent'].get('keys',0) if keys is None else keys
    return t!='wall' and (not t.startswith('door-') or bool(keys&keybit(t)))

def collected(w,x,y,keys):return keys|(keybit(w['tiles'][y][x]) if w['tiles'][y][x].startswith('key-') else 0)
def visible_tile(w,x,y):
    t=w['tiles'][y][x]
    return 'floor' if t.startswith('key-') and w['agent'].get('keys',0)&keybit(t) else t

def fresh(w,agent=None):
    out=copy.deepcopy(w);out.update(tick=0,actions=[],agent=copy.deepcopy(w['agent'] if agent is None else agent));return out

def expand(w):
    w=fresh(w)
    if expanded(w):return w
    w['mapName']=EXPANDED;w['agent']['keys']=0
    for y in range(1,15):
        for x in range(1,15):
            if x<4 or x>11 or y<4 or y>11:w['tiles'][y][x]='floor'
    return w

def preset(kind='detour',large=False):
    w=dict(version=WORLD_VERSION,seed=3001,mapName=SMALL,tick=0,tiles=[['floor' if 4<=x<=11 and 4<=y<=11 else 'wall' for x in range(16)] for y in range(16)],agent=dict(x=5,y=8,direction=1),goal=[10,8],actions=[])
    if kind=='detour':
        for y in range(6,10):w['tiles'][y][7]='wall'
    if kind=='zigzag':
        for y in range(4,9):w['tiles'][y][7]='wall'
        for y in range(7,12):w['tiles'][y][9]='wall'
        w.update(agent=dict(x=5,y=6,direction=1),goal=[10,9])
    w['tiles'][w['goal'][1]][w['goal'][0]]='lamp'
    if large:
        w=expand(w)
        if kind=='locks':
            for y in range(1,15):w['tiles'][y][8]='wall'
            w['tiles'][w['goal'][1]][w['goal'][0]]='floor'
            w.update(agent=dict(x=2,y=7,direction=1,keys=0),goal=[12,7])
            w['tiles'][7][12]='lamp';w['tiles'][7][8]='door-red';w['tiles'][11][8]='door-blue';w['tiles'][3][4]='key-red'
    return w

def validate(w):
    if not isinstance(w,dict) or w.get('mapName') not in (SMALL,EXPANDED):raise ValueError('Unsupported world adapter')
    lo,hi=bounds(w);tiles=w.get('tiles');allowed=['floor','wall','lamp']+(OBJECTS if expanded(w) else [])
    if not isinstance(tiles,list) or len(tiles)!=16 or any(not isinstance(r,list) or len(r)!=16 or any(t not in allowed for t in r) for r in tiles):raise ValueError('Choose a supported 16×16 layout')
    a=w.get('agent',{});g=w.get('goal',[])
    if len(g)!=2 or any(type(v) is not int or not lo<=v<=hi for v in [a.get('x'),a.get('y'),*g]) or type(a.get('direction')) is not int or not 0<=a['direction']<=3:raise ValueError('Place Ava and lamp inside the room')
    if expanded(w) and (type(a.get('keys')) is not int or not 0<=a['keys']<=3):raise ValueError('Invalid carried keys')
    flat=sum(tiles,[])
    if expanded(w) and (any(flat.count('key-'+c)>1 for c in ('red','blue')) or sum(t.startswith('door-') for t in flat)>8):raise ValueError('Use at most one key per colour and eight doors')
    if any(tiles[y][x]!='wall' for y in range(16) for x in range(16) if not(lo<=x<=hi and lo<=y<=hi)):raise ValueError('Keep fixed outside walls')
    if flat.count('lamp')!=1 or tiles[g[1]][g[0]]!='lamp' or tiles[a['y']][a['x']]=='wall' or [a['x'],a['y']]==g:raise ValueError('Keep one lamp and a separate clear Ava start')

def identity(w):
    tiles=w['tiles'] if expanded(w) else [r[4:12] for r in w['tiles'][4:12]]
    return json.dumps(dict(version=w['mapName'] if expanded(w) else SMALL,tiles=tiles,goal=w['goal']),separators=(',',':'))

def edit(w,x,y,tool):
    validate(w);lo,hi=bounds(w)
    if type(x) is not int or type(y) is not int or not(lo<=x<=hi and lo<=y<=hi):return copy.deepcopy(w)
    w=fresh(w);t=w['tiles'][y][x];a=w['agent']
    if tool=='lamp':
        if [x,y]==[a['x'],a['y']]:raise ValueError('Place lamp away from Ava')
        w['tiles'][w['goal'][1]][w['goal'][0]]='floor';w['goal']=[x,y];w['tiles'][y][x]='lamp'
    elif tool=='ava':
        if t!='floor':raise ValueError('Place Ava on clear floor')
        a.update(x=x,y=y)
    elif tool in OBJECTS:
        if not expanded(w):raise ValueError('Expand before placing keys and doors')
        if [x,y]==[a['x'],a['y']] or t=='lamp':raise ValueError('Move Ava or lamp before placing an object')
        if t==tool:w['tiles'][y][x]='floor'
        else:
            if tool.startswith('key-'):
                w['tiles']=[['floor' if tile==tool else tile for tile in row] for row in w['tiles']]
            w['tiles'][y][x]=tool
    elif tool in ('wall','erase'):
        if [x,y]==[a['x'],a['y']] or t=='lamp':raise ValueError('Move Ava or lamp before changing this cell')
        w['tiles'][y][x]='wall' if tool=='wall' else 'floor'
    else:raise ValueError('Unsupported editing tool')
    validate(w);return w

def phase(tick):
    n=tick%100;return 'day' if n<60 else 'dusk' if n<70 else 'night' if n<90 else 'dawn'

def line(x0,y0,x1,y1):
    out=[];dx=abs(x1-x0);sx=1 if x0<x1 else -1;dy=-abs(y1-y0);sy=1 if y0<y1 else -1;e=dx+dy
    while True:
        out.append((x0,y0))
        if (x0,y0)==(x1,y1):return out
        twice=2*e
        if twice>=dy:e+=dy;x0+=sx
        if twice<=dx:e+=dx;y0+=sy

def los(w,x0,y0,x1,y1):return all(w['tiles'][y][x]!='wall' and (not w['tiles'][y][x].startswith('door-') or enterable(w,x,y)) for x,y in line(x0,y0,x1,y1)[1:-1])
def observe(w):
    a=w['agent'];p=phase(w['tick']);distance=float('inf') if p=='day' else 3 if p=='night' else 5
    lamps=[(x,y) for y,row in enumerate(w['tiles']) for x,t in enumerate(row) if t=='lamp'];out=[]
    for ly in range(9):
        row=[]
        for lx in range(9):
            r=lx-4;f=8-ly;x,y=[(a['x']+r,a['y']-f),(a['x']+f,a['y']+r),(a['x']-r,a['y']+f),(a['x']-f,a['y']-r)][a['direction']]
            visible=inside(x,y) and (max(abs(x-a['x']),abs(y-a['y']))<=distance or any(max(abs(x-gx),abs(y-gy))<=3 and los(w,gx,gy,x,y) for gx,gy in lamps)) and los(w,a['x'],a['y'],x,y)
            c=dict(local_x=lx,local_y=ly,world_x=x if inside(x,y) else None,world_y=y if inside(x,y) else None,visible=visible,tile=visible_tile(w,x,y) if visible else None)
            if expanded(w):c['doorOpen']=visible and w['tiles'][y][x].startswith('door-') and enterable(w,x,y)
            row.append(c)
        out.append(row)
    return out

def reached(w):return [w['agent']['x'],w['agent']['y']]==w['goal']
def step(w,action,with_observation=True):
    if action not in ACTIONS+['wait']:raise ValueError('Unsupported world action')
    old=w;w=copy.deepcopy(w);a=w['agent'];w['tick']+=1;kind='waited';details={}
    if action in ACTIONS[1:]:a['direction']=(a['direction']+(3 if action=='turn-left' else 1))%4;kind='turned';details={'turn':'left' if action=='turn-left' else 'right'}
    if action=='forward':
        dx,dy=DELTAS[a['direction']];x,y=a['x']+dx,a['y']+dy
        if not enterable(w,x,y):
            kind='blocked';details={'cause':'action'}
            if inside(x,y) and w['tiles'][y][x].startswith('door-'):details['lockedDoor']=w['tiles'][y][x][5:]
        else:
            a.update(x=x,y=y)
            if expanded(w):a['keys']=collected(w,x,y,a.get('keys',0))
            kind='moved';details={'direction':['north','east','south','west'][a['direction']],'cause':'action'}
    events=[dict(tick=w['tick'],kind=kind,actor='ava',position=[a['x'],a['y']],details=details)]
    if a.get('keys',0)!=old['agent'].get('keys',0):events.append(dict(tick=w['tick'],kind='key-collected',actor='ava',position=[a['x'],a['y']],details={'colour':'red' if keybit(w['tiles'][a['y']][a['x']])==1 else 'blue'}))
    if kind=='moved' and w['tiles'][a['y']][a['x']].startswith('door-'):events.append(dict(tick=w['tick'],kind='door-unlocked',actor='ava',position=[a['x'],a['y']],details={'colour':w['tiles'][a['y']][a['x']][5:]}))
    if phase(w['tick'])!=phase(old['tick']):events.append(dict(tick=w['tick'],kind='phase-changed',actor=None,position=None,details={'phase':phase(w['tick'])}))
    w['actions'].append(action);result=dict(world=w,events=events)
    if with_observation:result['observation']=observe(w)
    return result

def features(w):
    a=w['agent'];dx,dy=DELTAS[a['direction']];scale=8 if expanded(w) else 4
    f=[(a['x']-7.5)/scale,(a['y']-7.5)/scale,dx,dy,(w['goal'][0]-7.5)/scale,(w['goal'][1]-7.5)/scale]
    if expanded(w):return f+[int(t=='wall') for r in w['tiles'] for t in r]+[(OBJECTS.index(visible_tile(w,x,y))+1 if visible_tile(w,x,y) in OBJECTS else 0) for y in range(16) for x in range(16)]+[int(bool(a['keys']&1)),int(bool(a['keys']&2))]
    return f+[int(t=='wall') for row in w['tiles'][4:12] for t in row[4:12]]

def tokens(f):
    if len(f) not in (70,520) or not all(math.isfinite(v) for v in f):raise ValueError('Expected finite full-map features')
    large=len(f)==520;coord=lambda v:math.floor(v*(8 if large else 4)+8)-(0 if large else 4)
    d=0 if f[3]==-1 else 1 if f[2]==1 else 2 if f[3]==1 else 3
    chunks=[16+sum(int(f[6+i*4+j]==1)*2**j for j in range(4)) for i in range(64 if large else 16)]
    if not large:return [1,2,*chunks,3,32+coord(f[0]),40+coord(f[1]),48+d,56+coord(f[4]),64+coord(f[5]),4]
    objects=[v for i,t in enumerate(f[262:518]) if t for v in (104+t,32+i%16,48+i//16)]
    if len(objects)>30:raise ValueError('Use at most two keys and eight doors')
    return [1,2,*chunks,*objects,*([0]*(30-len(objects))),3,32+coord(f[0]),48+coord(f[1]),100+d,64+coord(f[4]),80+coord(f[5]),109+f[518]+2*f[519],4]

def record(f,action):
    t=tokens(f)
    return [t[-7],t[-6],t[-5],96+action,t[-2]] if len(f)==520 else [t[-6],t[-5],t[-4],96+action]

def memory(history,bank,capacity):
    if bank=='recent':return sum(history[-capacity:],[])
    records=[];seen=set()
    for r in history:
        key=tuple(r[:3]+(r[4:] if len(r)==5 else []))
        if key not in seen:seen.add(key);records.append(r)
        if len(records)==capacity:break
    return sum(records,[])

def teaching_plan(w):
    validate(w);lo,hi=bounds(w);states={};reverse={};distance={};q=deque()
    available=0
    for r in w['tiles']:
        for t in r:
            if t.startswith('key-'):available|=keybit(t)
    masks=[k for k in range(4) if k&available==k] if expanded(w) else [None]
    for y in range(lo,hi+1):
        for x in range(lo,hi+1):
            for keys in masks:
                if enterable(w,x,y,keys or 0) and (keys is None or collected(w,x,y,keys)==keys):
                    for d in range(4):
                        a=dict(x=x,y=y,direction=d)
                        if keys is not None:a['keys']=keys
                        states[pose(a)]=a
    def next_pose(a,action):
        a=dict(a)
        if action:a['direction']=(a['direction']+(3 if action==1 else 1))%4
        else:
            dx,dy=DELTAS[a['direction']];x,y=a['x']+dx,a['y']+dy
            if enterable(w,x,y,a.get('keys',0)):
                a.update(x=x,y=y)
                if 'keys' in a:a['keys']=collected(w,x,y,a['keys'])
        return pose(a)
    for key,a in states.items():
        if [a['x'],a['y']]==w['goal']:distance[key]=0;q.append(key)
        for action in range(3):reverse.setdefault(next_pose(a,action),[]).append(key)
    while q:
        dest=q.popleft()
        for prior in reverse.get(dest,[]):
            if prior not in distance:distance[prior]=distance[dest]+1;q.append(prior)
    examples=[]
    for key,a in states.items():
        cost=distance.get(key)
        if not cost:continue
        optimal=[action for action in range(3) if distance.get(next_pose(a,action))==cost-1]
        examples.append(dict(id=key,agent=a,features=features(fresh(w,a)),label=optimal[0],optimal=optimal,cost=cost))
    starts=[dict(e['agent']) for e in examples if not e['agent'].get('keys',0)]
    trials=starts[::max(1,len(starts)//12)][:12] if expanded(w) else [dict(e['agent']) for e in shuffle(examples,20261023)[:12]]
    return dict(examples=examples,starts=starts,trials=trials,reachable=pose(w['agent']) in distance,identity=identity(w))

def situation_worlds(w,selection):
    if not isinstance(selection,list) or not selection or len(selection)!=len(set(selection)) or any(k not in ('layout','lamps','obstacles','routes','locks') for k in selection):raise ValueError('Choose supported training situations')
    if 'locks' in selection and not expanded(w):raise ValueError('Expand before selecting keys and doors')
    worlds=[fresh(w)] if 'layout' in selection else []
    if 'lamps' in selection:
        reachable={(e['agent']['x'],e['agent']['y']) for e in teaching_plan(w)['examples']};lo,hi=bounds(w)
        free=[(x,y) for y in range(lo,hi+1) for x in range(lo,hi+1) if w['tiles'][y][x]=='floor' and (x,y) in reachable and (x,y)!=(w['agent']['x'],w['agent']['y'])]
        for at in ([free[0],free[-1]] if free else []):
            other=fresh(w);other['tiles'][w['goal'][1]][w['goal'][0]]='floor';other['goal']=list(at);other['tiles'][at[1]][at[0]]='lamp'
            if teaching_plan(other)['reachable']:worlds.append(other)
    if 'obstacles' in selection:worlds.extend(preset(k,expanded(w)) for k in ('empty','detour'))
    if 'routes' in selection:worlds.append(preset('zigzag',expanded(w)))
    if 'locks' in selection:
        a=preset('locks',True);b=copy.deepcopy(a);b['tiles']=[[t.replace('-red','-blue') if t.endswith('-red') else t.replace('-blue','-red') if t.endswith('-blue') else t for t in r] for r in b['tiles']];worlds.extend([a,b])
    return list({identity(world):world for world in worlds}.values())

def dataset(w,selection,with_memory):
    if not teaching_plan(w)['reachable']:raise ValueError('Lamp unreachable from Ava; edit a route before training')
    worlds=situation_worlds(w,selection);train=[]
    for room in worlds:
        data=teaching_plan(room)
        if not data['reachable']:raise ValueError('Lamp unreachable from Ava; edit a route before training')
        train.extend(dict(e,world=room,memory=[]) for e in data['examples'])
        if with_memory:
            lookup={e['id']:e for e in data['examples']}
            for agent in data['trials'][:8]:
                walk=fresh(room,agent);history=[]
                while not reached(walk) and walk['tick']<40:
                    e=lookup.get(pose(walk['agent']))
                    if e is None:break
                    train.append(dict(e,features=features(walk),world=room,memory=copy.deepcopy(history)))
                    history.append(record(e['features'],e['label']));walk=step(walk,ACTIONS[e['label']],False)['world']
    return train,worlds
