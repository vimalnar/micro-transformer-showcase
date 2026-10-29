const $=id=>document.getElementById(id);let state,polling=false,acting=false,watching=false,timer,generation=0,operationQueue=Promise.resolve(),networkFailure=false;
const actions=['forward','turn-left','turn-right'];
async function api(path,body){
 let r;try{r=await fetch('/api/ava/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
 catch{const e=Error('Python service unavailable. Reconnecting…');e.transport=true;throw e;}
 const v=await r.json();if(!r.ok)throw Error(v.detail||r.statusText);return v;
}
function stopWatch(){watching=false;clearTimeout(timer);$('watch').textContent='Watch Ava';}
function fail(e){networkFailure=Boolean(e.transport);stopWatch();$('error').textContent=e.message;}
function robot(x,y,d,size){return `<g transform="translate(${x*size} ${y*size})"><rect x="${size*.18}" y="${size*.18}" width="${size*.64}" height="${size*.65}" rx="${size*.2}" fill="#30b0be" stroke="#197687"/><rect x="${size*.28}" y="${size*.35}" width="${size*.44}" height="${size*.22}" rx="3" fill="#dbf5ed"/><circle cx="${size*.4}" cy="${size*.45}" r="${size*.035}" fill="#24465a"/><circle cx="${size*.6}" cy="${size*.45}" r="${size*.035}" fill="#24465a"/><g transform="rotate(${90*d} ${size*.5} ${size*.5})"><path d="M${size*.5} 1 L${size*.35} ${size*.16} H${size*.65}Z" fill="#fff9dc"/></g></g>`;}
function tile(t,x,y,size,keys=0){const color={wall:'#344254',floor:'#f2eedb',lamp:'#ddd4aa',unknown:'#121930'},red=t?.endsWith('red'),bit=red?1:2;let glyph='';if(t==='lamp')glyph=`<circle cx="${size/2}" cy="${size/2}" r="${size*.3}" fill="#e8ae39"/>`;if(t?.startsWith('key-'))glyph=`<text x="${size*.25}" y="${size*.7}" fill="${red?'#b94e68':'#477dc4'}">⚿</text>`;if(t?.startsWith('door-'))glyph=`<rect x="${size*.2}" y="2" width="${size*.6}" height="${size-4}" fill="${keys&bit?'#f2eedb':red?'#e3a1b0':'#9ab8df'}" stroke="${red?'#b94e68':'#477dc4'}"/><text x="${size/2}" y="${size*.65}" text-anchor="middle" font-size="${size*.4}">${red?'R':'B'}</text>`;return `<g transform="translate(${x*size} ${y*size})"><rect width="${size-1}" height="${size-1}" rx="2" fill="${color[t]??color.floor}"/>${glyph}</g>`;}
function drawWorld(w,editor=false){const large=w.mapName==='editable-keys-doors-v2',lo=large?1:4,hi=large?14:11,size=30;const seen=new Set(state.observation.flat().filter(c=>c.visible).map(c=>`${c.world_x},${c.world_y}`));let cells='';for(let y=0;y<16;y++)for(let x=0;x<16;x++){let t=w.tiles[y][x];if(t.startsWith('key-')&&((w.agent.keys??0)&(t.endsWith('red')?1:2)))t='floor';if(!editor&&!$('reveal').checked&&!seen.has(`${x},${y}`))t='unknown';cells+=tile(t,x,y,size,w.agent.keys??0);}let targets='';if(editor)for(let y=lo;y<=hi;y++)for(let x=lo;x<=hi;x++)targets+=`<g role="button" tabindex="0" data-cell="${x},${y}" aria-label="Column ${x}, row ${y}: ${w.tiles[y][x]}"><rect x="${x*size}" y="${y*size}" width="${size}" height="${size}" fill="transparent"/></g>`;return `<svg viewBox="${large?'0 0 480 480':'90 90 300 300'}" role="${editor?'group':'img'}" aria-label="${editor?'Editable world':'Ava world'}"><rect width="480" height="480" fill="#121930" rx="12"/>${cells}${robot(w.agent.x,w.agent.y,w.agent.direction,size)}${targets}</svg>`;}
function render(){
 $('identity').textContent=`${state.parameters.toLocaleString()} parameters · ${state.steps} Python updates · ${state.origin} · state ${state.state_sha256.slice(0,16)}`;
 const ended=state.reached||state.repeated||state.world.tick>=state.move_limit;
 $('journey-status').textContent=state.reached?`Lamp reached in ${state.world.tick} moves.`:state.repeated?'Stopped after repeated pose. This model did not complete the journey.':state.world.tick>=state.move_limit?'Move limit reached.':`${state.world.tick} / ${state.move_limit} moves · ${state.world_changed?'Changed room; same learned weights.':'Current room.'}`;
 $('facing').value=state.layout.agent.direction;
 $('world').innerHTML=drawWorld(state.world);$('editor').innerHTML=drawWorld(state.layout,true);
 $('vision').innerHTML=`<svg viewBox="0 0 216 216" role="img" aria-label="Python local vision">${state.observation.flatMap((r,y)=>r.map((c,x)=>tile(c.visible?c.tile:'unknown',x,y,24,c.doorOpen?3:0))).join('')}${robot(4,8,0,24)}</svg>`;
 $('inventory').textContent=`Facing ${['north','east','south','west'][state.world.agent.direction]} · ${state.world.agent.keys===undefined?'small-room encoding':`Carried keys: ${state.world.agent.keys&1?'red ':''}${state.world.agent.keys&2?'blue':''}${state.world.agent.keys===0?'none':''}`}`;
 const d=state.last_decision;$('probabilities').replaceChildren();
 if(d)for(let i=0;i<3;i++){const row=document.createElement('div'),label=document.createElement('span'),meter=document.createElement('meter'),value=document.createElement('span');label.textContent=actions[i];meter.min=0;meter.max=1;meter.value=d.trace.probabilities[i];value.textContent=(100*meter.value).toFixed(1)+'%';row.append(label,meter,value);$('probabilities').append(row);}
 $('memory-status').textContent=d?`${d.action} · ${d.trace.memoryReads.map(m=>`${m.bank}: ${m.records} records read`).join(' · ')} · used state ${d.state_sha256.slice(0,16)}`:'No action yet. Journey memory is empty.';
 $('decision').textContent=d?JSON.stringify(d,null,2):'Take a move to inspect actual model values.';
 $('training-status').textContent=`${state.busy?state.phase:'Paused / ready'} · round ${state.epoch}/${state.settings.epochs} · ${state.done}/${state.total??'not yet generated'} examples this round · ${state.steps} optimizer updates · last batch loss ${state.loss===null?'—':state.loss.toFixed(6)}`;
 if(state.history.length)$('training-status').textContent+=` · last practice move agreement ${(100*state.history.at(-1).validation.accuracy).toFixed(1)}%`;
 $('progress').max=state.total||1;$('progress').value=state.done;
 const q=state.qualification;$('qualification').textContent=state.check_progress&&state.busy&&state.phase==='checking journeys'?`Move ${state.check_progress.move}; ${state.check_progress.finished}/${state.check_progress.total} journeys finished`:q?`${q.completed}/${q.total} reached · ${q.interrupted?'interrupted, incomplete journeys are unresolved':'all reachable starts'} · state ${q.state_sha256.slice(0,16)}`:'No Python journey check yet. Historical browser results are not current qualification.';
 $('world-status').textContent=`${state.starts} reachable starting poses · ${state.reachable?'A route exists; model success is not guaranteed.':'No route from Ava. Open a path or move the matching key.'}`;
 $('stages').replaceChildren(...state.config.capabilities.map(c=>{const el=document.createElement('div');el.textContent=Object.entries(c).map(([k,v])=>`${k}: ${v}`).join(' · ');return el;}));
 $('model-info').textContent=JSON.stringify({architecture:state.config,parameters:state.parameters,weights:state.state_sha256,checkpoint:state.checkpoint_sha256,source_checkpoint:state.base.checkpoint_sha256,implementation:state.implementation,framework:state.framework},null,2);
 for(const id of ['step','watch'])$(id).disabled=state.busy||acting||ended||!state.reachable;
 for(const id of ['retry','another','check','train-start','apply-preset','expand','facing','examples'])$(id).disabled=state.busy||acting;
 $('pause').disabled=!state.busy;
 for(const el of [$('epochs'),$('rate'),$('seed'),...$('situations').querySelectorAll('input')])el.disabled=state.busy||acting;
 if(state.error)$('error').textContent=state.error;
 if(ended||state.busy)stopWatch();
 $('editor').querySelectorAll('[data-cell]').forEach(el=>{const click=()=>{if(state.busy)return;const [x,y]=el.dataset.cell.split(',').map(Number);operate('world',{x,y,tool:$('tool').value==='wall'&&state.layout.tiles[y][x]==='wall'?'erase':$('tool').value});};el.addEventListener('click',click);el.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();click();}});});
}
async function refresh(){if(polling||acting)return;polling=true;const g=generation;try{const next=await api('status');if(g===generation){state=next;if(networkFailure){$('error').textContent='';networkFailure=false;}render();}}catch(e){if(g===generation)fail(e);}finally{polling=false;}}
function operate(path,body={}){
 const task=operationQueue.then(async()=>{
  if(path==='train'||path==='check')state.check_progress=null;
  acting=true;generation++;$('error').textContent='';if(state)render();
  try{const result=await api(path,body);state=result.showcase==='ava'?result:await api('status');render();}
  catch(e){fail(e);}finally{acting=false;if(state)render();}
 });
 operationQueue=task.catch(()=>{});return task;
}
async function move(){await operate('step');if(watching)timer=setTimeout(move,450);}
$('step').onclick=()=>{stopWatch();move();};$('watch').onclick=()=>{if(watching){stopWatch();return;}watching=true;$('watch').textContent='Stop watching';move();};
$('retry').onclick=()=>{stopWatch();operate('retry');};$('another').onclick=()=>{stopWatch();operate('another');};
$('reveal').onchange=()=>render();$('expand').onclick=()=>operate('world',{expand:true});
$('apply-preset').onclick=()=>operate('world',{preset:$('preset').value,large:state.layout.mapName==='editable-keys-doors-v2'||$('preset').value==='locks'});
$('facing').onchange=()=>operate('world',{direction:Number($('facing').value)});
$('train-start').onclick=async()=>{stopWatch();try{await operate('train',{settings:{epochs:Number($('epochs').value),rate:Number($('rate').value),seed:Number($('seed').value)},selection:[...$('situations').querySelectorAll('input:checked')].map(e=>e.value)});}catch(e){fail(e);}};
$('pause').onclick=async()=>{try{await api('pause',{});}catch(e){fail(e);}};
$('check').onclick=()=>operate('check');
$('examples').onclick=async()=>{try{$('examples-info').textContent=JSON.stringify(await api('examples'),null,2);}catch(e){fail(e);}};
async function savePractice(){
 if(!state||state.busy)return;
 try{await operate('settings',{settings:{epochs:Number($('epochs').value),rate:Number($('rate').value),seed:Number($('seed').value)},selection:[...$('situations').querySelectorAll('input:checked')].map(e=>e.value)});}catch(e){fail(e);}
}
for(const el of [$('epochs'),$('rate'),$('seed'),...$('situations').querySelectorAll('input')])el.addEventListener('change',savePractice);
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>{stopWatch();for(const id of ['play','train','model'])$(id).hidden=id!==b.dataset.tab;for(const el of document.querySelectorAll('[data-tab]'))el.setAttribute('aria-current',String(el===b));};
await refresh();if(state){$('epochs').value=state.settings.epochs;$('rate').value=state.settings.rate;$('seed').value=state.settings.seed;$('facing').value=state.layout.agent.direction;for(const el of $('situations').querySelectorAll('input'))el.checked=state.selection.includes(el.value);setInterval(refresh,1500);}
