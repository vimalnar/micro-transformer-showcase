const $=id=>document.getElementById(id);
const devices=document.querySelector('meta[name="showcase"]')?.content==='devices';
const prefix=devices?'/api/devices/':'/api/';
let state,polling=false,deviceBusy=false,latestRevision=-1,generation=0,requestBusy=false,lastAnswer=null,networkFailure=false;
async function api(path,body,method='POST'){
 let r;try{r=await fetch(prefix+path,body===undefined?{}:{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
 catch{const e=Error('Python service unavailable. Reconnecting…');e.transport=true;throw e;}
 const v=await r.json();if(!r.ok)throw Error(v.detail||r.statusText);return v;
}
function fail(e){networkFailure=Boolean(e.transport);$('error').textContent=e.message;}
function recipe(){return{text:$('text').value,rounds:Number($('rounds').value),rate:Number($('rate').value)};}
function renderSimulation(sim){
 if(sim.revision<latestRevision)return;latestRevision=sim.revision;
 $('device-actual').textContent=sim.actual?`Simulated position: ${sim.actual}`:'No event sent';
 $('device-prompt').value=sim.prompt;
 $('device-events').replaceChildren(...sim.events.map(e=>{const li=document.createElement('li');li.textContent=`${e.place} · update ${e.delivered?'delivered':'withheld'}`;return li;}));
 const r=sim.result,a=r?.assessment;
 $('device-report').textContent=sim.stale?'Model updated; reinterpret history':r?(a.valid?a.report:'Unrecognised model reply'):(sim.error?'No report generated':'Waiting for the model');
 $('device-raw').textContent=r?r.prediction:'No prediction yet.';
 $('device-result-info').textContent=r?`${sim.stale?'Previous checkpoint · ':''}${r.stop_reason} · ${r.steps} updates · state ${r.state_sha256.slice(0,16)} · token IDs ${r.token_ids.join(', ')}`:'';
 $('device-feedback').textContent=sim.error?`Prediction failed: ${sim.error}. Event retained; retry the same messages.`:r?`${sim.events.length}/6 events · ${sim.stale?'Saved report is stale. Reinterpret using current weights.':a.matches?'Matches delivered messages.':'Does not match delivered messages.'} Last delivered: ${sim.received}. Actual position: ${sim.actual}.`:`${sim.events.length}/6 events · ${sim.events.length?'Send or reinterpret messages for a real report.':'No messages yet. Learned weights are unchanged.'}`;
 $('device-repeat').disabled=deviceBusy||!sim.events.length;
}
async function refresh(){
 if(polling||deviceBusy||requestBusy)return;polling=true;const g=generation;
 try{
  const next=await api('status');if(g!==generation)return;state=next;if(networkFailure){$('error').textContent='';networkFailure=false;}
  $('identity').textContent=`${state.parameters.toLocaleString()} parameters · ${state.steps} local updates · state ${state.state_sha256.slice(0,16)}`;
  $('progress').max=state.total||1;$('progress').value=state.done;
  $('training-status').textContent=`${state.busy?'Training':state.done&&state.done<state.total?'Paused / resumable':'Ready'} · ${state.done}/${state.total} updates · last loss ${state.loss===null?'—':state.loss.toFixed(6)} · mean practice loss ${state.mean_loss===null?'—':state.mean_loss.toFixed(6)}`;
  $('start').disabled=state.busy;$('pause').disabled=!state.busy;
  $('model-info').textContent=JSON.stringify({showcase:state.showcase,config:state.config,parameters:state.parameters,updates:state.steps,state_sha256:state.state_sha256,checkpoint_sha256:state.checkpoint_sha256,framework:state.framework},null,2);
  if(state.error)$('error').textContent=state.error;
  if(!devices&&lastAnswer){$('answer-label').textContent=lastAnswer.state_sha256===state.state_sha256?'Actual greedy prediction':'Previous checkpoint answer · ask again with current weights';}
  if(devices&&!deviceBusy){const sim=await api('simulation');if(g===generation&&!deviceBusy)renderSimulation(sim);}
 }catch(e){if(g===generation)fail(e);}finally{polling=false;}
}
if($('open-chat'))$('open-chat').onclick=()=>{$('selection').hidden=true;$('chat').hidden=false;};
$('back').onclick=()=>{if(devices){location.href='/';return;}$('selection').hidden=false;$('chat').hidden=true;};
for(const b of document.querySelectorAll('[data-view]'))b.onclick=()=>{
 for(const id of ['play','train','model'])$(id).hidden=id!==b.dataset.view;
 for(const x of document.querySelectorAll('[data-view]'))x.setAttribute('aria-current',String(x===b));
};
if(devices){
 async function operation(path,body){
  if(deviceBusy)return;deviceBusy=true;generation++;$('error').textContent='';
  for(const id of ['device-send','device-repeat','device-clear','device-place','device-delivery'])$(id).disabled=true;
  try{renderSimulation(await api(path,body));}catch(e){fail(e);try{renderSimulation(await api('simulation'));}catch{}}
  finally{deviceBusy=false;for(const id of ['device-send','device-repeat','device-clear','device-place','device-delivery'])$(id).disabled=false;await refresh();}
 }
 $('device-send').onclick=()=>operation('events',{place:$('device-place').value,delivered:$('device-delivery').checked});
 $('device-repeat').onclick=()=>operation('interpret',{});
 $('device-clear').onclick=()=>operation('clear',{});
}else{
 $('ask').onclick=async()=>{
  $('ask').disabled=true;requestBusy=true;generation++;$('error').textContent='';
  try{const r=await api('predict',{prompt:$('prompt').value});lastAnswer=r;$('answer-label').textContent='Actual greedy prediction';$('answered-prompt').textContent='Answered: '+r.prompt;$('answer').textContent=r.prediction||'[no answer token]';$('answer-info').textContent=`${r.stop_reason} · ${r.elapsed_seconds.toFixed(3)}s · ${r.steps} updates · state ${r.state_sha256.slice(0,16)}`;}catch(e){fail(e);}finally{requestBusy=false;$('ask').disabled=false;await refresh();}
 };
 $('failure').onclick=()=>{$('prompt').value='gia move map twelve to cave. eri paint map twelve green. cy paint map twelve blue. what colour map twelve?';};
}
$('start').onclick=async()=>{if(requestBusy)return;requestBusy=true;generation++;$('start').disabled=true;try{$('error').textContent='';await api('train',recipe());}catch(e){fail(e);}finally{requestBusy=false;await refresh();}};
$('pause').onclick=async()=>{try{await api('pause',{});$('training-status').textContent='Pause requested; finishing and saving the current update.';}catch(e){fail(e);}};
let saveTimer;
function draft(){clearTimeout(saveTimer);saveTimer=setTimeout(async()=>{try{await api('draft',recipe(),'PUT');$('draft-status').textContent='Draft saved locally.';}catch(e){fail(e);}},350);}
for(const id of ['text','rounds','rate'])$(id).addEventListener('input',draft);
try{
 await refresh();if(!state)throw Error('Python service unavailable');
 $('text').value=state.draft.text;$('rounds').value=state.draft.rounds;$('rate').value=state.draft.rate;$('vocabulary').textContent=state.tokens.join(' · ');
 const sets=await api('practice');
 for(const [name,lines]of Object.entries(sets)){const b=document.createElement('button');b.textContent=name;b.onclick=()=>{$('text').value=lines.join('\n');draft();};$('practice').append(b);}
 if(!devices)for(const name of ['Locations','Ownership']){const b=document.createElement('button');b.textContent=name;b.onclick=()=>{$('prompt').value=sets[name][0].split('=>')[0].trim();};$('examples').append(b);}
 setInterval(refresh,1200);
}catch(e){fail(e);}
