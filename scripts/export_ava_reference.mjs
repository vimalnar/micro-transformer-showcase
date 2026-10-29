// Read-only reference execution. Never touches Browser Lab storage or source files.
import fs from 'node:fs';
import crypto from 'node:crypto';
import {gzipSync} from 'node:zlib';
import {fileURLToPath} from 'node:url';
const browser=process.env.BROWSER_LAB_ROOT??fileURLToPath(new URL('../../micro-transformer-browser-lab',import.meta.url));
const {presetWorld,expandedPreset,mapFeatures,teachingPlan,worldIdentity}=await import(browser+'/src/editable-world.js');
const {observe,stepWorld}=await import(browser+'/src/micro-world.js');
const {mapTokens}=await import(browser+'/src/map-token-adapter.js');
const {compileCapabilities,initialCapabilities,memoryRecord,memoryTokens,newCapability,CAPABILITIES}=await import(browser+'/src/capability-kit.js');
const {situationData}=await import(browser+'/src/training-situations.js');
const {CausalMapAgent,tf}=await import(browser+'/src/causal-map-agent.js');
const {roundExamples}=await import(browser+'/src/navigation-training.js');
await tf.setBackend('cpu');await tf.ready();
const hash=x=>crypto.createHash('sha256').update(JSON.stringify(x)).digest('hex');
const worlds=[];
for(const large of [false,true])for(const kind of ['empty','detour','zigzag',...(large?['locks']:[])]){
 const w=large?expandedPreset(kind):presetWorld(kind),p=teachingPlan(w);
 const data=situationData(w,['layout'],true);
 worlds.push({world:w,identity:worldIdentity(w),features:mapFeatures(w),tokens:mapTokens(mapFeatures(w)),planDigest:hash(p.examples.map(e=>[e.id,e.label,e.optimal,e.cost])),reachable:p.reachable,trials:p.trials,examples:p.examples.length,dataDigest:hash(data.train.map(e=>[e.id,mapTokens(e.features),e.label,e.memory])),train:data.train.length,shuffle:roundExamples(data.train,42,1).slice(0,12).map(e=>[e.id,e.memory])});
}
const source=JSON.parse(fs.readFileSync(browser+'/public/trained-example.json','utf8'));
const all=Object.keys(CAPABILITIES).map(type=>({...newCapability(type),...(type==='depth'?{layers:1}:{})}));
const configs=[source.config,compileCapabilities({...initialCapabilities(),width:16,capabilities:all}),compileCapabilities({...initialCapabilities(),width:16,capabilities:[{type:'refinement',multiplier:2,activation:'relu'},{type:'perspectives',heads:2},{type:'archive',capacity:4}]}),compileCapabilities({...initialCapabilities(),width:16,capabilities:[{type:'depth',layers:1}]})];
configs.push(...[
 [{type:'context',window:8},{type:'object',focus:'ava'},{type:'recent',capacity:2},{type:'archive',capacity:16},{type:'workspace',slots:4},{type:'perspectives',heads:1},{type:'refinement',multiplier:4,activation:'relu'},{type:'depth',layers:2}],
 [{type:'context',window:16},{type:'object',focus:'both'},{type:'recent',capacity:8},{type:'archive',capacity:4},{type:'workspace',slots:16},{type:'perspectives',heads:4},{type:'refinement',multiplier:1,activation:'gelu'},{type:'depth',layers:3}],
 [{type:'depth',layers:5},{type:'context',window:128}]
].map(capabilities=>compileCapabilities({...initialCapabilities(),width:16,capabilities})));
const models=[];
for(let i=0;i<configs.length;i++){
 const model=new CausalMapAgent(configs[i],42);if(i===0)model.loadWeights(source.weights);
 const cases=[];
 for(const w of [presetWorld('detour'),expandedPreset('locks')]){
  const features=mapFeatures(w),record=memoryRecord(features,0),history=[record,memoryRecord(features,1)];
  for(const memory of [[],history]){const t=model.inspectFeatures(features,memory);cases.push({features,memory,tokens:t.tokens,logits:t.vocabLogits,probabilities:t.probabilities,memoryReads:t.memoryReads,stages:t.passes[0].pieces.map(p=>({id:p.id,state:p.state}))});}
 }
 let update=null;
 if(i===1){const features=mapFeatures(presetWorld()),examples=[{features,label:0,memory:[memoryRecord(features,1)]}],optimizer=tf.train.adam(.0005,.9,.999,1e-8);const gradients=tf.tidy(()=>{const ts=examples.map(e=>mapTokens(e.features)),ids=tf.tensor2d(ts,[ts.length,ts[0].length],'int32'),targets=tf.oneHot(tf.tensor1d(examples.map(e=>96+e.label),'int32'),128);const r=tf.variableGrads(()=>tf.losses.softmaxCrossEntropy(targets,model.forward(ids,false,model.memoryBatch(examples)).logits.slice([0,ts[0].length-1,0],[ts.length,1,128]).reshape([ts.length,128])).mean(),model.variables);return Object.fromEntries(Object.entries(model.weights).map(([name,v])=>[name,Array.from(r.grads[v.name].dataSync())]));});const loss=model.trainExamples(examples,optimizer,.0005,.01,1);update={examples,loss,gradients,weights:model.exportWeights()};optimizer.dispose();}
 models.push({config:configs[i],source:i===0?'trained-example':'seed42',cases,update});model.dispose();
}
const observations=[];
for(const w of [presetWorld(),expandedPreset('locks')])for(const tick of [0,59,60,69,70,89,90]){
 w.tick=tick;observations.push({world:structuredClone(w),observation:observe(w),steps:['forward','turn-left','turn-right','wait'].map(action=>({action,...stepWorld(w,action)}))});
}
const situations=[];
for(const [w,selection] of [[source.editorWorld,['layout','lamps','obstacles','routes']],[expandedPreset('locks'),['layout','lamps','obstacles','routes','locks']]]){const data=situationData(w,selection,true);situations.push({world:w,selection,worldIdentities:data.worlds.map(worldIdentity),train:data.train.length,digest:hash(data.train.map(e=>[e.id,mapTokens(e.features),e.label,e.memory]))});}
const fixture={worlds,models,observations,situations};
fs.mkdirSync('tests/fixtures',{recursive:true});fs.writeFileSync('tests/fixtures/ava-browser.json.gz',gzipSync(JSON.stringify(fixture)));
console.log(JSON.stringify({worlds:worlds.length,models:models.length,cases:models.reduce((s,m)=>s+m.cases.length,0),bytes:fs.statSync('tests/fixtures/ava-browser.json.gz').size}));
