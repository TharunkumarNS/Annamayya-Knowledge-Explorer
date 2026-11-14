const $ = (s) => document.querySelector(s);
let searchResults = [];
let askHistory = [];
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const excerpt = (value, limit=300) => { const text = String(value || '').trim(); return text.length > limit ? text.slice(0, text.lastIndexOf(' ', limit)) + '…' : text; };
async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
function message(selector, text='', kind='') { const el=$(selector); el.textContent=text; el.className=`message ${kind}`; el.hidden=!text; }
function changeView(view) {
  document.querySelectorAll('.nav-link').forEach(x=>x.classList.toggle('active',x.dataset.view===view));
  $('#discover-view').hidden=view!=='discover'; $('#merge-view').hidden=view!=='merge'; $('#connections-view').hidden=view!=='connections';
}
document.querySelectorAll('.nav-link').forEach(x=>x.addEventListener('click',()=>changeView(x.dataset.view)));
$('#audio').addEventListener('change', e=> { const f=e.target.files[0]; $('#audio-label').innerHTML=f?`${esc(f.name)} <small>${(f.size/1048576).toFixed(1)} MB · selected</small>`:'Add an audio recording <small>WAV or MP3</small>'; });
async function init(){
  try { const state=await api('/api/health'); $('#service-status').textContent='API connected'; $('#service-status').classList.add('connected'); $('#graph-option').hidden=!state.neo4j_configured;
    $('#collection-stats').textContent=`${state.tables.compositions} compositions · ${state.tables.textrecords} text records · ${state.tables.audiorecords} recordings`;
  } catch(e) { $('#service-status').textContent='API unavailable'; message('#search-message',e.message,'error'); }
  loadDuetPairs();
}
let duetPairs=[];let duetResultUrl=null;
async function loadDuetPairs(){
  const select=$('#duet-pair');
  try{const data=await api('/api/duets/pairs');duetPairs=data.pairs||[];
    select.innerHTML=duetPairs.length?duetPairs.map(p=>`<option value="${p.pair_id}">${esc(p.display_name)}${p.composition_id?` · ${esc(p.composition_id)}`:''}</option>`).join(''):'<option value="">No complete pairs are available</option>';
    select.disabled=!duetPairs.length;$('#merge-button').disabled=!duetPairs.length;
    if(duetPairs.length)showDuetPreviews();else message('#duet-pair-message','Add a complete pair to the separate duet catalog to enable merging.');
  }catch(e){select.innerHTML='<option value="">Pair catalog unavailable</option>';select.disabled=true;$('#merge-button').disabled=true;message('#duet-pair-message',e.message,'error');}
}
function showDuetPreviews(){const id=$('#duet-pair').value;if(!id)return;$('#duet-previews').hidden=false;$('#vintage-preview').src=`/api/duets/pairs/${id}/track/vintage`;$('#modern-preview').src=`/api/duets/pairs/${id}/track/modern`;}
$('#duet-pair').addEventListener('change',showDuetPreviews);
$('#merge-button').addEventListener('click',async()=>{
  const id=$('#duet-pair').value;if(!id)return;const button=$('#merge-button');button.disabled=true;button.textContent='Merging…';$('#duet-output').hidden=true;message('#duet-pair-message','Loading pair settings and generating the duet…');
  try{const result=await api(`/api/duets/pairs/${id}/merge`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({crossfade_duration_ms:Number($('#duet-crossfade').value),bell_gain_db:Number($('#duet-bell-gain').value)})});
    $('#duet-result').src=result.audio_url;$('#duet-download').href=result.audio_url;$('#duet-download').download=result.filename;$('#duet-output').hidden=false;message('#duet-pair-message','Duet created. Play it here or download the WAV.','success');
  }catch(e){message('#duet-pair-message',e.message,'error');}finally{button.disabled=false;button.innerHTML='Create duet <span>→</span>';}
});
$('#search-form').addEventListener('submit', async e=>{
 e.preventDefault(); searchResults=[]; askHistory=[]; $('#results-section').hidden=true; $('#results').replaceChildren(); $('#ask-section').hidden=true; message('#search-message','Preparing indexes and searching…');
 const button=$('#search-button'); button.disabled=true; button.textContent='Searching…';
 const form=new FormData(); form.append('text_query',$('#query').value); form.append('enrich_graph',$('#enrich-graph').checked?'true':'false'); const file=$('#audio').files[0]; if(file)form.append('audio',file);
 try { const data=await api('/api/search',{method:'POST',body:form}); searchResults=data.results||[]; message('#search-message',data.graph_warning||Object.values(data.branch_errors||{}).join(' · '),data.graph_warning?'':'error'); renderResults(searchResults); }
 catch(err){message('#search-message',err.message,'error');}
 finally{button.disabled=false;button.innerHTML='Find compositions <span>→</span>';}
});
function renderResults(results){
 $('#results-section').hidden=false; $('#result-count').textContent=`${results.length} match${results.length===1?'':'es'}`;
 if(!results.length){$('#results').innerHTML='<div class="message">No matching compositions were found. Try different wording or another recording.</div>';return;}
 $('#results').innerHTML=results.map((r,i)=>{
   const local=r.metadata||{}, title=r.canonical_title||local.canonical_title||r.composition_id||'Composition'; const modalities=r.modalities||[];
   const badges=modalities.map(x=>`<span class="badge">${x==='text'?'Text match':'Audio match'}</span>`).join('');
   const text=r.text_evidence?.best_match||{}; const textRows=[['Telugu lyrics','lyrics'],['Romanized lyrics','transliteration'],['English meaning','translation']];
   const grid=textRows.map(([label,kind])=>{let row=kind===text.text_type?text:null; if(!row){row=(r.local_text_records||[]).find(t=>t.text_type===kind);} return row?.content?`<div><div class="excerpt-label">${label}</div><p class="excerpt">${esc(excerpt(row.content))}</p></div>`:'';}).join('');
   const audioId=r.best_matching_audio_id||r.audio_evidence?.best_matching_audio_id; const audioPath=r.audio_evidence?.recording_evidence?.find?.(a=>a.audio_id===audioId)?.audio_path||r.local_audio_records?.[0]?.audio_path;
   const src=(audioPath||'').replace(/^Recordings[\\/]/,''); const audio=src?`<audio controls preload="none" src="/media/${encodeURIComponent(src)}"></audio>`:'';
   const meta=[['Singer',local.singer],['Raga',local.raga],['Tala',local.tala]].filter(x=>x[1]).map(x=>`${x[0]}: ${esc(x[1])}`).join(' · ');
   const ev=[]; if(r.text_score!=null)ev.push(`Text ${r.text_rank||''} · ${Number(r.text_score).toFixed(4)}`);if(r.audio_score!=null)ev.push(`Audio ${r.audio_rank||''} · ${Number(r.audio_score).toFixed(4)}`);if(r.fused_score!=null)ev.push(`Combined rank score ${Number(r.fused_score).toFixed(5)} (not a probability)`);
   const links=[['text_source_url','Lyrics source'],['transliteration_source_url','Transliteration source'],['translation_source_url','Meaning source'],['audio_source_url','Recording source']].filter(([k])=>local[k]).map(([k,l])=>`<a href="${esc(local[k])}" target="_blank" rel="noopener noreferrer">${l}</a>`).join('');
   return `<article class="result-card"><div class="result-top"><div><h3 class="result-title">${i+1}. ${esc(title)}</h3><div class="result-id">${esc(r.composition_id)}</div></div></div>${badges}<div class="metadata">${meta}</div><div class="excerpt-grid">${grid}</div>${audio}<div class="source-links">${links}</div><div class="card-actions"><button class="subtle-button details-button" data-id="${esc(r.composition_id)}">Full lyrics & song details</button><button class="subtle-button evidence-button">Why this result?</button></div><div class="evidence" hidden>${ev.map(x=>`<p>${esc(x)}</p>`).join('')||'<p>Ranking evidence is unavailable.</p>'}${r.graph_data?`<p>Relationships returned by Neo4j:</p><div class="result-graph" id="result-graph-${i}"></div>`:''}</div></article>`;
 }).join('');
 document.querySelectorAll('.details-button').forEach(b=>b.addEventListener('click',()=>showDetails(b.dataset.id)));
 document.querySelectorAll('.evidence-button').forEach((b,i)=>b.addEventListener('click',()=>{const panel=b.closest('.result-card').querySelector('.evidence');panel.hidden=!panel.hidden;if(!panel.hidden&&searchResults[i].graph_data&&!panel.dataset.graphReady){mountResultGraph(searchResults[i].graph_data,panel.querySelector('.result-graph'));panel.dataset.graphReady='1';}}));
 $('#ask-composition').innerHTML=results.map(r=>`<option value="${esc(r.composition_id)}">${esc(r.canonical_title||r.composition_id)}</option>`).join(''); $('#ask-section').hidden=false;
}
function mountResultGraph(data,canvas){if(!window.vis||!canvas)return;const c=data.composition||{},nodes=[{id:c.composition_id||'composition',label:c.canonical_title||c.composition_id||'Composition',color:'#702b2b'}],edges=[];const center=nodes[0].id;const add=(id,label,type,relation)=>{if(!id)return;nodes.push({id,label:label||id,color:({Singer:'#2b5b70',Raga:'#5b2b70',Tala:'#705b2b',TextRecord:'#a58a4e',AudioRecording:'#2b7045'})[type]||'#777'});edges.push({from:center,to:id,label:relation,arrows:'to'});};for(const [type,key,rel] of [['Singer','singers','PERFORMED_BY'],['Raga','ragas','USES_RAGA'],['Tala','talas','USES_TALA']])for(const value of data[key]||[])add(`${type}:${value}`,value,type,rel);for(const t of data.text_records||[])add(t.text_id,`${t.text_type||'Text'} (${t.text_id})`,'TextRecord','HAS_TEXT');for(const a of data.audio_records||[])add(a.audio_id,`Audio ${a.audio_id}`,'AudioRecording','HAS_RECORDING');new vis.Network(canvas,{nodes:new vis.DataSet(nodes),edges:new vis.DataSet(edges)},{interaction:{hover:true,navigationButtons:true},physics:{stabilization:{iterations:120}},nodes:{shape:'dot',size:17,font:{face:'DM Sans',size:12,color:'#292720'}},edges:{color:'#c6baa0',font:{size:9,color:'#726c5e',align:'middle'},smooth:{type:'continuous'}}});}
async function showDetails(id){try{const d=await api(`/api/compositions/${encodeURIComponent(id)}`);const c=d.composition;$('#details-content').innerHTML=`<h2 class="detail-title">${esc(c.canonical_title||id)}</h2><p class="detail-meta">${[['Singer',c.singer],['Raga',c.raga],['Tala',c.tala]].filter(x=>x[1]).map(x=>`${x[0]}: ${esc(x[1])}`).join(' · ')}</p>${d.text_records.map(t=>`<section class="detail-block"><h3>${esc(t.language||t.text_type)}</h3><p>${esc(t.content)}</p>${t.source_url?`<a class="text-source" href="${esc(t.source_url)}" target="_blank" rel="noopener">Open source</a>`:''}</section>`).join('')||'<p>No text variants are available.</p>'}${d.audio_records.map(a=>`<section class="detail-block"><h3>${esc(a.recording_title||a.audio_id)}</h3><audio class="detail-audio" controls preload="none" src="/media/${encodeURIComponent(a.audio_path.split(/[\\/]/).pop())}"></audio></section>`).join('')}`;$('#details-dialog').showModal();}catch(e){message('#search-message',e.message,'error');}}
$('#details-dialog').querySelector('.dialog-close').addEventListener('click',()=>$('#details-dialog').close());
$('#ask-composition').addEventListener('change',()=>{askHistory=[];$('#answer').hidden=true;});
$('#ask-form').addEventListener('submit',async e=>{e.preventDefault();const question=$('#question').value.trim();if(!question)return;const area=$('#answer');area.hidden=false;area.textContent='Preparing an evidence-grounded explanation…';try{const result=await api('/api/explain',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,composition_id:$('#ask-composition').value,results:searchResults,history:askHistory.slice(-3)})});const answer=result.answer||result.text||JSON.stringify(result);area.textContent=answer;askHistory.push({question,answer});askHistory=askHistory.slice(-3);}catch(err){area.textContent=err.message;}});
$('#graph-mode').addEventListener('change',e=>$('#graph-value').hidden=e.target.value==='all');
$('#connections-form').addEventListener('submit',async e=>{e.preventDefault();message('#connection-message','Loading graph…');$('#graph-canvas').replaceChildren();$('#graph-summary').hidden=true;const mode=$('#graph-mode').value,value=$('#graph-value').value.trim();if(mode!=='all'&&!value){message('#connection-message','Enter a composition ID or exact singer, raga, or tala.','error');return;}const params=new URLSearchParams();if(mode==='composition')params.set('composition_id',value);if(['singer','raga','tala'].includes(mode)){params.set('field',mode);params.set('value',value);}try{const d=await api(`/api/connections?${params}`);const edges=d.edges||[];message('#connection-message',edges.length?'':'Neo4j is reachable, but no connected nodes were returned.');$('#graph-summary').hidden=false;$('#graph-summary').textContent=`${new Set(edges.flatMap(e=>[e.source_id,e.target_id])).size} nodes · ${edges.length} relationships`;
const ns=new Map();const nodes=[],links=[];const colors={Composition:'#702b2b',Singer:'#2b5b70',Raga:'#5b2b70',Tala:'#705b2b',TextRecord:'#a58a4e',AudioRecording:'#2b7045'};for(const edge of edges){for(const [id,label,type] of [[edge.source_id,edge.source_label,edge.source_type],[edge.target_id,edge.target_label,edge.target_type]])if(id&&!ns.has(id)){ns.set(id,true);nodes.push({id,label:label||id,title:`${type||'Node'}: ${label||id}`,color:colors[type]||'#777'});}if(edge.source_id&&edge.target_id)links.push({from:edge.source_id,to:edge.target_id,label:edge.relation||'',arrows:'to',font:{size:10,color:'#726c5e',align:'middle'}});}
if(nodes.length&&window.vis)new vis.Network($('#graph-canvas'),{nodes:new vis.DataSet(nodes),edges:new vis.DataSet(links)},{interaction:{hover:true,navigationButtons:true,keyboard:true},physics:{stabilization:{iterations:180},barnesHut:{gravitationalConstant:-5500,springLength:130}},nodes:{shape:'dot',size:18,font:{face:'DM Sans',size:13,color:'#292720'},borderWidth:1},edges:{color:{color:'#c6baa0',highlight:'#702b2b'},smooth:{type:'continuous'}}});}catch(err){message('#connection-message',err.message,'error');}});
init();
