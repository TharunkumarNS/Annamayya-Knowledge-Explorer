/* Remix Studio: live "modern touch" effects on the packaged Annamayya recordings.
   Pure browser (Web Audio API). Speed changes pitch together with tempo, like a record. */
(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  const PRESETS = {
    original: { label: 'Original',        speed: 1.00, reverb: 0.00, warmth: 0.00, bass: 0, crackle: 0.00 },
    slowed:   { label: 'Slowed + Reverb', speed: 0.80, reverb: 0.90, warmth: 0.45, bass: 5,  crackle: 0.00 },
    lofi:     { label: 'Lo-fi Chill',     speed: 0.90, reverb: 0.40, warmth: 0.90, bass: 3,  crackle: 0.80 },
    temple:   { label: 'Temple Hall',     speed: 1.00, reverb: 1.00, warmth: 0.15, bass: 0,  crackle: 0.00 },
    sped:     { label: 'Sped-up',         speed: 1.25, reverb: 0.10, warmth: 0.00, bass: 0,  crackle: 0.00 },
    club:     { label: 'Bass Boost',      speed: 1.00, reverb: 0.10, warmth: 0.00, bass: 12, crackle: 0.00 }
  };
  const SLIDERS = [
    ['speed',   'Speed',   0.6, 1.4, 0.01, v => `${Number(v).toFixed(2)}×`],
    ['reverb',  'Reverb',  0,    1,    0.01, v => `${Math.round(v * 100)}%`],
    ['warmth',  'Warmth (soften highs)', 0, 1, 0.01, v => `${Math.round(v * 100)}%`],
    ['bass',    'Bass',    0,    18,   1,    v => `+${v} dB`],
    ['crackle', 'Vinyl crackle', 0, 1, 0.01, v => `${Math.round(v * 100)}%`]
  ];

  let recordings = [], params = { ...PRESETS.slowed }, remixOn = true, activePreset = 'slowed';
  let ctx = null, chain = null, mediaSource = null, rafId = null;

  /* ---------- audio graph, shared by live playback and the offline export ---------- */
  function makeImpulse(c, seconds = 3, decay = 2.6) {
    const len = Math.floor(c.sampleRate * seconds), buf = c.createBuffer(2, len, c.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
    }
    return buf;
  }
  function makeCrackle(c) {
    const len = c.sampleRate * 4, buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) {
      d[i] = (Math.random() * 2 - 1) * 0.03;                       // faint hiss
      if (Math.random() < 0.0006) d[i] += (Math.random() * 2 - 1) * 0.9; // pops
    }
    return buf;
  }
  function createChain(c, input) {
    const warm = c.createBiquadFilter(); warm.type = 'lowpass'; warm.Q.value = 0.7;
    const bass = c.createBiquadFilter(); bass.type = 'lowshelf'; bass.frequency.value = 220;
    const dry = c.createGain(), wet = c.createGain(), conv = c.createConvolver(), master = c.createGain();
    conv.buffer = makeImpulse(c, 4.5, 2.0);
    input.connect(warm); warm.connect(bass);
    bass.connect(dry); dry.connect(master);
    bass.connect(conv); conv.connect(wet); wet.connect(master);
    const noise = c.createBufferSource(); noise.buffer = makeCrackle(c); noise.loop = true;
    const noiseGain = c.createGain(); noise.connect(noiseGain); noiseGain.connect(master); noise.start(0);
    master.gain.value = 0.9;
    const analyser = c.createAnalyser(); analyser.fftSize = 128; analyser.smoothingTimeConstant = 0.8;
    master.connect(analyser); analyser.connect(c.destination);
    function set(p) {
      warm.frequency.value = 20000 * Math.pow(1000 / 20000, p.warmth);   // 20 kHz -> 2.5 kHz
      bass.gain.value = p.bass;
      wet.gain.value = p.reverb * 2.5;
      dry.gain.value = 1 - 0.6 * p.reverb;
      noiseGain.gain.value = p.crackle * 0.7;
    }
    return { set, analyser, master };
  }

  /* ---------- live playback ---------- */
  function current() { return remixOn ? params : PRESETS.original; }
  function applyParams() {
    const audio = $('#remix-audio'), p = current();
    audio.playbackRate = p.speed;
    if (chain) chain.set(p);
    SLIDERS.forEach(([k, , , , , fmt]) => {
      const el = $(`#rmx-${k}`), out = $(`#rmx-${k}-out`);
      if (el) { el.value = params[k]; out.textContent = fmt(params[k]); }
    });
    document.querySelectorAll('.preset-chip').forEach(b => b.classList.toggle('active', b.dataset.preset === activePreset));
  }
  function ensureContext() {
    if (ctx) { if (ctx.state === 'suspended') ctx.resume(); return; }
    const audio = $('#remix-audio');
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    mediaSource = ctx.createMediaElementSource(audio);
    chain = createChain(ctx, mediaSource);
    applyParams();
    draw();
  }
  function draw() {
    const canvas = $('#remix-viz'); if (!canvas || !chain) return;
    const g = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
    const data = new Uint8Array(chain.analyser.frequencyBinCount);
    chain.analyser.getByteFrequencyData(data);
    g.clearRect(0, 0, w, h);
    const bars = data.length, bw = w / bars;
    for (let i = 0; i < bars; i++) {
      const bh = Math.max(2, (data[i] / 255) * h);
      g.fillStyle = i % 2 ? '#702b2b' : '#a58a4e';
      g.fillRect(i * bw + 1, h - bh, bw - 3, bh);
    }
    rafId = requestAnimationFrame(draw);
  }

  /* ---------- offline export (WAV) ---------- */
  function encodeWav(buffer) {
    const ch = buffer.numberOfChannels, len = buffer.length, sr = buffer.sampleRate;
    const out = new DataView(new ArrayBuffer(44 + len * ch * 2));
    const str = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    str(0, 'RIFF'); out.setUint32(4, 36 + len * ch * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, ch, true);
    out.setUint32(24, sr, true); out.setUint32(28, sr * ch * 2, true); out.setUint16(32, ch * 2, true);
    out.setUint16(34, 16, true); str(36, 'data'); out.setUint32(40, len * ch * 2, true);
    const chans = [...Array(ch).keys()].map(i => buffer.getChannelData(i));
    let o = 44;
    for (let i = 0; i < len; i++) for (let c = 0; c < ch; c++) {
      const s = Math.max(-1, Math.min(1, chans[c][i])); out.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true); o += 2;
    }
    return new Blob([out], { type: 'audio/wav' });
  }
  async function exportRemix() {
    const rec = recordings[$('#remix-song').selectedIndex]; if (!rec) return;
    const btn = $('#remix-export'), msg = $('#remix-message');
    const p = params, sr = 44100;
    btn.disabled = true; msg.hidden = false; msg.className = 'message'; msg.textContent = 'Rendering your remix…';
    try {
      const bytes = await (await fetch(`/media/${encodeURIComponent(rec.file)}`)).arrayBuffer();
      const src = await new OfflineAudioContext(2, 1, sr).decodeAudioData(bytes);
      const tail = 3 * (p.reverb > 0 ? 1 : 0);
      const off = new OfflineAudioContext(2, Math.ceil((src.duration / p.speed + tail) * sr), sr);
      const node = off.createBufferSource(); node.buffer = src; node.playbackRate.value = p.speed;
      const c = createChain(off, node); c.set(p); node.start(0);
      const rendered = await off.startRendering();
      const url = URL.createObjectURL(encodeWav(rendered));
      const a = document.createElement('a'); a.href = url; a.download = `${rec.file.replace(/\.mp3$/i, '')}_${activePreset}_remix.wav`;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000);
      msg.className = 'message success'; msg.textContent = 'Remix downloaded.';
    } catch (e) {
      msg.className = 'message error'; msg.textContent = `Could not render the remix (${e.message || e}).`;
    } finally { btn.disabled = false; }
  }

  /* ---------- UI ---------- */
  function showSong() {
    const rec = recordings[$('#remix-song').selectedIndex]; if (!rec) return;
    const audio = $('#remix-audio'); audio.pause(); audio.src = `/media/${encodeURIComponent(rec.file)}`; audio.load();
    $('#remix-meta').textContent = [['Singer', rec.singer], ['Raga', rec.raga], ['Tala', rec.tala]].filter(x => x[1]).map(x => `${x[0]}: ${x[1]}`).join(' · ');
    applyParams();
  }
  function setPreset(key) {
    activePreset = key; params = { ...PRESETS[key] }; remixOn = true; $('#remix-on').checked = true; applyParams();
  }
  let started = false;
  async function initRemix() {
    const root = $('#remix-view'); if (!root || started) return;
    started = true;
    $('#remix-presets').innerHTML = Object.entries(PRESETS).map(([k, v]) => `<button type="button" class="preset-chip" data-preset="${k}">${esc(v.label)}</button>`).join('');
    $('#remix-sliders').innerHTML = SLIDERS.map(([k, label, min, max, step]) =>
      `<label class="slider-row"><span>${esc(label)}</span><input type="range" id="rmx-${k}" min="${min}" max="${max}" step="${step}" value="${params[k]}"><output id="rmx-${k}-out"></output></label>`).join('');
    document.querySelectorAll('.preset-chip').forEach(b => b.addEventListener('click', () => setPreset(b.dataset.preset)));
    SLIDERS.forEach(([k]) => $(`#rmx-${k}`).addEventListener('input', e => { params[k] = Number(e.target.value); activePreset = ''; applyParams(); }));
    $('#remix-on').addEventListener('change', e => { remixOn = e.target.checked; applyParams(); });
    $('#remix-song').addEventListener('change', showSong);
    $('#remix-audio').addEventListener('play', ensureContext);
    $('#remix-export').addEventListener('click', exportRemix);
    try {
      const data = await (await fetch('/api/recordings')).json();
      recordings = data.recordings || [];
      $('#remix-song').innerHTML = recordings.map(r => `<option>${esc(r.title)}${r.singer ? ' — ' + esc(r.singer) : ''}</option>`).join('');
      showSong();
    } catch (e) { $('#remix-meta').textContent = 'Could not load the recording list.'; }
  }
  window.openRemix = function (file) {
    if (typeof changeView === 'function') changeView('remix');
    const i = recordings.findIndex(r => r.file === decodeURIComponent(file || ''));
    if (i >= 0) { $('#remix-song').selectedIndex = i; showSong(); }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  window.__remixTest = { PRESETS, createChain, encodeWav };   // handy for quick checks
  document.addEventListener('DOMContentLoaded', initRemix);
  if (document.readyState !== 'loading') initRemix();
})();