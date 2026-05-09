// GAME ggml browser demo — loads WASM + GGUF, runs model.infer on a WAV.
// Everything runs client-side; no network traffic after model download.

import createGameGGML from './game_ggml.js';

// ---- small helpers ---------------------------------------------------------

const $ = id => document.getElementById(id);
const log = (...args) => {
    const line = args.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' ');
    $('log').textContent += '\n' + line;
    $('log').scrollTop = $('log').scrollHeight;
    console.log(...args);
};

function fmtBytes(n) {
    const units = ['B','KB','MB','GB'];
    let u = 0;
    while (n >= 1024 && u < units.length - 1) { n /= 1024; ++u; }
    return n.toFixed(n < 10 ? 2 : n < 100 ? 1 : 0) + ' ' + units[u];
}

// ---- cached model fetch (IndexedDB via Cache API) --------------------------

async function fetchModel(dtype) {
    const url = `./assets/game_small_${dtype}.gguf`;
    const cache = await caches.open('game-ggml-v1');
    let resp = await cache.match(url);
    const fresh = !resp;
    if (!resp) {
        log(`fetching ${url} ...`);
        const r = await fetch(url);
        if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
        resp = r.clone();
        await cache.put(url, r);
    } else {
        log(`using cached ${url}`);
    }
    const bytes = await resp.arrayBuffer();
    log(`  → ${fmtBytes(bytes.byteLength)} ${fresh ? '(downloaded)' : '(cache hit)'}`);
    return bytes;
}

// ---- WASM module lifecycle -------------------------------------------------

let Module = null;           // Emscripten module
let model  = null;           // ModelJs instance
let audioBuffer = null;      // most recent decoded Float32Array
let lastNotes = null;        // most recent result

async function ensureModule() {
    if (Module) return Module;
    log('loading game_ggml.wasm ...');
    Module = await createGameGGML({
        locateFile: (p) => p,  // JS, wasm in the same folder
    });
    log(`game_ggml ${Module.version()} · ggml ${Module.ggmlVersion()}`);
    $('buildStats').textContent = `· game_ggml ${Module.version()} · ggml ${Module.ggmlVersion()}`;
    return Module;
}

async function loadModel() {
    $('modelStatus').textContent = '(loading…)';
    $('loadBtn').disabled = true;
    try {
        const M = await ensureModule();
        if (model) { model.delete(); model = null; }
        const dtype = $('modelChoice').value;
        const bytes = await fetchModel(dtype);

        const ptr = M._malloc(bytes.byteLength);
        M.HEAPU8.set(new Uint8Array(bytes), ptr);
        log(`building model (heap at 0x${ptr.toString(16)}) ...`);
        const t0 = performance.now();
        model = M.createModelFromPtr(ptr, bytes.byteLength);
        const dt = performance.now() - t0;
        M._free(ptr);
        log(`model ready in ${dt.toFixed(0)} ms: arch=${model.arch()}, sr=${model.sampleRate()}, mels=${model.nMels()}`);
        $('modelStatus').textContent = `loaded · ${dtype} · ${fmtBytes(bytes.byteLength)}`;
        $('runBtn').disabled = !(audioBuffer && model);
    } catch (e) {
        log('load failed: ' + e.message);
        $('modelStatus').textContent = 'load failed';
    } finally {
        $('loadBtn').disabled = false;
    }
}

// ---- audio decoding --------------------------------------------------------

async function decodeFile(file) {
    log(`decoding ${file.name} (${fmtBytes(file.size)}) ...`);
    const t0 = performance.now();
    const buf = await file.arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    const ab = await ac.decodeAudioData(buf);
    ac.close();
    // Downmix to mono if needed.
    let mono;
    if (ab.numberOfChannels === 1) {
        mono = ab.getChannelData(0).slice();
    } else {
        mono = new Float32Array(ab.length);
        for (let c = 0; c < ab.numberOfChannels; c++) {
            const ch = ab.getChannelData(c);
            for (let i = 0; i < ab.length; i++) mono[i] += ch[i];
        }
        for (let i = 0; i < ab.length; i++) mono[i] /= ab.numberOfChannels;
    }
    if (ab.sampleRate !== 44100) {
        log(`  resampling ${ab.sampleRate} Hz → 44100 Hz ...`);
        mono = linearResample(mono, ab.sampleRate, 44100);
    }
    log(`  → ${mono.length} samples (${(mono.length / 44100).toFixed(1)} s)` +
        ` in ${(performance.now() - t0).toFixed(0)} ms`);
    audioBuffer = mono;
    $('runBtn').disabled = !(model && audioBuffer);
}

// Simple linear interpolation resampler.  Adequate for demo; for production
// use a proper sinc filter.  Audio sources are usually 44.1/48 kHz so we're
// not doing heavy resampling.
function linearResample(src, srIn, srOut) {
    const ratio = srIn / srOut;
    const n = Math.floor(src.length / ratio);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const pos = i * ratio;
        const i0 = Math.floor(pos);
        const frac = pos - i0;
        out[i] = src[i0] * (1 - frac) + (src[i0 + 1] ?? src[i0]) * frac;
    }
    return out;
}

// ---- inference -------------------------------------------------------------

async function transcribe() {
    if (!model || !audioBuffer) return;
    $('runBtn').disabled = true;
    $('runStatus').textContent = 'running…';
    try {
        const M = Module;
        const bytes = audioBuffer.byteLength;
        const wavPtr = M._malloc(bytes);
        // Float32Array aligned copy into HEAPF32.
        M.HEAPF32.set(audioBuffer, wavPtr / 4);

        const lang    = parseInt($('language').value, 10);
        const nsteps  = parseInt($('nsteps').value, 10);
        const seed    = BigInt($('seed').value);
        const seedLo  = Number(seed & 0xFFFFFFFFn);
        const seedHi  = Number((seed >> 32n) & 0xFFFFFFFFn);

        log(`inference: ${audioBuffer.length} samples, lang=${lang}, nsteps=${nsteps}, seed=${seed}`);
        const t0 = performance.now();
        const notes = model.infer(wavPtr, audioBuffer.length, lang, seedLo, seedHi, nsteps);
        const dt = (performance.now() - t0) / 1000;
        M._free(wavPtr);

        const voiced = notes.filter(n => n.voiced);
        const dur = audioBuffer.length / 44100;
        log(`→ ${notes.length} notes (${voiced.length} voiced) in ${dt.toFixed(2)} s` +
            ` (RTF ${(dur / dt).toFixed(1)}×)`);
        $('runStatus').textContent =
            `${voiced.length} notes · ${dt.toFixed(2)} s · RTF ${(dur / dt).toFixed(1)}×`;
        lastNotes = notes;
        draw(notes);
        $('dlBtn').disabled = false;
        $('dlTxtBtn').disabled = false;
    } catch (e) {
        log('inference failed: ' + e.message);
        $('runStatus').textContent = 'failed';
    } finally {
        $('runBtn').disabled = false;
    }
}

// ---- piano-roll rendering --------------------------------------------------

function draw(notes) {
    const cv = $('roll');
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const voiced = notes.filter(n => n.voiced);
    if (voiced.length === 0) {
        ctx.fillStyle = '#888';
        ctx.fillText('no voiced notes', 10, 20);
        return;
    }
    const tEnd   = Math.max(...notes.map(n => n.offset + n.duration));
    const pitches = voiced.map(n => n.pitch);
    const pLo = Math.floor(Math.min(...pitches) - 1);
    const pHi = Math.ceil (Math.max(...pitches) + 1);
    const W = cv.width, H = cv.height;
    const xs = t => t * W / tEnd;
    const ys = p => H - (p - pLo) * H / (pHi - pLo);

    // grid
    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    for (let p = pLo; p <= pHi; p++) {
        ctx.moveTo(0, ys(p)); ctx.lineTo(W, ys(p));
    }
    ctx.stroke();

    // notes
    for (const n of voiced) {
        const x = xs(n.offset);
        const w = Math.max(1, xs(n.offset + n.duration) - x);
        const y = ys(n.pitch);
        ctx.fillStyle = `hsl(${(n.pitch * 13) % 360}, 65%, 50%)`;
        ctx.fillRect(x, y - 3, w, 6);
    }
}

// ---- download .mid / .txt --------------------------------------------------

function downloadMidi() {
    if (!lastNotes) return;
    const mid = encodeMidi(lastNotes, 120);
    const blob = new Blob([mid], { type: 'audio/midi' });
    triggerDownload(blob, 'game_output.mid');
}
function downloadTxt() {
    if (!lastNotes) return;
    const lines = lastNotes.map(n => {
        const p = n.voiced ? n.pitch.toFixed(3) : 'rest';
        return `${n.offset.toFixed(3)}\t${n.duration.toFixed(3)}\t${p}`;
    }).join('\n');
    const blob = new Blob([lines], { type: 'text/plain' });
    triggerDownload(blob, 'game_output.txt');
}
function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

// Minimal SMF type-0 writer in JS (same layout as src/cli/midi_writer.cpp).
function encodeMidi(notes, tempoBpm) {
    const TPQN = 480;
    const tps  = TPQN * tempoBpm / 60;
    const events = [];
    const mpqn = Math.round(60000000 / tempoBpm);
    events.push({ tick: 0, bytes: [0xff, 0x51, 0x03, (mpqn >> 16) & 0xff, (mpqn >> 8) & 0xff, mpqn & 0xff] });
    for (const n of notes) {
        if (!n.voiced) continue;
        const p = Math.max(0, Math.min(127, Math.round(n.pitch)));
        const onT  = Math.round(n.offset * tps);
        const offT = Math.max(onT + 1, Math.round((n.offset + n.duration) * tps));
        events.push({ tick: onT,  bytes: [0x90, p, 96] });
        events.push({ tick: offT, bytes: [0x80, p, 64] });
    }
    const last = events.reduce((m, e) => Math.max(m, e.tick), 0);
    events.push({ tick: last, bytes: [0xff, 0x2f, 0x00] });
    events.sort((a, b) => a.tick - b.tick);
    const varlen = (n) => {
        const out = [n & 0x7f];
        n >>= 7;
        while (n > 0) { out.unshift(0x80 | (n & 0x7f)); n >>= 7; }
        return out;
    };
    const track = [];
    let prev = 0;
    for (const e of events) {
        varlen(e.tick - prev).forEach(b => track.push(b));
        e.bytes.forEach(b => track.push(b));
        prev = e.tick;
    }
    const u32 = (n) => [(n>>>24)&0xff,(n>>>16)&0xff,(n>>>8)&0xff,n&0xff];
    const u16 = (n) => [(n>>>8)&0xff, n&0xff];
    const out = [
        ...new TextEncoder().encode('MThd'),
        ...u32(6),
        ...u16(0),           // format 0
        ...u16(1),           // 1 track
        ...u16(TPQN),
        ...new TextEncoder().encode('MTrk'),
        ...u32(track.length),
        ...track,
    ];
    return new Uint8Array(out);
}

// ---- wire events -----------------------------------------------------------

$('loadBtn').addEventListener('click', loadModel);
$('runBtn').addEventListener('click',  transcribe);
$('dlBtn').addEventListener('click',   downloadMidi);
$('dlTxtBtn').addEventListener('click', downloadTxt);

const dz = $('dropZone');
dz.addEventListener('click', () => $('file').click());
$('file').addEventListener('change', e => e.target.files[0] && decodeFile(e.target.files[0]));
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', async e => {
    e.preventDefault(); dz.classList.remove('drag');
    if (e.dataTransfer.files[0]) decodeFile(e.dataTransfer.files[0]);
});
