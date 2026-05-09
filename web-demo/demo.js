// GAME ggml browser demo — main-thread architecture.
//
// Why main thread?
//   * Emscripten pthread workers don't spawn reliably when the outer JS is
//     already inside a Worker (silent `undefined:undefined: undefined` crash
//     on Chrome), so the threaded CPU build needs to live on the main thread.
//   * JSPI + embind integration (needed for the WebGPU backend to suspend on
//     `requestAdapter` / `onSubmittedWorkDone`) also works most reliably on
//     the main thread.
//
// UI blocking is mitigated by the slicer — long audio is chunked into ~20 s
// pieces; between chunks we `await yieldToBrowser()` so the browser can paint
// partial results and stay responsive.

// ---- tiny helpers ----------------------------------------------------------

const $ = (id) => document.getElementById(id);
const log = (...args) => {
    const line = args.map((x) => typeof x === 'string' ? x : JSON.stringify(x)).join(' ');
    $('log').textContent += '\n' + line;
    $('log').scrollTop = $('log').scrollHeight;
    console.log(...args);
};
function fmtBytes(n) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let u = 0;
    while (n >= 1024 && u < units.length - 1) { n /= 1024; ++u; }
    return n.toFixed(n < 10 ? 2 : n < 100 ? 1 : 0) + ' ' + units[u];
}
function setStatus(dot, statusEl, text, kind = '') {
    dot.className = 'dot ' + kind;
    statusEl.textContent = text;
}
const yieldToBrowser = () => new Promise((r) => setTimeout(r, 0));

// ---- environment banner ----------------------------------------------------

(function detectEnv() {
    const banner = $('env');
    const cores    = navigator.hardwareConcurrency || 4;
    const isolated = !!self.crossOriginIsolated;
    const hasGpu   = !!navigator.gpu;
    const hasJSPI  = typeof WebAssembly !== 'undefined'
                  && typeof WebAssembly.Suspending === 'function';
    const gpuUsable = hasGpu && hasJSPI;
    const bits = [
        `<span class="pill"><strong>${cores}</strong> cores</span>`,
        isolated
            ? '<span class="pill ok">threads ✓</span>'
            : '<span class="pill bad">threads ✗ (reload once after SW installs)</span>',
        gpuUsable
            ? '<span class="pill ok">WebGPU ✓</span>'
            : hasGpu
                ? '<span class="pill">WebGPU (no JSPI) — GPU disabled</span>'
                : '<span class="pill">WebGPU ✗</span>',
    ];
    banner.innerHTML = bits.join('');
    const nt = $('nthreads');
    if (nt) nt.value = isolated ? Math.max(1, Math.min(8, cores)) : 1;
})();

// ---- state ----------------------------------------------------------------

let Module = null;
let model  = null;
let audioBuffer = null;
let audioName   = null;
let lastNotes   = null;

// Which WASM variant is currently loaded ('gpu' | 'cpu' | null).  Keeping
// this lets us skip redundant re-inits when the user clicks Load twice in a
// row with the same backend choice.
let currentVariant = null;

function updateRunBtn() { $('runBtn').disabled = !(model && audioBuffer); }

// ---- WASM module (lazy) ---------------------------------------------------

// JSPI (`WebAssembly.Suspending` + `WebAssembly.promising`) is required by
// the WebGPU variant because ggml-webgpu / emdawnwebgpu expresses async Dawn
// calls as sync WASM through JSPI suspend/resume.  As of 2025:
//   * Chrome 123+        ✓
//   * Safari 18          ✗
//   * Firefox (stable)   ✗
// If we try to `await import('./game_ggml-gpu.js')` in a non-JSPI browser,
// the glue code throws at module evaluation time (`new WebAssembly.Suspending
// is not a constructor`).  Detect up front and short-circuit.
function canUseJSPI() {
    return typeof WebAssembly !== 'undefined'
        && typeof WebAssembly.Suspending === 'function'
        && typeof WebAssembly.promising  === 'function';
}

async function requestGpuDevice() {
    if (!navigator.gpu) return null;
    if (!canUseJSPI()) {
        log('WebGPU present but JSPI not supported (needs Chrome 123+) — skipping GPU');
        return null;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance',
        });
        if (!adapter) return null;
        return await adapter.requestDevice();
    } catch (e) {
        log(`WebGPU device request failed: ${e.message}`);
        return null;
    }
}

async function loadVariant(variant, preDevice = null) {
    const jsUrl   = variant === 'gpu' ? './game_ggml-gpu.js'   : './game_ggml-cpu.js';
    const wasmUrl = variant === 'gpu' ? './game_ggml-gpu.wasm' : './game_ggml-cpu.wasm';
    log(`loading ${jsUrl} · backend=${variant === 'gpu' ? 'WebGPU' : 'CPU (multi-threaded)'}`);
    const { default: createGameGGML } = await import(jsUrl);
    const args = {
        // Both WASM variants were compiled expecting `game_ggml.{js,wasm}`.
        // Override both filenames so the renamed files (and the inlined
        // pthread worker URL) resolve correctly.
        locateFile: (p) => (p === 'game_ggml.wasm' ? wasmUrl : p),
        mainScriptUrlOrBlob: new URL(jsUrl, import.meta.url).href,
    };
    if (preDevice) args.preinitializedWebGPUDevice = preDevice;
    const M = await createGameGGML(args);
    log(`module ready · game_ggml ${M.version()} · ggml ${M.ggmlVersion()}`);
    $('buildStats').textContent =
        ` · game_ggml ${M.version()} · backend ${variant === 'gpu' ? 'WebGPU' : 'CPU (multi-threaded)'}`;
    if (variant === 'gpu') {
        $('nthreads').disabled = true;
        $('nthreads').title = 'not applicable to WebGPU backend';
    } else {
        $('nthreads').disabled = false;
        $('nthreads').title = '';
    }
    return M;
}

async function ensureModule() {
    const choice = $('backendChoice').value;      // 'auto' | 'gpu' | 'cpu'

    // Fast path: keep the existing module if it already matches the request.
    if (Module) {
        if (choice === 'cpu'  && currentVariant === 'cpu') return;
        if (choice === 'gpu'  && currentVariant === 'gpu') return;
        if (choice === 'auto' && currentVariant) return;    // either variant is fine
    }

    // Dispose any previous model (Emscripten modules aren't reusable across
    // variants, so a backend switch means re-init from scratch).
    if (model) {
        try { if (model.delete) await model.delete(); } catch {}
        model = null;
        updateRunBtn();
    }
    Module = null;
    currentVariant = null;

    if (choice === 'cpu') {
        Module = await loadVariant('cpu');
        currentVariant = 'cpu';
        return;
    }

    // GPU or Auto — try WebGPU first.
    const preDevice = await requestGpuDevice();
    if (preDevice) {
        try {
            Module = await loadVariant('gpu', preDevice);
            currentVariant = 'gpu';
            return;
        } catch (e) {
            log(`GPU init failed: ${e.message}`);
            if (choice === 'gpu') throw e;             // explicit GPU — propagate
            log('falling back to CPU (multi-threaded) …');
        }
    } else if (choice === 'gpu') {
        throw new Error(
            canUseJSPI()
                ? 'WebGPU adapter request failed'
                : 'WebGPU backend requires JSPI (Chrome 123+); current browser does not support it');
    } else {
        log('no WebGPU adapter — using CPU (multi-threaded)');
    }
    Module = await loadVariant('cpu');
    currentVariant = 'cpu';
}

// ---- GGUF fetch (cached) --------------------------------------------------

async function fetchGguf(dtype) {
    const url = `./assets/game_small_${dtype}.gguf`;
    const cache = await caches.open('game-ggml-v1');
    let resp = await cache.match(url);
    const fresh = !resp;
    if (!resp) {
        log(`fetching ${url} …`);
        const r = await fetch(url);
        if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
        resp = r.clone();
        await cache.put(url, r);
    }
    const bytes = await resp.arrayBuffer();
    log(`  → ${fmtBytes(bytes.byteLength)} ${fresh ? '(downloaded)' : '(cache hit)'}`);
    return bytes;
}

// ---- Load model -----------------------------------------------------------

async function loadModel() {
    const dtype = $('modelChoice').value;
    $('loadBtn').disabled = true;
    setStatus($('modelDot'), $('modelStatus'), 'initializing…', 'running');
    try {
        await ensureModule();
        const gguf = await fetchGguf(dtype);
        const ptr = Module._malloc(gguf.byteLength);
        Module.HEAPU8.set(new Uint8Array(gguf), ptr);
        const t0 = performance.now();
        if (model) {
            try { if (model.delete) await model.delete(); } catch {}
            model = null;
        }
        if (currentVariant === 'gpu') {
            // Plain-C path: JSPI-safe, required for WebGPU backend.  The
            // create call *suspends* while Dawn requests adapter + device.
            const handle = await Module._gg_create_model(ptr, gguf.byteLength);
            if (!handle) throw new Error('gg_create_model returned 0');
            model = makeCModel(handle);
        } else {
            // embind path is fine for CPU; no JSPI suspension happens here.
            model = Module.createModelFromPtr(ptr, gguf.byteLength);
        }
        const dt = performance.now() - t0;
        Module._free(ptr);
        log(`model ready in ${dt.toFixed(0)} ms · arch=${model.arch()} · sr=${model.sampleRate()}`);
        setStatus($('modelDot'), $('modelStatus'),
            `loaded · ${dtype} · ${currentVariant === 'gpu' ? 'WebGPU' : 'CPU'}`, 'ok');
    } catch (e) {
        log('load failed: ' + e.message);
        setStatus($('modelDot'), $('modelStatus'), 'load failed', 'bad');
    } finally {
        $('loadBtn').disabled = false;
        updateRunBtn();
    }
}

// Plain-C model shim, mirroring the embind Model surface used by
// `transcribe()`.  `infer` is async because it suspends via JSPI.
function makeCModel(handle) {
    return {
        __handle: handle,
        arch() { return 'game-me'; },
        sampleRate() { return Module._gg_sample_rate(handle); },
        setNumThreads(n) { Module._gg_set_threads(handle, n); },

        sliceWaveform(wavPtr, nSamples) {
            const maxChunks = 128;
            const outPtr = Module._malloc(maxChunks * 3 * 4);
            const n = Module._gg_slice(handle, wavPtr, nSamples, outPtr, maxChunks);
            const out = [];
            const u32 = Module.HEAPU32;
            for (let i = 0; i < n; i++) {
                const base = (outPtr >> 2) + i * 3;
                out.push({
                    offsetSamples: u32[base + 0],
                    lengthSamples: u32[base + 1],
                    offsetSeconds: u32[base + 2] / 1000,
                });
            }
            Module._free(outPtr);
            return out;
        },

        async infer(wavPtr, nSamples, lang, seedLo, seedHi, nsteps) {
            const countPtr = Module._malloc(4);
            const notesPtr = await Module._gg_infer(
                handle, wavPtr, nSamples, lang, seedLo, seedHi, nsteps, countPtr);
            const count = Module.HEAPU32[countPtr >> 2];
            Module._free(countPtr);
            const notes = new Array(count);
            if (notesPtr && count > 0) {
                const f = Module.HEAPF32;
                const base = notesPtr >> 2;
                for (let i = 0; i < count; i++) {
                    notes[i] = {
                        offset:   f[base + i * 4 + 0],
                        duration: f[base + i * 4 + 1],
                        pitch:    f[base + i * 4 + 2],
                        voiced:   f[base + i * 4 + 3] > 0.5,
                    };
                }
                Module._gg_free(notesPtr);
            }
            notes.length = count;
            return notes;
        },

        async delete() {
            await Module._gg_destroy_model(handle);
            this.__handle = 0;
        },
    };
}

// ---- Audio decode ---------------------------------------------------------

async function decodeFile(file) {
    setStatus($('audioDot'), $('audioStatus'), `decoding ${file.name}…`, 'running');
    log(`decoding ${file.name} (${fmtBytes(file.size)}) …`);
    const t0 = performance.now();
    try {
        const buf = await file.arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
        const ab = await ac.decodeAudioData(buf);
        ac.close();
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
            log(`  resampling ${ab.sampleRate} → 44100 Hz …`);
            mono = linearResample(mono, ab.sampleRate, 44100);
        }
        const sec = mono.length / 44100;
        log(`  → ${mono.length} samples (${sec.toFixed(1)} s) in ${(performance.now() - t0).toFixed(0)} ms`);
        audioBuffer = mono;
        audioName = file.name;
        setStatus($('audioDot'), $('audioStatus'),
            `${file.name} · ${sec.toFixed(1)} s`, 'ok');
    } catch (e) {
        audioBuffer = null;
        log('decode failed: ' + e.message);
        setStatus($('audioDot'), $('audioStatus'), 'decode failed', 'bad');
    }
    updateRunBtn();
}

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

// ---- Inference ------------------------------------------------------------

async function transcribe() {
    if (!model || !audioBuffer) return;
    $('runBtn').disabled = true;
    setStatus($('runDot'), $('runStatus'), 'running…', 'running');
    try {
        const lang     = parseInt($('language').value, 10);
        const nsteps   = parseInt($('nsteps').value, 10);
        const nthreads = parseInt($('nthreads').value, 10);
        const seed     = BigInt($('seed').value);
        const seedLo   = Number(seed & 0xFFFFFFFFn);
        const seedHi   = Number((seed >> 32n) & 0xFFFFFFFFn);

        if (typeof model.setNumThreads === 'function') model.setNumThreads(nthreads);

        const dur = audioBuffer.length / 44100;
        log(`inference: ${audioBuffer.length} samples (${dur.toFixed(1)} s) · lang=${lang} · steps=${nsteps} · threads=${nthreads}`);

        const wavPtr = Module._malloc(audioBuffer.byteLength);
        Module.HEAPF32.set(audioBuffer, wavPtr / 4);

        const t0 = performance.now();
        let allNotes = [];

        if (dur > 60 && typeof model.sliceWaveform === 'function') {
            const chunks = model.sliceWaveform(wavPtr, audioBuffer.length);
            log(`  sliced into ${chunks.length} chunks`);
            for (let i = 0; i < chunks.length; i++) {
                const c = chunks[i];
                const chunkPtr = wavPtr + c.offsetSamples * 4;
                const t1 = performance.now();
                const notes = await model.infer(chunkPtr, c.lengthSamples,
                    lang, seedLo, seedHi, nsteps);
                const chunkMs = performance.now() - t1;
                for (let k = 0; k < notes.length; k++) {
                    allNotes.push({
                        offset:   notes[k].offset + c.offsetSeconds,
                        duration: notes[k].duration,
                        pitch:    notes[k].pitch,
                        voiced:   !!notes[k].voiced,
                    });
                }
                log(`  chunk ${i + 1}/${chunks.length} · ${(c.lengthSamples / 44100).toFixed(1)}s · ${notes.length} notes · ${chunkMs.toFixed(0)} ms`);
                setStatus($('runDot'), $('runStatus'),
                    `chunk ${i + 1} / ${chunks.length} …`, 'running');
                draw(allNotes);
                await yieldToBrowser();
            }
        } else {
            const notes = await model.infer(wavPtr, audioBuffer.length,
                lang, seedLo, seedHi, nsteps);
            for (let i = 0; i < notes.length; i++) {
                allNotes.push({
                    offset:   notes[i].offset,
                    duration: notes[i].duration,
                    pitch:    notes[i].pitch,
                    voiced:   !!notes[i].voiced,
                });
            }
        }

        Module._free(wavPtr);
        const totalMs = performance.now() - t0;
        const voiced = allNotes.filter((n) => n.voiced).length;
        log(`→ ${allNotes.length} notes (${voiced} voiced) in ${(totalMs / 1000).toFixed(2)} s · RTF ${(dur / (totalMs / 1000)).toFixed(1)}×`);
        setStatus($('runDot'), $('runStatus'),
            `${voiced} voiced · ${(totalMs / 1000).toFixed(2)} s · RTF ${(dur / (totalMs / 1000)).toFixed(1)}×`,
            'ok');
        lastNotes = allNotes;
        draw(allNotes);
        $('dlBtn').disabled = false;
        $('dlTxtBtn').disabled = false;
        $('pianoBtn').disabled = false;
    } catch (e) {
        log('inference failed: ' + e.message);
        setStatus($('runDot'), $('runStatus'), 'failed', 'bad');
    } finally {
        $('runBtn').disabled = !(model && audioBuffer);
    }
}

// ---- Piano roll -----------------------------------------------------------

function draw(notes) {
    const cv = $('roll');
    const rect = cv.getBoundingClientRect();
    cv.width = rect.width | 0;
    cv.height = 240;
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);

    const voiced = notes.filter((n) => n.voiced);
    if (voiced.length === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '14px ui-monospace';
        ctx.fillText('no voiced notes', 14, 28);
        return;
    }
    const tEnd = Math.max(...notes.map((n) => n.offset + n.duration));
    const pitches = voiced.map((n) => n.pitch);
    const pLo = Math.floor(Math.min(...pitches) - 1);
    const pHi = Math.ceil (Math.max(...pitches) + 1);
    const W = cv.width, H = cv.height, pad = 10;
    const xs = (t) => pad + t * (W - 2 * pad) / tEnd;
    const ys = (p) => H - pad - (p - pLo) * (H - 2 * pad) / (pHi - pLo);

    ctx.strokeStyle = '#eee';
    ctx.beginPath();
    for (let p = pLo; p <= pHi; p++) {
        const y = ys(p);
        ctx.moveTo(pad, y); ctx.lineTo(W - pad, y);
    }
    ctx.stroke();

    for (const n of voiced) {
        const x = xs(n.offset);
        const w = Math.max(1, xs(n.offset + n.duration) - x);
        const y = ys(n.pitch);
        ctx.fillStyle = `hsl(${(n.pitch * 13) % 360}, 60%, 45%)`;
        ctx.fillRect(x, y - 3, w, 6);
    }
}

// ---- Downloads ------------------------------------------------------------

function downloadMidi() {
    if (!lastNotes) return;
    const mid = encodeMidi(lastNotes, 120);
    triggerDownload(new Blob([mid], { type: 'audio/midi' }),
        (audioName || 'game_output').replace(/\.[^/.]+$/, '') + '.mid');
}
function downloadTxt() {
    if (!lastNotes) return;
    const lines = lastNotes.map((n) => {
        const p = n.voiced ? n.pitch.toFixed(3) : 'rest';
        return `${n.offset.toFixed(3)}\t${n.duration.toFixed(3)}\t${p}`;
    }).join('\n');
    triggerDownload(new Blob([lines], { type: 'text/plain' }),
        (audioName || 'game_output').replace(/\.[^/.]+$/, '') + '.txt');
}
function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

function encodeMidi(notes, tempoBpm) {
    const TPQN = 480;
    const tps = TPQN * tempoBpm / 60;
    const events = [];
    const mpqn = Math.round(60000000 / tempoBpm);
    events.push({ tick: 0, bytes: [0xff, 0x51, 0x03, (mpqn >> 16) & 0xff, (mpqn >> 8) & 0xff, mpqn & 0xff] });
    for (const n of notes) {
        if (!n.voiced) continue;
        const p = Math.max(0, Math.min(127, Math.round(n.pitch)));
        const onT = Math.round(n.offset * tps);
        const offT = Math.max(onT + 1, Math.round((n.offset + n.duration) * tps));
        events.push({ tick: onT,  bytes: [0x90, p, 96] });
        events.push({ tick: offT, bytes: [0x80, p, 64] });
    }
    const last = events.reduce((m, e) => Math.max(m, e.tick), 0);
    events.push({ tick: last, bytes: [0xff, 0x2f, 0x00] });
    events.sort((a, b) => a.tick - b.tick);
    const varlen = (n) => {
        const out = [n & 0x7f]; n >>= 7;
        while (n > 0) { out.unshift(0x80 | (n & 0x7f)); n >>= 7; }
        return out;
    };
    const track = [];
    let prev = 0;
    for (const e of events) {
        varlen(e.tick - prev).forEach((b) => track.push(b));
        e.bytes.forEach((b) => track.push(b));
        prev = e.tick;
    }
    const u32 = (n) => [(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff];
    const u16 = (n) => [(n >>> 8) & 0xff, n & 0xff];
    return new Uint8Array([
        ...new TextEncoder().encode('MThd'), ...u32(6),
        ...u16(0), ...u16(1), ...u16(TPQN),
        ...new TextEncoder().encode('MTrk'), ...u32(track.length), ...track,
    ]);
}

// ---- Event wiring ---------------------------------------------------------

$('loadBtn').addEventListener('click', loadModel);
$('runBtn').addEventListener('click',  transcribe);
$('dlBtn').addEventListener('click',   downloadMidi);
$('dlTxtBtn').addEventListener('click', downloadTxt);

// Changing the backend invalidates the currently loaded model; user needs to
// re-click Load to re-init with the new backend.
$('backendChoice').addEventListener('change', async () => {
    if (!Module) return;
    if (model) {
        try { if (model.delete) await model.delete(); } catch {}
        model = null;
    }
    Module = null;
    currentVariant = null;
    updateRunBtn();
    setStatus($('modelDot'), $('modelStatus'), 'click Load to apply', '');
    log(`backend preference changed to "${$('backendChoice').value}" — click Load to re-init`);
});

const dz = $('dropZone');
dz.addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => e.target.files[0] && decodeFile(e.target.files[0]));
dz.addEventListener('dragover',  (e) => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    if (e.dataTransfer.files[0]) decodeFile(e.dataTransfer.files[0]);
});

new ResizeObserver(() => lastNotes && draw(lastNotes)).observe($('roll'));

// ---- Piano roll + playback ------------------------------------------------
//
// The piano-roll modal opens after transcription; it renders the notes on a
// keyboard-lane canvas with a moving playhead, and can play the MIDI through
// Web Audio while optionally playing the original WAV alongside so the user
// can A/B the transcription against the source.

// Platform-aware shortcut hints
(function initShortcutHint() {
    const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent);
    const mod = isMac ? '⌘' : 'Ctrl';
    const hint = `· space = play · ${mod}+wheel = zoom X · ${mod}+Shift+wheel = zoom Y · Shift+wheel = pan · 0 = fit`;
    const el = $('shortcutHint');
    if (el) el.textContent = hint;
})();

// Slider progress fill (Webkit needs CSS var trick)
function updateRangeProgress(input) {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty('--range-pct', pct + '%');
}
document.querySelectorAll('.piano-controls input[type=range]').forEach((el) => {
    updateRangeProgress(el);
    el.addEventListener('input', () => updateRangeProgress(el));
});

const SVG_PLAY = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><polygon points="2,0 14,7 2,14"/></svg>';
const SVG_PAUSE = '<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor"><rect x="1" y="0" width="4" height="14" rx="1"/><rect x="9" y="0" width="4" height="14" rx="1"/></svg>';

const PianoRoll = (function () {
    const modal  = $('pianoModal');
    const canvas = $('pianoCanvas');
    const scroll = $('pianoScroll');
    const spacer = $('pianoSpacer');
    const ctx    = canvas.getContext('2d', { alpha: false });

    let notes    = [];
    let duration = 0;
    let pxPerSec = 60;          // horizontal scale
    let rowH     = 14;          // px per semitone (vertical scale)
    const keyW   = 64;          // piano keys width
    const rulerH = 22;          // top time ruler height
    // Full 88-key piano range (A0=21 … C8=108)
    const PITCH_LO = 21, PITCH_HI = 108;

    let pendingFrame = false;
    function schedule() {
        if (pendingFrame) return;
        pendingFrame = true;
        requestAnimationFrame(() => { pendingFrame = false; draw(); });
    }

    // Playback
    let audioCtx = null;
    let wavBufferNode = null;
    let masterMidi = null;
    let masterWav  = null;
    let isPlaying  = false;
    let startedAt  = 0;
    let seekedTo   = 0;
    let scheduledOsc = [];
    let rafId = null;

    function ensureCtx() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 44100,
            });
            masterMidi = audioCtx.createGain();
            masterMidi.connect(audioCtx.destination);
            masterWav  = audioCtx.createGain();
            masterWav.connect(audioCtx.destination);
        }
        masterMidi.gain.value = parseInt($('midiVol').value, 10) / 100 * 0.5;
        masterWav.gain.value  = parseInt($('wavVol' ).value, 10) / 100;
        return audioCtx;
    }

    function open(noteList, audioDur) {
        notes = noteList.filter((n) => n.voiced)
                        .slice().sort((a, b) => a.offset - b.offset);
        duration = Math.max(audioDur || 0,
                            notes.reduce((m, n) => Math.max(m, n.offset + n.duration), 0));
        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
        zoomFit();
        requestAnimationFrame(() => {
            const meanPitch = notes.length
                ? notes.reduce((s, n) => s + n.pitch, 0) / notes.length
                : 60;
            const yMid = rulerH + (PITCH_HI - meanPitch) * rowH;
            scroll.scrollTop = Math.max(0, yMid - scroll.clientHeight / 2);
        });
        updateTime(0);
    }

    function close() {
        stop();
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }

    function setPxPerSec(v) {
        pxPerSec = Math.max(4, Math.min(600, v));
        resize();
        schedule();
    }
    function zoomIn()  { setPxPerSec(pxPerSec * 1.5); }
    function zoomOut() { setPxPerSec(pxPerSec / 1.5); }
    function zoomFit() {
        // Reset scroll so the spacer shrinks cleanly without leftover extent.
        scroll.scrollLeft = 0;
        const w = scroll.clientWidth - keyW - 2;
        if (duration > 0) setPxPerSec(w / duration);
    }

    function resize() {
        spacer.style.width  = (keyW + duration * pxPerSec) + 'px';
        spacer.style.height = (rulerH + (PITCH_HI - PITCH_LO + 1) * rowH) + 'px';
        const viewW = Math.max(1, scroll.clientWidth);
        const viewH = Math.max(1, scroll.clientHeight);
        if (canvas.width !== viewW || canvas.height !== viewH) {
            canvas.width  = viewW;
            canvas.height = viewH;
        }
        canvas.style.width  = viewW + 'px';
        canvas.style.height = viewH + 'px';
        // Pin the canvas to the viewport top-left, clamped to prevent
        // elastic overscroll from displacing it.
        const maxX = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
        const maxY = Math.max(0, scroll.scrollHeight - scroll.clientHeight);
        canvas.style.marginLeft = Math.max(0, Math.min(scroll.scrollLeft, maxX)) + 'px';
        canvas.style.marginTop  = Math.max(0, Math.min(scroll.scrollTop, maxY)) + 'px';
    }

    const isBlack = (p) => [1, 3, 6, 8, 10].includes(((p % 12) + 12) % 12);

    // Real piano black keys are not centered vertically between their
    // neighboring white keys.  This returns a Y-offset (in fraction of rowH)
    // that shifts the black key's lane position to match real piano geometry.
    // Positive = shift down (toward lower pitch visually, since higher pitch
    // is at the top).
    //
    // In a real piano octave (C-B), the black keys sit:
    //   C# — slightly closer to C (shift up / negative)
    //   D# — slightly closer to E (shift down / positive)
    //   F# — slightly closer to F (shift up / negative)
    //   G# — roughly centered
    //   A# — slightly closer to B (shift down / positive)
    function blackKeyYOffset(pitch) {
        const pc = ((pitch % 12) + 12) % 12;
        const offsets = { 1: -0.12, 3: 0.12, 6: -0.10, 8: 0.0, 10: 0.10 };
        return (offsets[pc] || 0) * rowH;
    }

    // Compute the Y position for a pitch, applying black key Y-offset.
    function pitchToYAdjusted(pitch, sy) {
        const baseY = (rulerH + (PITCH_HI - pitch) * rowH) - sy;
        if (isBlack(pitch)) return baseY + blackKeyYOffset(pitch);
        return baseY;
    }

    // Black key X geometry: black keys are narrower than white keys and
    // aligned to the right edge (the side facing the note grid), matching
    // how real piano black keys protrude from the back of the keyboard.
    function blackKeyGeometry(pitch) {
        const bw = keyW * 0.62;
        const x = keyW - bw;
        return { x, w: bw };
    }

    function draw() {
        const viewW = canvas.width, viewH = canvas.height;
        // Clamp scroll values to prevent elastic overscroll from distorting coords.
        const sx = Math.max(0, scroll.scrollLeft);
        const sy = Math.max(0, scroll.scrollTop);

        ctx.fillStyle = '#1a1a1a';
        ctx.fillRect(0, 0, viewW, viewH);

        // Visible time + pitch ranges.
        const tLo = Math.max(0, (sx - keyW) / pxPerSec);
        const tHi = Math.max(0, (sx - keyW + viewW) / pxPerSec);
        const pTop = Math.max(PITCH_LO, Math.floor(PITCH_HI - (sy - rulerH + viewH) / rowH) - 1);
        const pBot = Math.min(PITCH_HI, Math.ceil (PITCH_HI - (sy - rulerH) / rowH) + 1);

        const timeToX  = (t) => (keyW + t * pxPerSec) - sx;
        const pitchToY = (p) => pitchToYAdjusted(p, sy);

        // Lane backgrounds — black key lanes get Y-offset applied.
        for (let p = pTop; p <= pBot; p++) {
            const y = pitchToY(p);
            ctx.fillStyle = isBlack(p) ? '#222' : '#2a2a2a';
            ctx.fillRect(keyW, y, viewW - keyW, rowH);
        }

        // Octave separators (C notes).
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let p = pTop; p <= pBot; p++) {
            if (((p % 12) + 12) % 12 === 0) {
                const y = pitchToY(p) + rowH - 0.5;
                ctx.moveTo(keyW, y); ctx.lineTo(viewW, y);
            }
        }
        ctx.stroke();

        // Notes — use the same Y-adjusted positions so they align with lanes.
        for (let i = 0; i < notes.length; i++) {
            const n = notes[i];
            if (n.offset > tHi) break;
            if (n.offset + n.duration < tLo) continue;
            if (n.pitch > pBot || n.pitch < pTop) continue;
            const x = timeToX(n.offset);
            const w = Math.max(1, timeToX(n.offset + n.duration) - x);
            const y = pitchToY(n.pitch);
            const hue = (n.pitch * 13) % 360;
            ctx.fillStyle = `hsl(${hue}, 65%, 55%)`;
            ctx.fillRect(x, y + 1, w, rowH - 2);
            if (w > 3) {
                ctx.strokeStyle = 'rgba(0,0,0,0.35)';
                ctx.strokeRect(x + 0.5, y + 1.5, w - 1, rowH - 3);
            }
        }

        // Playhead on top of notes, under the frozen keys/ruler.
        const t = currentTime();
        if (isPlaying || seekedTo > 0) {
            const x = timeToX(t);
            if (x >= keyW - 1 && x <= viewW + 1) {
                ctx.strokeStyle = '#ff3a3a';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, 0); ctx.lineTo(x, viewH);
                ctx.stroke();
                ctx.lineWidth = 1;
            }
        }

        // Piano keys column — realistic layout with proper black key offsets.
        // Clip to below the ruler so keys don't bleed into the time ruler area.
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, rulerH, keyW, viewH - rulerH);
        ctx.clip();

        ctx.fillStyle = '#fff';
        ctx.fillRect(0, rulerH, keyW, viewH - rulerH);

        // Draw white keys first.
        for (let p = pTop; p <= pBot; p++) {
            if (isBlack(p)) continue;
            const y = pitchToY(p);
            ctx.strokeStyle = '#ccc';
            ctx.beginPath();
            ctx.moveTo(0, y + rowH - 0.5);
            ctx.lineTo(keyW, y + rowH - 0.5);
            ctx.stroke();
            if (((p % 12) + 12) % 12 === 0) {
                ctx.fillStyle = '#666';
                ctx.font = '10px ui-monospace, Menlo, monospace';
                ctx.textAlign = 'right';
                ctx.fillText(`C${Math.floor(p / 12) - 1}`, keyW - 4, y + rowH - 2);
                ctx.textAlign = 'start';
            }
        }

        // Draw black keys on top with realistic geometry (Y-offset applied).
        for (let p = pTop; p <= pBot; p++) {
            if (!isBlack(p)) continue;
            const y = pitchToY(p);
            const geo = blackKeyGeometry(p);
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(geo.x, y, geo.w, rowH);
            ctx.fillStyle = '#3a3a3a';
            ctx.fillRect(geo.x, y, geo.w, 1);
            ctx.fillStyle = '#000';
            ctx.fillRect(geo.x + geo.w - 1, y, 1, rowH);
        }

        // End piano keys clip region.
        ctx.restore();

        // Separator line between keys and roll area.
        ctx.strokeStyle = '#555';
        ctx.beginPath();
        ctx.moveTo(keyW - 0.5, rulerH);
        ctx.lineTo(keyW - 0.5, viewH);
        ctx.stroke();

        // Time ruler — always pinned to the top of the viewport.
        ctx.fillStyle = '#eaeaea';
        ctx.fillRect(keyW, 0, viewW - keyW, rulerH);
        ctx.fillStyle = '#333';
        ctx.font = '10px ui-monospace, Menlo, monospace';
        const tickStep = pxPerSec >= 60 ? 1 : pxPerSec >= 20 ? 5 : 10;
        const tStart = Math.ceil(tLo / tickStep) * tickStep;
        for (let t2 = tStart; t2 <= tHi; t2 += tickStep) {
            const x = timeToX(t2);
            ctx.fillRect(x, rulerH - 6, 1, 6);
            ctx.fillText(`${t2}s`, x + 2, 12);
        }
        // Top-left corner above the keys column.
        ctx.fillStyle = '#eaeaea';
        ctx.fillRect(0, 0, keyW, rulerH);
    }

    function currentTime() {
        if (!isPlaying) return seekedTo;
        return seekedTo + (audioCtx.currentTime - startedAt);
    }

    function updateTime(t) {
        const fmt = (s) => {
            const m = Math.floor(s / 60);
            const r = s - m * 60;
            return `${m}:${r.toFixed(1).padStart(4, '0')}`;
        };
        $('timeDisplay').textContent = `${fmt(Math.max(0, t))} / ${fmt(duration)}`;
    }

    function scheduleMidi(fromT) {
        for (const n of notes) {
            const nStart = n.offset;
            const nEnd   = n.offset + n.duration;
            if (nEnd <= fromT) continue;
            const offsetFromNow = Math.max(0, nStart - fromT);
            const when = audioCtx.currentTime + offsetFromNow;
            const dur  = nEnd - Math.max(nStart, fromT);
            scheduleNote(n, when, dur);
        }
    }

    function scheduleNote(note, startTime, duration) {
        const freq = 440 * Math.pow(2, (note.pitch - 69) / 12);
        const osc = audioCtx.createOscillator();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, startTime);
        const gain = audioCtx.createGain();
        const attack = 0.012, release = Math.min(0.08, duration * 0.3);
        const peak = 0.5;
        gain.gain.setValueAtTime(0, startTime);
        gain.gain.linearRampToValueAtTime(peak, startTime + attack);
        gain.gain.setValueAtTime(peak, Math.max(startTime + attack, startTime + duration - release));
        gain.gain.linearRampToValueAtTime(0, startTime + duration);
        osc.connect(gain); gain.connect(masterMidi);
        osc.start(startTime);
        osc.stop(startTime + duration + 0.05);
        scheduledOsc.push(osc);
    }

    function play() {
        if (notes.length === 0) return;
        ensureCtx();
        if (audioCtx.state === 'suspended') audioCtx.resume();

        const from = seekedTo;
        scheduledOsc = [];
        scheduleMidi(from);

        if ($('playOrig').checked && audioBuffer) {
            const buf = audioCtx.createBuffer(1, audioBuffer.length, 44100);
            buf.getChannelData(0).set(audioBuffer);
            wavBufferNode = audioCtx.createBufferSource();
            wavBufferNode.buffer = buf;
            wavBufferNode.connect(masterWav);
            wavBufferNode.start(audioCtx.currentTime, from);
        }

        startedAt = audioCtx.currentTime;
        isPlaying = true;
        $('playBtn').innerHTML = SVG_PAUSE;
        loop();
    }

    function pause() {
        if (!isPlaying) return;
        const t = currentTime();
        stopScheduled();
        seekedTo = Math.min(t, duration);
        isPlaying = false;
        $('playBtn').innerHTML = SVG_PLAY;
        schedule();
        updateTime(seekedTo);
    }

    function stop() {
        stopScheduled();
        seekedTo = 0;
        isPlaying = false;
        $('playBtn').innerHTML = SVG_PLAY;
        schedule();
        updateTime(0);
    }

    function stopScheduled() {
        for (const o of scheduledOsc) {
            try { o.stop(); } catch {}
        }
        scheduledOsc = [];
        if (wavBufferNode) {
            try { wavBufferNode.stop(); } catch {}
            wavBufferNode = null;
        }
    }

    function loop() {
        if (!isPlaying) return;
        const t = currentTime();
        updateTime(t);

        // Auto-scroll: page-flip when playhead exits the visible region.
        // We set scrollLeft directly and immediately sync the canvas margin
        // in the same frame to avoid a flash.
        if ($('followPlayhead').checked) {
            const xHead = keyW + t * pxPerSec;
            const viewL = scroll.scrollLeft;
            const viewW = scroll.clientWidth;
            const viewR = viewL + viewW;
            let newScrollX = -1;
            if (xHead > viewR - 40) {
                newScrollX = Math.max(0, xHead - keyW - 20);
            } else if (xHead < viewL + keyW) {
                newScrollX = Math.max(0, xHead - keyW - 20);
            }
            if (newScrollX >= 0) {
                scroll.scrollLeft = newScrollX;
                // Immediately sync margin so the canvas doesn't flash at old position.
                const maxX = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
                canvas.style.marginLeft = Math.max(0, Math.min(scroll.scrollLeft, maxX)) + 'px';
            }
        }

        schedule();

        if (t >= duration) { stop(); return; }
        rafId = requestAnimationFrame(loop);
    }

    function seekFromClick(ev) {
        const rect = scroll.getBoundingClientRect();
        const mouseX = ev.clientX - rect.left;
        const mouseY = ev.clientY - rect.top;

        // Click in the piano keys column → play the note, don't seek.
        if (mouseX < keyW) {
            if (mouseY <= rulerH) return; // top-left corner, ignore
            const sy = scroll.scrollTop;
            const contentY = sy + mouseY;
            // Find which pitch row the click lands in (accounting for Y offsets).
            // Use base pitch calculation then check neighbors for adjusted match.
            const basePitch = Math.round(PITCH_HI - (contentY - rulerH) / rowH);
            let bestPitch = basePitch;
            let bestDist = Infinity;
            for (let p = basePitch - 2; p <= basePitch + 2; p++) {
                if (p < PITCH_LO || p > PITCH_HI) continue;
                // pitchToYAdjusted subtracts sy, so pass 0 to get content-space Y
                const y0 = pitchToYAdjusted(p, 0);
                if (contentY >= y0 && contentY < y0 + rowH) {
                    bestPitch = p;
                    break;
                }
                const mid = y0 + rowH / 2;
                const d = Math.abs(contentY - mid);
                if (d < bestDist) { bestDist = d; bestPitch = p; }
            }
            if (bestPitch >= PITCH_LO && bestPitch <= PITCH_HI) {
                playKeySound(bestPitch);
            }
            return;
        }

        // Click in the time ruler → seek.
        const x = mouseX + scroll.scrollLeft;
        const t = (x - keyW) / pxPerSec;
        const wasPlaying = isPlaying;
        stopScheduled();
        seekedTo = Math.max(0, Math.min(duration, t));
        if (wasPlaying) play();
        else { schedule(); updateTime(seekedTo); }
    }

    // Play a short tone when clicking a piano key.
    function playKeySound(pitch) {
        ensureCtx();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        const freq = 440 * Math.pow(2, (pitch - 69) / 12);
        const osc = audioCtx.createOscillator();
        osc.type = 'triangle';
        const gain = audioCtx.createGain();
        const now = audioCtx.currentTime;
        gain.gain.setValueAtTime(0, now);
        gain.gain.linearRampToValueAtTime(0.4, now + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.8);
        osc.frequency.setValueAtTime(freq, now);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.85);
    }

    // Scroll events: update the canvas margin to keep it pinned to viewport,
    // then repaint the visible slice.  Clamp to prevent elastic overscroll
    // (macOS rubber-band) from displacing the canvas.
    scroll.addEventListener('scroll', () => {
        const maxX = Math.max(0, scroll.scrollWidth - scroll.clientWidth);
        const maxY = Math.max(0, scroll.scrollHeight - scroll.clientHeight);
        const clampedX = Math.max(0, Math.min(scroll.scrollLeft, maxX));
        const clampedY = Math.max(0, Math.min(scroll.scrollTop, maxY));
        canvas.style.marginLeft = clampedX + 'px';
        canvas.style.marginTop  = clampedY + 'px';
        schedule();
    }, { passive: true });

    scroll.addEventListener('click', seekFromClick);

    // DAW-style wheel handling:
    //   * Cmd/Ctrl + wheel         → horizontal zoom, anchored at cursor
    //   * Cmd/Ctrl + Shift + wheel → vertical zoom, anchored at cursor
    //   * Shift + wheel            → horizontal pan
    //   * plain wheel              → native vertical scroll
    scroll.addEventListener('wheel', (e) => {
        const zoom = (e.ctrlKey || e.metaKey);
        if (!zoom && !e.shiftKey) return;
        e.preventDefault();
        const rect = scroll.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        if (zoom && e.shiftKey) {
            const contentY = scroll.scrollTop + mouseY;
            const pitchAtMouse = PITCH_HI - (contentY - rulerH) / rowH;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            rowH = Math.max(6, Math.min(40, rowH * factor));
            resize();
            const newContentY = rulerH + (PITCH_HI - pitchAtMouse) * rowH;
            scroll.scrollTop = Math.max(0, newContentY - mouseY);
            schedule();
        } else if (zoom) {
            const contentX = scroll.scrollLeft + mouseX;
            const timeAtMouse = Math.max(0, (contentX - keyW) / pxPerSec);
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            setPxPerSec(pxPerSec * factor);
            const newContentX = keyW + timeAtMouse * pxPerSec;
            scroll.scrollLeft = Math.max(0, newContentX - mouseX);
        } else {
            scroll.scrollLeft += e.deltaY;
        }
    }, { passive: false });
    $('playBtn').addEventListener('click', () => (isPlaying ? pause() : play()));
    $('stopBtn').addEventListener('click', stop);
    $('closePiano').addEventListener('click', close);
    modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
    document.addEventListener('keydown', (e) => {
        if (!modal.classList.contains('open')) return;
        if (e.key === 'Escape') { close(); return; }
        if (e.target.tagName === 'INPUT') return;         // don't hijack range/checkbox input
        if (e.key === ' ') { e.preventDefault(); isPlaying ? pause() : play(); }
        else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomIn(); }
        else if (e.key === '-' || e.key === '_') { e.preventDefault(); zoomOut(); }
        else if (e.key === '0')                   { e.preventDefault(); zoomFit(); }
        else if (e.key === 'Home') { e.preventDefault(); scroll.scrollLeft = 0; stop(); }
    });
    $('zoomIn' ).addEventListener('click', zoomIn);
    $('zoomOut').addEventListener('click', zoomOut);
    $('zoomFit').addEventListener('click', zoomFit);
    $('midiVol').addEventListener('input', () => { if (audioCtx) masterMidi.gain.value = $('midiVol').value / 100 * 0.5; updateRangeProgress($('midiVol')); });
    $('wavVol' ).addEventListener('input', () => { if (audioCtx) masterWav .gain.value = $('wavVol' ).value / 100; updateRangeProgress($('wavVol')); });

    // Allow toggling the original-audio track mid-playback.  Without this
    // the checkbox only took effect on the next play() call.
    $('playOrig').addEventListener('change', () => {
        if (!isPlaying || !audioCtx) return;
        const want = $('playOrig').checked;
        if (want && !wavBufferNode && audioBuffer) {
            const buf = audioCtx.createBuffer(1, audioBuffer.length, 44100);
            buf.getChannelData(0).set(audioBuffer);
            wavBufferNode = audioCtx.createBufferSource();
            wavBufferNode.buffer = buf;
            wavBufferNode.connect(masterWav);
            wavBufferNode.start(audioCtx.currentTime, currentTime());
        } else if (!want && wavBufferNode) {
            try { wavBufferNode.stop(); } catch {}
            wavBufferNode = null;
        }
    });

    new ResizeObserver(() => { resize(); schedule(); }).observe(scroll);

    return { open, close };
})();

$('pianoBtn').addEventListener('click', () => {
    if (lastNotes) PianoRoll.open(lastNotes, audioBuffer ? audioBuffer.length / 44100 : 0);
});

// Wait for the user — module init is deferred so the backend choice honors
// whatever the user has selected before clicking Load.
setStatus($('modelDot'), $('modelStatus'), 'click Load', '');
