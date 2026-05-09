/**
 * Cross-Origin Isolation service worker.
 *
 * GitHub Pages doesn't let us set COOP/COEP response headers directly.
 * SharedArrayBuffer (required by threaded WASM) needs the page to be
 * cross-origin isolated, which needs those headers.  This SW intercepts
 * every same-origin fetch and injects them.
 *
 * Based on https://github.com/gzuidhof/coi-serviceworker (MIT).
 *
 * Usage — add this single line early in <head> of any page that needs
 * SharedArrayBuffer:
 *
 *     <script src="coi-serviceworker.js"></script>
 *
 * On the very first visit the page reloads once; subsequent visits are
 * already isolated.
 */

if (typeof window === 'undefined') {
    // --- service-worker context ---------------------------------------------
    self.addEventListener('install',  () => self.skipWaiting());
    self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

    self.addEventListener('fetch', (event) => {
        const req = event.request;
        if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

        event.respondWith(
            fetch(req)
                .then((r) => {
                    if (r.status === 0) return r;
                    const h = new Headers(r.headers);
                    h.set('Cross-Origin-Opener-Policy',   'same-origin');
                    h.set('Cross-Origin-Embedder-Policy', 'require-corp');
                    h.set('Cross-Origin-Resource-Policy', 'cross-origin');
                    return new Response(r.body, { status: r.status, statusText: r.statusText, headers: h });
                })
                .catch((e) => console.error('coi-sw: fetch failed', e))
        );
    });
} else {
    // --- page context (inline <script>) -------------------------------------
    (() => {
        if (window.crossOriginIsolated) return;           // already good
        if (!('serviceWorker' in navigator)) {
            console.warn('[coi-sw] no service worker support — threads disabled');
            return;
        }
        const url = document.currentScript.src;
        navigator.serviceWorker.register(url, { scope: './' }).then((reg) => {
            reg.addEventListener('updatefound', () => window.location.reload());
            // First-install reload so subsequent requests hit the SW.
            if (reg.active && !navigator.serviceWorker.controller) {
                window.location.reload();
            }
        }).catch((e) => console.error('[coi-sw] register failed:', e));
    })();
}
