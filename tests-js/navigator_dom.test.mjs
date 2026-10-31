/*
 * tests-js/navigator_dom.test.mjs — 1b-4 DOM-load smoke test (the real guard).
 *
 * Builds a JSDOM from the REAL web/index.html so the actual DOM tree exists, then
 * runs the page's inline IIFE in that window with Plotly + THREE mocked and the
 * REAL web/kinematics.js wired in as window.PhononKinematics (displaceMode spied).
 * This catches the class the static HTML test cannot: a structurally valid page
 * whose script dies at load. plotly.min.js / three.min.js are NOT executed inside
 * jsdom (they need canvas/WebGL); only the inline viewer script runs.
 *
 *   Run:  node --test tests-js/navigator_dom.test.mjs
 *   (kinematics.test.mjs stays zero-dependency; only this file needs jsdom.)
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
import { JSDOM } from "jsdom";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const INDEX = join(__dirname, "..", "web", "index.html");
const KINEMATICS = join(__dirname, "..", "web", "kinematics.js");
// 0.6.73 real-payload blend probe: the bundled example lives OUTSIDE the src
// tree (a sibling of the tree), so this test SKIPS from a bare-zip audit and
// runs where the example is present (the developer's tree). Not a tree fixture.
const REAL_NAV = join(__dirname, "..", "..", "navigator_site", "navigator_data.json");

// 0.6.95: the surf3d click listener DEFERS its selection to the next
// macrotask (plotly emits gl3d clicks from inside its render loop; mutating
// synchronously re-enters and throws). Tests firing surf3d clicks must
// flush that deferral before asserting.
const flushClick = () => new Promise(r => setTimeout(r, 3));

// ---- minimal mocks (reused from the 0.6.7 wiring proof) ----
function v3(x, y, z) {
  // storing vector stub with REAL arithmetic so camera fits, the DIM-aware
  // orientation math (dot/cross/normalize), Object3D matrix recording, and the
  // Reset-view clone/copy round-trip all execute under the mock (0.6.22 made
  // the fit lean on dot/cross/length — a no-op stub would hide the geometry).
  return { x: x || 0, y: y || 0, z: z || 0,
           set(a, b, c) { this.x = a; this.y = b; this.z = c; return this; },
           setScalar(s) { this.x = s; this.y = s; this.z = s; return this; },
           clone() { return v3(this.x, this.y, this.z); },
           copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; },
           length() { return Math.sqrt(this.x*this.x + this.y*this.y + this.z*this.z); },
           normalize() { const l = this.length() || 1; this.x /= l; this.y /= l; this.z /= l; return this; },
           dot(v) { return this.x*v.x + this.y*v.y + this.z*v.z; },
           angleTo(v) {   // real math is fine in the TEST file (conformance grep scans only index.html)
             const d = this.dot(v) / ((this.length() || 1) * (v.length() || 1));
             return Math.acos(Math.max(-1, Math.min(1, d)));
           },
           multiplyScalar(s) { this.x *= s; this.y *= s; this.z *= s; return this; },
           addScaledVector(v, s) { this.x += v.x*s; this.y += v.y*s; this.z += v.z*s; return this; },
           crossVectors(a, b) {
             const ax = a.x, ay = a.y, az = a.z, bx = b.x, by = b.y, bz = b.z;
             this.x = ay*bz - az*by; this.y = az*bx - ax*bz; this.z = ax*by - ay*bx;
             return this;
           },
           applyQuaternion() { return this; },
           lookAt() { return this; } };
}
function makeTHREE(window) {
  const T = {
    WebGLRenderer: function () { return { domElement: window.document.createElement("canvas"), setSize() {}, setPixelRatio() {}, setClearColor() {}, render() {}, setViewport() {}, setScissor() {}, setScissorTest() {} }; },
    Scene: function () { return { background: null, add() {}, remove() {} }; },
    PerspectiveCamera: function () {
      return { isPerspectiveCamera: true, position: v3(), up: v3(0, 1, 0),
               quaternion: { copy() { return this; } }, near: 0, far: 0, aspect: 1,
               lookAt() {}, updateProjectionMatrix() {} };
    },
    OrthographicCamera: function (l, r, t2, b, n, f) {
      // record the frustum bounds so the fit/resize assertions can read them
      return { isOrthographicCamera: true, position: v3(), up: v3(0, 1, 0),
               quaternion: { copy() { return this; } },
               left: l, right: r, top: t2, bottom: b, near: n, far: f,
               lookAt() {}, updateProjectionMatrix() {} };
    },
    Group: function () { return { visible: true, children: [], add(o) { this.children.push(o); } }; },
    ArrowHelper: function () { return { visible: true }; },
    Sprite: function (m) { return { material: m, position: v3(), scale: v3() }; },
    SpriteMaterial: function (o) { return o || {}; },
    CanvasTexture: function () { return {}; },
    AmbientLight: function () { return { position: v3() }; },
    DirectionalLight: function () { return { position: v3() }; },
    SphereGeometry: function () { return { dispose() {} }; },
    CylinderGeometry: function () { return { dispose() {} }; },
    ConeGeometry: function () { return { dispose() {} }; },
    MeshLambertMaterial: function (o) { return { _t: "lambert", color: { _v: (o && o.color) || null, set(c) { this._v = c; } } }; },
    MeshBasicMaterial: function (o) { return { color: { _v: (o && o.color) || null, set(c) { this._v = c; } } }; },
    Object3D: function () {
      return { position: v3(), scale: v3(), matrix: {},
               updateMatrix() {
                 this.matrix._p = { x: this.position.x, y: this.position.y, z: this.position.z };
                 this.matrix._s = { x: this.scale.x, y: this.scale.y, z: this.scale.z };
               } };
    },
    Color: function () { return { setRGB() {} }; },
    // THREE.Vector3 IS the storing v3 factory (single source) — the 0.6.22 fit
    // builds vectors with `new THREE.Vector3(...)` and leans on dot/cross/
    // normalize, so they must be the real-arithmetic stub, not a no-op.
    Vector3: function (x, y, z) { return v3(x, y, z); },
    Quaternion: function () { return { setFromUnitVectors() { return this; }, copy() { return this; } }; },
    Float32BufferAttribute: function (arr, n) { return { array: arr, itemSize: n, count: (arr ? arr.length : 0) / n }; },
    BufferGeometry: function () { return { _attr: {}, setAttribute(name, a) { this._attr[name] = a; }, dispose() {} }; },
    LineBasicMaterial: function () { return { color: { set() {} } }; },
    LineSegments: function (g, m) { return { geometry: g, material: m, visible: true, frustumCulled: true, position: v3() }; },  // geometry._attr.position.count = vertex count
    // compose RECORDS the position it is handed (by value — apos is reused), so a
    // setMatrixAt downstream can read the per-instance translation back out.
    Matrix4: function () { return { _p: { x: 0, y: 0, z: 0 }, _s: { x: 1, y: 1, z: 1 },
      compose(p, q, s) { this._p = { x: p.x, y: p.y, z: p.z };
                         if (s) this._s = { x: s.x, y: s.y, z: s.z }; return this; } }; },
    InstancedMesh: function (g, m, n) {
      return { geometry: g, count: n, visible: true, frustumCulled: true, positions: [], scales: [],
               setMatrixAt(k, mat) {
                 this.positions[k] = (mat && mat._p) ? { x: mat._p.x, y: mat._p.y, z: mat._p.z } : null;
                 this.scales[k] = (mat && mat._s) ? { x: mat._s.x, y: mat._s.y, z: mat._s.z } : null;
               },
               setColorAt() {}, instanceMatrix: { needsUpdate: false }, instanceColor: { needsUpdate: false } };
    },
  };
  T.TrackballControls = function (cam) {
    return { object: cam, target: v3(), update() {}, handleResize() {},
             rotateSpeed: 1, staticMoving: false, dynamicDampingFactor: 0.2 };
  };
  return T;
}
function makePlotly(state) {
  // mirror the real Plotly: newPlot attaches .on to the graph DIV the page then
  // calls document.getElementById(id).on(...) on — so we can fire the click.
  return {
    newPlot: (id, data, layout) => {
      state.newPlotCalls.push(id);                          // 0.6.75 spy: every call
      // real Plotly newPlot on an existing div REPLACES the plot; model that so
      // find(id) always returns the current plot (init + each toggle rebuild).
      const existing = state.newPlot.find(p => p.id === id);
      if (existing) { existing.n = (data || []).length; existing.data = data; existing.layout = layout; }
      else state.newPlot.push({ id, n: (data || []).length, data: data, layout: layout });
      const el = state.window.document.getElementById(id);
      if (el) el.on = (evt, cb) => { state.cb[id + ":" + evt] = cb; };
      return el || {};
    },
    purge: (id) => {
      state.purge.push(id);
      const i = state.newPlot.findIndex(p => p.id === id);
      if (i >= 0) state.newPlot.splice(i, 1);
    },
    restyle: (id, update, idx) => {
      state.restyle.push({ id, update, idx });
      // teeth for the 0.6.36 regression class: real Plotly throws when a
      // sub-property of a STRING-valued colorbar title is set (the string->{text}
      // coercion only runs at newPlot, not on restyle). Track the last
      // marker.colorbar / marker.colorbar.title set per trace; if it's a string
      // and a later restyle writes marker.colorbar.title.font.*, throw.
      const idxs = Array.isArray(idx) ? idx : (idx == null ? [0] : [idx]);
      state._cbTitle = state._cbTitle || {};
      for (const key of Object.keys(update)) {
        const val = update[key];
        idxs.forEach((tr, pos) => {
          const v = Array.isArray(val) ? val[pos] : val;
          if (key === "marker.colorbar") {
            state._cbTitle[tr] = v && typeof v.title === "object" ? "object"
                               : (v && typeof v.title === "string" ? "string" : undefined);
          } else if (key === "marker.colorbar.title") {
            state._cbTitle[tr] = typeof v === "string" ? "string" : (v && typeof v === "object" ? "object" : undefined);
          } else if (/^marker\.colorbar\.title\.font\./.test(key)) {
            if (state._cbTitle[tr] === "string")
              throw new Error("Plotly: cannot set " + key + " on a string-valued colorbar title (trace " + tr + ")");
          }
        });
      }
    },
    relayout: () => {}, react: () => {},
    // 0.6.74: mode switch via trace REPLACEMENT. These MUTATE the live gd.data array
    // (the same array newPlot stored), mirroring real Plotly, so tests read the
    // post-toggle traces directly and observe NEW object references. restyle stays
    // non-mutating (effMarker/effSurf replay it over the build snapshot).
    deleteTraces: (id, indices) => {
      state.deleteTraces.push({ id, indices });
      const gd = state.newPlot.find(p => p.id === id); if (!gd) return;
      const idxs = (Array.isArray(indices) ? indices : [indices]).slice().sort((a, b) => b - a);
      for (const i of idxs) gd.data.splice(i, 1);
    },
    addTraces: (id, traces, indices) => {
      state.addTraces.push({ id, traces, indices });
      const gd = state.newPlot.find(p => p.id === id); if (!gd) return;
      const trs = Array.isArray(traces) ? traces : [traces];
      const idxs = indices == null ? trs.map((_, k) => gd.data.length + k)
                                   : (Array.isArray(indices) ? indices : [indices]);
      trs.map((t, k) => ({ t, i: idxs[k] })).sort((a, b) => a.i - b.i)
         .forEach(({ t, i }) => gd.data.splice(i, 0, t));
    },
    Plots: { resize: (id) => state.recResize.push(id) },   // B1: rail-collapse re-fit
  };
}

// small synthetic payload: 2 q-points (Γ, M), 9 branches, 3 atoms
function synthNavData() {
  const Nq = 2, Nb = 9, Na = 3;
  const lattice = [[3.8, 0, 0], [-1.9, 3.29, 0], [0, 0, 20]];
  const frac = [[0, 0, 0.5], [1 / 3, 2 / 3, 0.45], [2 / 3, 1 / 3, 0.55]];
  const er = [], ei = [], fr = [], ch = [], qc = [], qf = [];
  for (let q = 0; q < Nq; q++) {
    const erq = [], eiq = [], frq = [], chq = [];
    for (let b = 0; b < Nb; b++) {
      const erb = [], eib = [];
      for (let a = 0; a < Na; a++) { erb.push([0.1 * (a + 1), 0.2, 0.0]); eib.push([0.0, 0.0, 0.05 * (b + 1)]); }
      erq.push(erb); eiq.push(eib);
      frq.push(b < 3 ? b * 0.5 : (b === 4 ? -0.3 : b));   // one imaginary branch
      chq.push(0.5);
    }
    er.push(erq); ei.push(eiq); fr.push(frq); ch.push(chq);
    qc.push([0.1 * q, 0.05 * q]);
    qf.push(q === 0 ? [0, 0, 0] : [0.5, 0, 0]);            // q0 = Γ, q1 = M
  }
  const kx = [-0.1, 0.1], ky = [-0.1, 0.1];
  const bands = [];
  for (let b = 0; b < Nb; b++) bands.push({ z: [[b, b], [b, b]], character: [[0.5, 0.5], [0.5, 0.5]] });
  const hex = [];
  for (let i = 0; i < 6; i++) { const t = Math.PI / 3 * i; hex.push([0.17 * Math.cos(t), 0.17 * Math.sin(t)]); }
  return {
    meta: {
      material: {
        symbols: ["Sn", "Se", "Se"], masses: [118.71, 78.97, 78.97],
        lattice, fractional_coords: frac,
        species_style: { Sn: { color: [0.4, 0.5, 0.5], radius: 1.39 },
                         Se: { color: [1, 0.63, 0], radius: 1.2 } },
      },
      conventions: {},
    },
    band: {
      distance: [0, 1], segments: [[0, 1]], labels: ["$\\Gamma$", "M"], tick_positions: [0, 1],
      high_symmetry_points: [{ label: "$\\Gamma$", q_cart: [0, 0], distance: 0 },
                             { label: "M", q_cart: [0.1, 0.05], distance: 1 }],
      frequency: fr, character: ch, q_cart: qc, q_frac: qf,
      eigenvectors_real: er, eigenvectors_imag: ei,
    },
    surface: { kx, ky, bands },
    bz_polygon: hex, ibz_polygon: [[0, 0], hex[0], hex[1]], z_range: [-0.3, 9],
  };
}

// 3-segment band with a DISCONTINUOUS middle junction (0.6.16): segments
// q0-1 / q2-3 / q4-5; junction q1|q2 is continuous (same q_frac), junction
// q3|q4 is discontinuous (different q_frac) -> breaks=[[3,4]], merged tick.
// Eigenvectors are made q-DEPENDENT so the click-map assertion can prove the
// engine received the UNSHIFTED qi. No surface block (presence-keyed away).
function synthNavDataBreaks() {
  const Nb = 9, Na = 3;
  const lattice = [[3.8, 0, 0], [-1.9, 3.29, 0], [0, 0, 20]];
  const frac = [[0, 0, 0.5], [1 / 3, 2 / 3, 0.45], [2 / 3, 1 / 3, 0.55]];
  const qf = [[0, 0, 0], [0.5, 0, 0],
              [0.5, 0, 0], [0.5, 0.5, 0],
              [0, 0, 0.25], [0, 0.5, 0.25]];      // qf[3] != qf[4]
  const distArr = [0, 1, 1, 2, 2, 3];
  const er = [], ei = [], fr = [], ch = [], qc = [];
  for (let q = 0; q < 6; q++) {
    const erq = [], eiq = [], frq = [], chq = [];
    for (let b = 0; b < Nb; b++) {
      const erb = [], eib = [];
      for (let a = 0; a < Na; a++) {
        erb.push([0.1 * (a + 1) + 0.01 * q, 0.2, 0.0]);   // q-dependent
        eib.push([0.0, 0.0, 0.05 * (b + 1)]);
      }
      erq.push(erb); eiq.push(eib);
      frq.push(b + 0.1 * q); chq.push(0.5);
    }
    er.push(erq); ei.push(eiq); fr.push(frq); ch.push(chq);
    qc.push([0.1 * q, 0.05 * q]);
  }
  return {
    meta: {
      material: {
        symbols: ["Sn", "Se", "Se"], masses: [118.71, 78.97, 78.97],
        lattice, fractional_coords: frac, dimensionality: 2,
        species_style: { Sn: { color: [0.4, 0.5, 0.5], radius: 1.39 },
                         Se: { color: [1, 0.63, 0], radius: 1.2 } },
      },
      conventions: {},
    },
    band: {
      distance: distArr, segments: [[0, 1], [1, 2], [2, 3]],
      labels: ["A", "B", "C|D", "E"], tick_positions: [0, 1, 2, 3],
      breaks: [[3, 4]],
      high_symmetry_points: [{ label: "A", q_cart: [0, 0], distance: 0 }],
      frequency: fr, character: ch, q_cart: qc, q_frac: qf,
      eigenvectors_real: er, eigenvectors_imag: ei,
    },
  };
}

// payload + dos block (1d-2): total (dashed) + two projected curves on their
// own native grids — deliberately DIFFERENT frequency arrays (no resampling).
function synthNavDataDOS() {
  const d = synthNavData();
  // a Γ-M-K-Γ loop: Γ appears TWICE (same q_cart) -> the floor labels must
  // dedupe to 3 (Γ, M, K). The viewer's first HSP consumer is 0.6.19 Item 3.
  d.band.high_symmetry_points = [
    { label: "$\\Gamma$", q_cart: [0, 0], distance: 0 },
    { label: "M", q_cart: [0.1, 0.05], distance: 1 },
    { label: "K", q_cart: [0.08, 0.09], distance: 2 },
    { label: "$\\Gamma$", q_cart: [0, 0], distance: 3 },
  ];
  d.meta.conventions.character_group_a = ["Sn"];   // group A drives the DOS reds
  d.dos = {
    total: { frequency: [-0.5, 0, 1, 2], dos: [0, 0.2, 0.9, 0.1] },
    projected: {
      frequency: [-0.4, 0, 1, 2],
      curves: [{ label: "Sn", dos: [0, 0.1, 0.4, 0.05] },
               { label: "Se", dos: [0, 0.1, 0.5, 0.05] }],
    },
  };
  return d;
}

function loadPage({ withThree = true, navData = null, manifest = null, active = null } = {}) {
  const html = readFileSync(INDEX, "utf-8");
  const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true });
  const window = dom.window;
  const state = { newPlot: [], newPlotCalls: [], purge: [], restyle: [], deleteTraces: [], addTraces: [], cb: {}, recResize: [], window };
  window.Plotly = makePlotly(state);
  window.THREE = withThree ? makeTHREE(window) : undefined;

  const realK = require(KINEMATICS);
  const dmCalls = [];
  window.PhononKinematics = Object.assign({}, realK, {
    displaceMode: (...args) => { dmCalls.push(args); return realK.displaceMode(...args); },
  });

  // navData === "NONE" boots the bare TEMPLATE (no payload at all) — the
  // 0.6.96 explainer path; null/undefined keeps the synthetic default.
  const nd = navData === "NONE" ? undefined : (navData || synthNavData());
  window.NAV_DATA = nd;
  if (manifest) { window.NAV_MANIFEST = manifest; if (active) window.NAV_ACTIVE = active; }
  window.requestAnimationFrame = () => {};                 // NO-OP: must not loop
  if (!window.performance || !window.performance.now) window.performance = { now: () => 0 };

  const i = html.indexOf("(function ()");
  const j = html.lastIndexOf("})();") + "})();".length;
  let threw = null;
  try { window.eval(html.slice(i, j)); } catch (e) { threw = e; }
  return { window, state, dmCalls, navData: nd, threw, realK };
}

// effective marker.color / showscale for a band2d trace index, after replaying
// the recorded restyles over the initial newPlot data. applyColorMode drives
// the on-screen colour via restyle (0.6.36), so tests read the EFFECTIVE state,
// not the pre-restyle trace.
function effMarker(state, traceIdx) {
  const bp = state.newPlot.find(p => p.id === "band2d");
  const m = Object.assign({}, bp.data[traceIdx].marker);
  for (const r of state.restyle) {
    if (r.id !== "band2d") continue;
    const idxs = Array.isArray(r.idx) ? r.idx : [r.idx];
    const pos = idxs.indexOf(traceIdx);
    if (pos < 0) continue;
    for (const key of Object.keys(r.update)) {
      if (!key.startsWith("marker.")) continue;
      const sub = key.slice("marker.".length);
      const val = r.update[key];
      m[sub] = Array.isArray(val) ? val[pos] : val;
    }
  }
  return m;
}

// 3D-material payload (1c-5): dimensionality 3, NO surface/bz_polygon/
// ibz_polygon/z_range keys (omitted, exactly like the dim=3 exporter), and
// 3-component q_cart on the band + high-symmetry points.
// 2D payload WITHOUT the surface blocks (0.6.15): dimensionality 2, but baked
// with no mesh — surface/bz_polygon/ibz_polygon/z_range simply absent. The page
// must key the right panel off block PRESENCE, not dimensionality.
function synthNavData2DNoSurf() {
  const d = synthNavData();
  d.meta.material.dimensionality = 2;
  delete d.surface; delete d.bz_polygon; delete d.ibz_polygon; delete d.z_range;
  return d;
}

function synthNavData3D() {
  const d = synthNavData();
  d.meta.material.dimensionality = 3;
  delete d.surface; delete d.bz_polygon; delete d.ibz_polygon; delete d.z_range;
  d.band.q_cart = d.band.q_cart.map((q, i) => [q[0], q[1], 0.02 * i]);
  d.band.high_symmetry_points.forEach((h, i) => {
    h.q_cart = [h.q_cart[0], h.q_cart[1], 0.02 * i];
  });
  return d;
}

test("1: inline IIFE runs to completion without throwing", () => {
  const { threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
});

test("2: status text is no longer 'Loading…'", () => {
  const { window } = loadPage();
  assert.notEqual(window.document.getElementById("status").textContent, "Loading…");
});

test("3: Plotly.newPlot called for BOTH band2d and surf3d", () => {
  const { state } = loadPage();
  const ids = state.newPlot.map(p => p.id);
  assert.ok(ids.includes("band2d") && ids.includes("surf3d"), ids.join(","));
});

test("4: atom panel initialized; default tiling 3 -> Ns = Na*9 = 27", () => {
  const { window, navData } = loadPage();
  assert.ok(window.__atoms, "window.__atoms not defined");
  const slots = window.__atoms.getSlots();
  const Na = navData.meta.material.fractional_coords.length;
  assert.equal(slots.j.length, Na * 9);
});

test("5: click -> engine; eigenvectors === NAV_DATA.band.eigenvectors_real[qi][branch]", () => {
  const { window, state, dmCalls, navData } = loadPage();
  const qi = 1, branch = 3;
  const cb = state.cb["band2d:plotly_click"];
  assert.ok(cb, "no band2d plotly_click handler captured");
  cb({ points: [{ curveNumber: branch, pointNumber: qi }] });
  window.__atoms.applyFrame();                              // force one frame
  assert.ok(dmCalls.length > 0, "displaceMode never called");
  const last = dmCalls[dmCalls.length - 1];
  assert.equal(last[0], navData.band.eigenvectors_real[qi][branch]);  // single source
  assert.equal(last[1], navData.band.eigenvectors_imag[qi][branch]);
});

test("6: peak readout shows a finite number ending in Å", () => {
  const { window, state } = loadPage();
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const txt = window.document.getElementById("a-amp-val").textContent;
  assert.match(txt, /Å$/, txt);
  const num = parseFloat(txt.replace(/[^0-9.]/g, ""));
  assert.ok(Number.isFinite(num) && num >= 0, "peak not finite: " + txt);
});

test("7: graceful degradation without THREE (Plotly still inits, fallback shown)", () => {
  const { window, state, threw } = loadPage({ withThree: false });
  assert.equal(threw, null, threw && threw.stack);
  const ids = state.newPlot.map(p => p.id);
  assert.ok(ids.includes("band2d") && ids.includes("surf3d"), "Plotly panels did not init");
  assert.equal(window.__atoms, undefined, "atoms should be inert without THREE");
  assert.match(window.document.getElementById("atoms3d").innerHTML, /unavailable/i);
});

test("8: default-on-load selects branch 0 at M and sets BOTH markers (no click)", () => {
  const { state, dmCalls, navData } = loadPage();
  // qiM = index of the band distance closest to the first interior tick
  const tp = navData.band.tick_positions[1];
  let qiM = 0, best = Infinity;
  navData.band.distance.forEach((d, i) => { const e = Math.abs(d - tp); if (e < best) { best = e; qiM = i; } });

  // (a) the engine already ran on load (default selection fired) for branch 0 @ qiM
  assert.ok(dmCalls.length > 0, "displaceMode not called on load");
  assert.ok(dmCalls.some(c => c[0] === navData.band.eigenvectors_real[qiM][0]),
    "default selection did not drive the engine for branch 0 at qiM");

  // (b) band2d PICK2D marker got finite x/y
  const b2 = state.restyle.filter(r => r.id === "band2d" && r.update && r.update.x && r.update.y);
  assert.ok(b2.some(r => Number.isFinite(r.update.x[0][0]) && Number.isFinite(r.update.y[0][0])),
    "band2d selected-mode marker (PICK2D) has no finite x/y");

  // (c) surf3d PICK marker got finite x/y/z (PICK = the single-point restyle, z length 1)
  const pick = state.restyle.filter(r => r.id === "surf3d" && r.update && r.update.z
                                         && r.update.z[0] && r.update.z[0].length === 1);
  assert.ok(pick.some(r => Number.isFinite(r.update.x[0][0]) && Number.isFinite(r.update.y[0][0])
                           && Number.isFinite(r.update.z[0][0])),
    "surf3d selected-mode marker (PICK) has no finite x/y/z");
});

test("9: displacement arrows default ON (counts == slot count) and #a-arrows toggles visibility", () => {
  const { window } = loadPage();
  const n = window.__atoms.getSlots().j.length;
  const a0 = window.__atoms.arrows();
  // default ON: two instanced arrow meshes, each sized to the slot count (one
  // instance per displaced atom), both visible.
  assert.equal(a0.on, true, "arrows should default ON");
  assert.equal(a0.shaft.count, n, "shaft instance count != slot count");
  assert.equal(a0.head.count, n, "head instance count != slot count");
  assert.equal(a0.shaft.visible, true, "shaft not visible by default");
  assert.equal(a0.head.visible, true, "head not visible by default");
  // uncheck #a-arrows -> both hidden (the toggle path; turning ON also forces an
  // applyFrame, so this exercises the orientation loop under the mock too)
  const box = window.document.getElementById("a-arrows");
  box.checked = false; box.dispatchEvent(new window.Event("change"));
  const aOff = window.__atoms.arrows();
  assert.equal(aOff.shaft.visible, false, "shaft not hidden after uncheck");
  assert.equal(aOff.head.visible, false, "head not hidden after uncheck");
  // re-check -> visible again
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  const aOn = window.__atoms.arrows();
  assert.equal(aOn.shaft.visible, true, "shaft not re-shown after re-check");
  assert.equal(aOn.head.visible, true, "head not re-shown after re-check");
});

test("10: arrow base is at/outside the displaced sphere surface for a moving mode (not buried) + finite", () => {
  // This exercises the ORIENTATION else-branch (the path that shipped buried in
  // 0.6.11): the mock Vector3 now has real length()/normalize(), Matrix4.compose
  // records the translation, and setMatrixAt stores it per instance, so we can read
  // the arrow bases back. Default-on-load picks the M mode (branch 0), which moves.
  const { window, dmCalls, realK } = loadPage();
  window.__atoms.applyFrame();                                  // fresh frame; curER set on load
  assert.ok(dmCalls.length > 0, "displaceMode never called");
  // recompute the EXACT disp the frame used, from the recorded args
  const disp = realK.displaceMode(...dmCalls[dmCalls.length - 1]);
  const r0c = window.__atoms.getR0c();
  const rad = window.__atoms.getRad();
  const shaft = window.__atoms.arrows().shaft.positions;
  const mag = v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  let moving = 0;
  for (let k = 0; k < disp.length; k++) {
    const p = shaft[k];
    // zero-guard safety: NO position is ever NaN (even hidden/zero-scaled slots)
    assert.ok(p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
      "shaft position not finite at slot " + k);
    if (mag(disp[k]) > 1e-9) {
      moving++;
      const dx = p.x - r0c[k][0], dy = p.y - r0c[k][1], dz = p.z - r0c[k][2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      // base anchored at/outside the sphere — on 0.6.11 this was ~shaftLen/2 < rad
      // (buried); on 0.6.12 it is |disp| + rad + shaftLen/2 >= rad.
      assert.ok(dist >= rad[k],
        "arrow buried at slot " + k + ": dist " + dist.toFixed(3) + " < rad " + rad[k].toFixed(3));
    }
  }
  assert.ok(moving > 0, "no moving slots in the default M mode — assertion would be vacuous");
});

test("11: retiling via the three inputs -> 2x2x2 gives Na*8 slots, [1,1,1] replica, rebuilt arrows, finite frames", () => {
  const { window, navData } = loadPage();
  const Na = navData.meta.material.fractional_coords.length;
  const doc = window.document;
  doc.getElementById("a-t1").value = "2";
  doc.getElementById("a-t2").value = "2";
  doc.getElementById("a-t3").value = "2";
  doc.getElementById("a-t3").dispatchEvent(new window.Event("change"));
  const slots = window.__atoms.getSlots();
  assert.equal(slots.j.length, Na * 8, "2x2x2 should give Na*8 slots");
  // a genuine out-of-plane replica exists (n_int 3-component; blochPhase already
  // handles it — 2π·(n_int·q_frac) — so no engine change was needed)
  assert.ok(slots.n_int.some(n => n[0] === 1 && n[1] === 1 && n[2] === 1),
    "no [1,1,1] replica in the slot table");
  // arrow meshes were rebuilt to the new count (fixed at construction)
  const a = window.__atoms.arrows();
  assert.equal(a.shaft.count, Na * 8, "shaft not rebuilt to new slot count");
  assert.equal(a.head.count, Na * 8, "head not rebuilt to new slot count");
  // the surface-anchor machinery keeps working on the new tiling
  window.__atoms.applyFrame();
  const pos = window.__atoms.arrows().shaft.positions;
  assert.equal(pos.length, Na * 8, "shaft positions not populated for all slots");
  for (let k = 0; k < pos.length; k++) {
    const p = pos[k];
    assert.ok(p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
      "shaft position not finite at slot " + k + " after retile");
  }
});

test("12: slot cap refuses the rebuild (message shown, tiling unchanged); clamp writes back; valid entry clears", () => {
  const { window, navData } = loadPage();
  const Na = navData.meta.material.fractional_coords.length;
  const doc = window.document;
  const msg = () => doc.getElementById("a-tile-msg").textContent;
  // over the 5000-slot cap: 12*12*12*3 = 5184 -> REFUSED, previous tiling stays
  doc.getElementById("a-t1").value = "12";
  doc.getElementById("a-t2").value = "12";
  doc.getElementById("a-t3").value = "12";
  doc.getElementById("a-t1").dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.getSlots().j.length, Na * 9, "tiling changed despite cap refusal");
  assert.ok(msg().length > 0, "no cap message shown");
  assert.match(msg(), /5184/, "message should cite the offending slot count");
  // out-of-range entry clamps and is WRITTEN BACK to the input
  doc.getElementById("a-t1").value = "99";
  doc.getElementById("a-t1").dispatchEvent(new window.Event("change"));
  assert.equal(doc.getElementById("a-t1").value, "12", "clamp not written back to a-t1");
  // back to a valid tiling -> message cleared, 27 slots
  doc.getElementById("a-t1").value = "3";
  doc.getElementById("a-t2").value = "3";
  doc.getElementById("a-t3").value = "1";
  doc.getElementById("a-t2").dispatchEvent(new window.Event("change"));
  assert.equal(msg(), "", "message not cleared on valid entry");
  assert.equal(window.__atoms.getSlots().j.length, Na * 9, "valid 3x3x1 should give 27 slots");
});

test("13: 3D payload -> band2d-only layout, hidden surf3d/Zones, 3x3x3 default, ZERO surf3d restyles", () => {
  const { window, state, navData, threw } = loadPage({ navData: synthNavData3D() });
  assert.equal(threw, null, threw && threw.stack);
  // Plotly: band2d only — surf3d is never built for DIM === 3
  const ids = state.newPlot.map(p => p.id);
  assert.ok(ids.includes("band2d"), "band2d not plotted");
  assert.ok(!ids.includes("surf3d"), "surf3d must NOT be plotted for DIM=3");
  // wrapper + Zones hidden
  const sEl = window.document.getElementById("surf3d");
  assert.equal(sEl.parentElement.style.display, "none", "surf3d wrapper not hidden");
  const grp = window.document.getElementById("t-bz").closest(".group");
  assert.equal(grp.style.display, "none", "Zones group not hidden");
  // default tiling 3x3x3 with the a-t3 input synced
  const Na = navData.meta.material.fractional_coords.length;
  assert.equal(window.__atoms.getSlots().j.length, Na * 27, "3D default tiling should be 3x3x3");
  assert.equal(window.document.getElementById("a-t3").value, "3", "a-t3 not synced to 3");
  // click -> PICK2D finite; surf3d restyles ZERO across init+theme+default+click
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const b2 = state.restyle.filter(r => r.id === "band2d" && r.update && r.update.x && r.update.y);
  assert.ok(b2.some(r => Number.isFinite(r.update.x[0][0]) && Number.isFinite(r.update.y[0][0])),
    "band2d PICK2D marker has no finite x/y");
  assert.equal(state.restyle.filter(r => r.id === "surf3d").length, 0,
    "surf3d restyles must be ZERO in 3D mode (no dangling targets)");
  // the status formats a 3-component k generically
  assert.match(window.document.getElementById("status").innerHTML, /k=\([-0-9.]+, [-0-9.]+, [-0-9.]+\)/,
    "status should show a 3-component k");
  // arrows/atoms machinery intact on the 3x3x3 tiling
  window.__atoms.applyFrame();
  const pos = window.__atoms.arrows().shaft.positions;
  assert.equal(pos.length, Na * 27, "shaft positions not populated for all slots");
  for (let k = 0; k < pos.length; k++) {
    const p = pos[k];
    assert.ok(p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
      "shaft position not finite at slot " + k);
  }
});

test("14: 2D payload WITHOUT surface blocks -> band2d-only, hidden surf3d/Zones, 3x3x1 default, ZERO surf3d restyles", () => {
  const { window, state, navData, threw } = loadPage({ navData: synthNavData2DNoSurf() });
  assert.equal(threw, null, threw && threw.stack);
  const ids = state.newPlot.map(p => p.id);
  assert.ok(ids.includes("band2d"), "band2d not plotted");
  assert.ok(!ids.includes("surf3d"), "surf3d must NOT be plotted without a surface block");
  const sEl = window.document.getElementById("surf3d");
  assert.equal(sEl.parentElement.style.display, "none", "surf3d wrapper not hidden");
  const grp = window.document.getElementById("t-bz").closest(".group");
  assert.equal(grp.style.display, "none", "Zones group not hidden");
  // DIM is 2 -> default tiling stays 3x3x1 (27 slots), a-t3 untouched at "1"
  const Na = navData.meta.material.fractional_coords.length;
  assert.equal(window.__atoms.getSlots().j.length, Na * 9, "no-surface 2D default should stay 3x3x1");
  assert.equal(window.document.getElementById("a-t3").value, "1", "a-t3 should remain 1 for DIM=2");
  // click -> PICK2D finite; zero restyles ever target surf3d
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const b2 = state.restyle.filter(r => r.id === "band2d" && r.update && r.update.x && r.update.y);
  assert.ok(b2.some(r => Number.isFinite(r.update.x[0][0]) && Number.isFinite(r.update.y[0][0])),
    "band2d PICK2D marker has no finite x/y");
  assert.equal(state.restyle.filter(r => r.id === "surf3d").length, 0,
    "surf3d restyles must be ZERO without a surface block");
  // arrows/atoms machinery intact
  window.__atoms.applyFrame();
  const pos = window.__atoms.arrows().shaft.positions;
  assert.equal(pos.length, Na * 9, "shaft positions not populated for all slots");
  for (let k = 0; k < pos.length; k++) {
    const p = pos[k];
    assert.ok(p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z),
      "shaft position not finite at slot " + k);
  }
});

test("15: discontinuous path -> one null at the break; clicks past it map to the UNSHIFTED qi", () => {
  const { window, state, navData, dmCalls, threw } = loadPage({ navData: synthNavDataBreaks() });
  assert.equal(threw, null, threw && threw.stack);
  // plotted branch arrays: 6 q-points + 1 null after q-index 3 -> 7 entries,
  // null exactly at plotted index 4; color array aligned (same length).
  const tr0 = state.newPlot.find(p => p.id === "band2d").data[0];
  assert.equal(tr0.x.length, 7, "plotted x should be 6 points + 1 null");
  assert.equal(tr0.x[4], null, "x null must sit right after q-index 3");
  assert.equal(tr0.x.filter(v => v === null).length, 1, "exactly ONE null");
  assert.equal(tr0.y[4], null, "y null must align with x");
  assert.equal(tr0.marker.color.length, 7, "color array must stay aligned");
  assert.ok(Number.isFinite(tr0.x[5]) && Number.isFinite(tr0.y[5]), "post-break points intact");
  // click PAST the break: plotted pointNumber 6 -> band qi 5 (UNSHIFTED). The
  // synthetic eigenvectors are q-dependent, so receiving qi-5's block proves
  // the inverse map, not a coincidence. (On 0.6.15's identity routing this
  // click would have addressed qi 6 — out of range — or, one point earlier,
  // the WRONG qi.)
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 2, pointNumber: 6 }] });
  const qi = 5, branch = 2;
  assert.ok(dmCalls.length > 0, "engine never called after the click");
  const gotER = dmCalls[dmCalls.length - 1][0];
  assert.equal(JSON.stringify(gotER), JSON.stringify(navData.band.eigenvectors_real[qi][branch]),
    "engine must receive eigenvectors_real[5][2] — the unshifted qi");
  // PICK2D lands at qi-5's distance/frequency
  const picks = state.restyle.filter(r => r.id === "band2d" && r.update && r.update.x && r.update.y);
  const last = picks[picks.length - 1];
  assert.equal(last.update.x[0][0], navData.band.distance[qi], "PICK2D x must be distance[5]");
  assert.equal(last.update.y[0][0], navData.band.frequency[qi][branch], "PICK2D y must be freq[5][2]");
});

test("16: DOS payload -> xaxis2 + 3 traces after PICK2D; DOS clicks inert; theme recolors Total; no-DOS untouched", () => {
  const { window, state, navData, dmCalls, threw } = loadPage({ navData: synthNavDataDOS() });
  assert.equal(threw, null, threw && threw.stack);
  const nB = navData.band.frequency[0].length;            // 9
  const bp = state.newPlot.find(p => p.id === "band2d");
  // layout: split domain + the DOS sub-axis
  assert.equal(JSON.stringify(bp.layout.xaxis.domain), "[0,0.8]", "xaxis domain not split for DOS");
  assert.ok(bp.layout.xaxis2, "xaxis2 missing");
  assert.equal(JSON.stringify(bp.layout.xaxis2.domain), "[0.84,1]");
  // exactly 3 new traces, all AFTER PICK2D; hoisted indices untouched
  assert.equal(bp.n, nB + 2 + 3, "trace count should be branches + CBAR + PICK2D + 3 DOS");
  assert.equal(bp.data[nB + 1].marker.symbol, "diamond", "PICK2D index shifted");
  for (let i = nB + 2; i < nB + 5; i++) assert.equal(bp.data[i].xaxis, "x2", "DOS trace not on x2");
  assert.equal(bp.data[nB + 2].fill, "tozerox", "projected curve must fill from zero");
  assert.equal(bp.data[nB + 4].line.dash, "dash", "Total must be dashed");
  assert.equal(bp.data[nB + 4].name, "Total");
  // a branch click still works (PICK2D functional through the DOS additions)
  const r0 = state.restyle.length, c0 = dmCalls.length;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const picks = state.restyle.slice(r0).filter(r => r.id === "band2d" && r.update && r.update.x && r.update.y);
  assert.ok(picks.some(r => Number.isFinite(r.update.x[0][0]) && Number.isFinite(r.update.y[0][0])),
    "branch click no longer moves PICK2D");
  assert.ok(dmCalls.length > c0, "branch click should reach the engine");
  // a click whose curveNumber is a DOS trace is INERT (the >= nB guard)
  const r1 = state.restyle.length, c1 = dmCalls.length;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: nB + 2, pointNumber: 1 }] });
  assert.equal(state.restyle.length, r1, "DOS click must not restyle");
  assert.equal(dmCalls.length, c1, "DOS click must not reach the engine");
  // theme toggle recolors the Total line (hoisted DOS_TOTAL index)
  const r2 = state.restyle.length;
  window.document.getElementById("theme-toggle").click();
  const recolor = state.restyle.slice(r2).filter(r =>
    r.id === "band2d" && r.update && r.update["line.color"] &&
    Array.isArray(r.idx) && r.idx[0] === nB + 4);
  assert.equal(recolor.length, 1, "theme toggle must recolor the Total line exactly once");
  // no-DOS payloads: zero new traces, no xaxis2, no domain key
  const plain = loadPage();
  const bp0 = plain.state.newPlot.find(p => p.id === "band2d");
  assert.equal(bp0.n, nB + 2, "no-DOS payload must have no extra traces");
  assert.equal(bp0.layout.xaxis2, undefined, "no-DOS layout must lack xaxis2");
  assert.equal(bp0.layout.xaxis.domain, undefined, "no-DOS layout must lack the domain split");
});

test("17: DOS legend on DOS traces only; colorbar moved left; deduped HSP floor labels; no-DOS untouched", () => {
  const { window, state, navData, threw } = loadPage({ navData: synthNavDataDOS() });
  assert.equal(threw, null, threw && threw.stack);
  const nB = navData.band.frequency[0].length;
  const bp = state.newPlot.find(p => p.id === "band2d");
  // ITEM 1: every DOS trace in the legend, every non-DOS band2d trace out
  for (let i = 0; i < bp.n; i++) {
    const want = i >= nB + 2;            // DOS traces sit after CBAR + PICK2D
    assert.equal(!!bp.data[i].showlegend, want,
      "showlegend wrong at band2d trace " + i);
  }
  assert.ok(bp.layout.legend, "band2d layout must carry the legend block");
  assert.equal(bp.layout.legend.orientation, "h");
  // ITEM 2 (E1): the colorbar is RETIRED — the carrier keeps its index slot
  // (so PICK2D = nB+1 etc. don't shift) but renders no scale.
  assert.equal(bp.data[nB].marker.showscale, false, "CBAR carrier must have showscale:false (colorbar retired in E1)");
  assert.ok(!bp.data[nB].marker.colorbar, "CBAR carrier must carry no colorbar config in E1");
  // ITEM 3: exactly ONE trace after DROP on surf3d — the floor labels
  const sp = state.newPlot.find(p => p.id === "surf3d");
  assert.equal(sp.n, nB + 10, "surf3d: ...DROP + HSP + the 0.7.2 zone-tiling placeholder");
  const hsp = sp.data[nB + 8];
  assert.ok(hsp.mode.includes("text"), "floor-label trace must be a text trace");
  const zmin = navData.z_range[0];
  hsp.z.forEach(z => assert.equal(z, zmin, "floor labels must sit at z = zmin"));
  assert.equal(hsp.text.length, 3, "Γ-M-K-Γ must dedupe to 3 labels");
  assert.equal(hsp.text[0], "Γ", "Γ must render through tickLabel");
  // theme toggle restyles the floor-label color (hoisted HSP3D index)
  const r0 = state.restyle.length;
  window.document.getElementById("theme-toggle").click();
  const recolor = state.restyle.slice(r0).filter(r =>
    r.id === "surf3d" && r.update && r.update["textfont.color"] &&
    Array.isArray(r.idx) && r.idx[0] === nB + 8);
  assert.equal(recolor.length, 1, "theme toggle must recolor the floor labels once");
  // no-DOS payload: no legend key; the carrier is still present but hidden (E1)
  const plain = loadPage();
  const bp0 = plain.state.newPlot.find(p => p.id === "band2d");
  assert.equal(bp0.layout.legend, undefined, "no-DOS layout must lack the legend block");
  assert.equal(bp0.data[nB].marker.showscale, false, "carrier stays showscale:false regardless of DOS (E1)");
  assert.ok(!bp0.data[nB].marker.colorbar, "carrier carries no colorbar regardless of DOS (E1)");
});

test("18: bonds — geometric pair count, half-split midpoints under motion, toggle, live radius", () => {
  const { window, navData, dmCalls, realK, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  // recompute the EXPECTED pair list with the SAME criterion (real Math is
  // allowed here, never in index.html): equilibrium positions, TOL=1.2,
  // TRUE covalent radii from species_style.
  const M = navData.meta.material, L = M.lattice, fr = M.fractional_coords;
  const Na = fr.length, TOL = 1.2;
  const cart = fr.map(f => [
    f[0]*L[0][0] + f[1]*L[1][0] + f[2]*L[2][0],
    f[0]*L[0][1] + f[1]*L[1][1] + f[2]*L[2][1],
    f[0]*L[0][2] + f[1]*L[1][2] + f[2]*L[2][2]]);
  const pos = [], rc = [];
  for (let i1 = 0; i1 < 3; i1++) for (let i2 = 0; i2 < 3; i2++) for (let a = 0; a < Na; a++) {
    pos.push([cart[a][0] + i1*L[0][0] + i2*L[1][0],
              cart[a][1] + i1*L[0][1] + i2*L[1][1],
              cart[a][2] + i1*L[0][2] + i2*L[1][2]]);
    rc.push(M.species_style[M.symbols[a]].radius);
  }
  let expected = 0;
  const expectedPairs = [];
  for (let i = 0; i < pos.length; i++) for (let j = i + 1; j < pos.length; j++) {
    const d = Math.hypot(pos[i][0]-pos[j][0], pos[i][1]-pos[j][1], pos[i][2]-pos[j][2]);
    if (d <= TOL * (rc[i] + rc[j])) { expected++; expectedPairs.push([i, j]); }
  }
  assert.ok(expected > 0, "synthetic cell must produce bonds (vacuity guard)");
  const B = window.__atoms.bonds();
  assert.equal(B.list.length, expected, "viewer bond count != recomputed criterion");
  assert.equal(B.mesh.count, 2 * expected, "half-bond instance count != 2*pairs");
  assert.equal(JSON.stringify(B.list), JSON.stringify(expectedPairs), "pair lists differ");
  // half-split geometry under MOTION: i-half and j-half centers average to the
  // midpoint of the two DISPLACED atom centers
  window.__atoms.applyFrame();
  const disp = realK.displaceMode(...dmCalls[dmCalls.length - 1]);
  const r0c = window.__atoms.getR0c();
  const hp = window.__atoms.bonds().mesh.positions;
  for (let b = 0; b < B.list.length; b++) {
    const p0 = hp[2*b], p1 = hp[2*b + 1];
    assert.ok(p0 && Number.isFinite(p0.x) && Number.isFinite(p0.y) && Number.isFinite(p0.z),
      "i-half position not finite at bond " + b);
    assert.ok(p1 && Number.isFinite(p1.x) && Number.isFinite(p1.y) && Number.isFinite(p1.z),
      "j-half position not finite at bond " + b);
    const [bi, bj] = B.list[b];
    const mid = [
      (r0c[bi][0] + disp[bi][0] + r0c[bj][0] + disp[bj][0]) / 2,
      (r0c[bi][1] + disp[bi][1] + r0c[bj][1] + disp[bj][1]) / 2,
      (r0c[bi][2] + disp[bi][2] + r0c[bj][2] + disp[bj][2]) / 2];
    assert.ok(Math.abs((p0.x + p1.x)/2 - mid[0]) < 1e-9 &&
              Math.abs((p0.y + p1.y)/2 - mid[1]) < 1e-9 &&
              Math.abs((p0.z + p1.z)/2 - mid[2]) < 1e-9,
      "half centers do not average to the displaced midpoint at bond " + b);
  }
  // toggle
  const box = window.document.getElementById("a-bonds");
  box.checked = false; box.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.bonds().mesh.visible, false, "bonds not hidden");
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.bonds().mesh.visible, true, "bonds not re-shown");
  // live radius: 0.3 takes effect via per-instance scale, no rebuild
  const br = window.document.getElementById("a-bondr");
  br.value = "0.3"; br.dispatchEvent(new window.Event("change"));
  const sc = window.__atoms.bonds().mesh.scales[0];
  assert.equal(sc.x, 0.3, "bond radius scale.x not live");
  assert.equal(sc.z, 0.3, "bond radius scale.z not live");
  assert.ok(sc.y > 0, "bond half length must be positive");
  // GRID-BRANCH equivalence: 8x8x8 = 1536 slots crosses the 1500 threshold
  // (and stays under the 5000 cap), so the page's spatial-grid detector runs;
  // its pair SET must equal a brute-force O(N^2) recompute of the criterion.
  const doc = window.document;
  doc.getElementById("a-t1").value = "8";
  doc.getElementById("a-t2").value = "8";
  doc.getElementById("a-t3").value = "8";
  doc.getElementById("a-t3").dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.getSlots().j.length, Na * 512, "8x8x8 retile failed");
  const pos3 = [], rc3 = [];
  for (let i1 = 0; i1 < 8; i1++) for (let i2 = 0; i2 < 8; i2++) for (let i3 = 0; i3 < 8; i3++)
    for (let a = 0; a < Na; a++) {
      pos3.push([cart[a][0] + i1*L[0][0] + i2*L[1][0] + i3*L[2][0],
                 cart[a][1] + i1*L[0][1] + i2*L[1][1] + i3*L[2][1],
                 cart[a][2] + i1*L[0][2] + i2*L[1][2] + i3*L[2][2]]);
      rc3.push(M.species_style[M.symbols[a]].radius);
    }
  const brute = [];
  for (let i = 0; i < pos3.length; i++) for (let j = i + 1; j < pos3.length; j++) {
    const dx = pos3[i][0]-pos3[j][0], dy = pos3[i][1]-pos3[j][1], dz = pos3[i][2]-pos3[j][2];
    const cut = TOL * (rc3[i] + rc3[j]);
    if (dx*dx + dy*dy + dz*dz <= cut*cut) brute.push([i, j]);
  }
  const canon = lst => JSON.stringify(lst.map(p => p[0] + ":" + p[1]).sort());
  const gridList = window.__atoms.bonds().list;
  assert.ok(brute.length > 0, "8x8x8 brute-force found no bonds — vacuous");
  assert.equal(gridList.length, brute.length, "grid pair COUNT != brute force");
  assert.equal(canon(gridList), canon(brute), "grid pair SET != brute force");
});

test("19: cell outline toggles; Reset view restores the fitted camera", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const cell = window.__atoms.cell();
  assert.ok(cell, "cell outline missing");
  assert.equal(cell.visible, true, "cell outline should default visible");
  // 0.6.25: the cell control is now a <select>; "off" removes the lines,
  // "primitive" (the default) brings them back.
  const sel = window.document.getElementById("a-cell");
  sel.value = "off"; sel.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.cell(), null, "cell not removed for mode off");
  sel.value = "primitive"; sel.dispatchEvent(new window.Event("change"));
  assert.ok(window.__atoms.cell(), "cell not rebuilt for mode primitive");
  // Reset view: mutate, click, restored to the captured fit. The DIM-2
  // synthetic views DOWN a3 = [0,0,20], so the fitted camera sits on the
  // z-axis (x=y=0); we assert against the recorded camHome, projection-
  // agnostic (the full ortho/iso pose is owned by assertions 22/23).
  const C = window.__atoms.cam();
  assert.ok(C.home, "camera home not captured at fit");
  const hx = C.home.pos.x, hy = C.home.pos.y, hz = C.home.pos.z;
  assert.ok(Number.isFinite(hz) && Math.abs(hz) > 0, "fitted camera should be off-origin on z");
  C.camera.position.set(999, 888, 777);
  C.controls.target.set(5, 5, 5);
  window.document.getElementById("a-cam").click();
  assert.equal(C.camera.position.x, hx, "camera x not restored");
  assert.equal(C.camera.position.y, hy, "camera y not restored");
  assert.equal(C.camera.position.z, hz, "camera z not restored");
  assert.equal(C.controls.target.x, 0, "target x not restored");
  assert.equal(C.controls.target.y, 0, "target y not restored");
});

// the page's CHAR_SCALE sampler, mirrored here so the test asserts EXACT strings
// (the sampler is deterministic; this is the same math, kept in the test file).
function _sampleCharScale(t) {
  const S = [[0.0, "#2166ac"], [0.25, "#67a9cf"], [0.5, "#f7f7f7"],
             [0.75, "#ef8a62"], [1.0, "#b2182b"]];
  const x = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < S.length - 2 && x > S[i + 1][0]) i++;
  const t0 = S[i][0], t1 = S[i + 1][0];
  const f = t1 > t0 ? (x - t0) / (t1 - t0) : 0;
  const ch = (h, k) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);
  const mix = k => Math.round(ch(S[i][1], k) + f * (ch(S[i + 1][1], k) - ch(S[i][1], k)));
  const hx = v => (v < 16 ? "0" : "") + v.toString(16);
  return "#" + hx(mix(0)) + hx(mix(1)) + hx(mix(2));
}
function _rgba35(hex) {
  return "rgba(" + parseInt(hex.slice(1, 3), 16) + "," + parseInt(hex.slice(3, 5), 16) +
    "," + parseInt(hex.slice(5, 7), 16) + ",0.35)";
}

test("20: DOS fills take species_style colours (E1; the legend is the band-blend key)", () => {
  const { window, state, navData, threw } = loadPage({ navData: synthNavDataDOS() });
  assert.equal(threw, null, threw && threw.stack);
  const nB = navData.band.frequency[0].length;
  const bp = state.newPlot.find(p => p.id === "band2d");
  const sn = bp.data[nB + 2], se = bp.data[nB + 3], total = bp.data[nB + 4];
  assert.equal(sn.name, "Sn"); assert.equal(se.name, "Se"); assert.equal(total.name, "Total");
  // E1: each species fill/line is its OWN species_style colour (the DOS legend
  // == the fatband blend key), NOT a CHAR_SCALE gradient.
  const STYLE = navData.meta.material.species_style;
  const rgbOf = c => Math.round(c[0]*255) + "," + Math.round(c[1]*255) + "," + Math.round(c[2]*255);
  assert.equal(sn.line.color, "rgb(" + rgbOf(STYLE.Sn.color) + ")", "Sn line must be Sn's species colour");
  assert.equal(sn.fillcolor, "rgba(" + rgbOf(STYLE.Sn.color) + ",0.45)", "Sn fill must be Sn's species colour @0.45");
  assert.equal(se.line.color, "rgb(" + rgbOf(STYLE.Se.color) + ")", "Se line must be Se's species colour");
  assert.equal(se.fillcolor, "rgba(" + rgbOf(STYLE.Se.color) + ",0.45)", "Se fill must be Se's species colour @0.45");
  // Total untouched: theme fg, dashed, and still restyled on theme toggle
  assert.equal(total.line.dash, "dash");
  assert.equal(total.line.color, "#1a1a1a", "Total starts theme-fg (light)");
  const r0 = state.restyle.length;
  window.document.getElementById("theme-toggle").click();
  const tline = state.restyle.slice(r0).filter(r =>
    r.id === "band2d" && r.update && r.update["line.color"] &&
    Array.isArray(r.idx) && r.idx[0] === nB + 4);
  assert.equal(tline.length, 1, "theme toggle must still recolor the Total line");
  // legend flags unchanged (DOS traces in, others out)
  for (let i = 0; i < bp.n; i++)
    assert.equal(!!bp.data[i].showlegend, i >= nB + 2, "showlegend changed at trace " + i);
});

test("20b: each projected DOS curve takes its own species_style colour (E1; no CHAR_SCALE, no group-A logic)", () => {
  const nav = synthNavDataDOS();
  // three species with distinct baked colours; E1 colours each curve directly.
  nav.meta.material.species_style = {
    A: { color: [1.0, 0.0, 0.0], radius: 1 },
    B: { color: [0.0, 1.0, 0.0], radius: 1 },
    C: { color: [0.0, 0.0, 1.0], radius: 1 },
  };
  nav.dos.projected.curves = [
    { label: "A", dos: [0, 0.1, 0.2, 0.1] },
    { label: "B", dos: [0, 0.1, 0.3, 0.1] },
    { label: "C", dos: [0, 0.1, 0.4, 0.1] }];
  const { window, state, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const nB = nav.band.frequency[0].length;
  const bp = state.newPlot.find(p => p.id === "band2d");
  const want = {
    A: { line: "rgb(255,0,0)", fill: "rgba(255,0,0,0.45)" },
    B: { line: "rgb(0,255,0)", fill: "rgba(0,255,0,0.45)" },
    C: { line: "rgb(0,0,255)", fill: "rgba(0,0,255,0.45)" },
  };
  for (let k = 0; k < 3; k++) {
    const tr = bp.data[nB + 2 + k];
    assert.equal(tr.line.color, want[tr.name].line, tr.name + " line must be its species colour");
    assert.equal(tr.fillcolor, want[tr.name].fill, tr.name + " fill must be its species colour @0.45");
  }
  // E1 has no group-A coupling: character_group_a no longer affects DOS colours.
  // (Sanity: a curve whose label isn't in species_style falls back to grey,
  // proving the colour comes from species_style, not from any group logic.)
  const nav2 = synthNavDataDOS();
  nav2.meta.material.species_style = { A: { color: [1, 0, 0], radius: 1 } };
  nav2.dos.projected.curves = [
    { label: "A", dos: [0, 0.1, 0.2, 0.1] },
    { label: "Zz", dos: [0, 0.1, 0.3, 0.1] }];
  const r2 = loadPage({ navData: nav2 });
  const bp2 = r2.state.newPlot.find(p => p.id === "band2d");
  assert.equal(bp2.data[nB + 2].line.color, "rgb(255,0,0)", "A must take its species colour");
  assert.equal(bp2.data[nB + 3].line.color, "rgb(179,179,179)", "unknown species falls back to grey [0.7]");
});

test("22: ortho default + DIM-2 top-down fit; perspective toggle preserves the pose", () => {
  const { window, navData, threw } = loadPage();            // synthNavData -> DIM 2
  assert.equal(threw, null, threw && threw.stack);
  const cams = window.__atoms.cameras();
  const C = window.__atoms.cam();
  // ORTHOGRAPHIC is the active default (parallel projection, the scientific choice)
  assert.equal(C.camera, cams.ortho, "active camera must be orthographic on load");
  assert.equal(C.camera.isOrthographicCamera, true, "default not the ortho mock");
  assert.equal(C.home.persp, false, "camHome.persp must be false");
  assert.ok(window.__atoms.fit() > 0, "H_ext must be positive");
  // DIM-2 fit: view DOWN a3 (=[0,0,20] -> +z), up = a1 (=[3.8,0,0] -> +x).
  // dirV is the camera position direction (placed along +a3 from the centroid).
  const L = navData.meta.material.lattice;
  const norm = v => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0]/n, v[1]/n, v[2]/n]; };
  const a3 = norm(L[2]), a1 = norm(L[0]);
  const pos = C.camera.position, plen = Math.hypot(pos.x, pos.y, pos.z);
  const viewDir = [pos.x/plen, pos.y/plen, pos.z/plen];
  const dotV = viewDir[0]*a3[0] + viewDir[1]*a3[1] + viewDir[2]*a3[2];
  assert.ok(Math.abs(Math.abs(dotV) - 1) < 1e-6, "DIM-2 view direction must align with a3");
  const up = C.camera.up;
  const dotU = up.x*a1[0] + up.y*a1[1] + up.z*a1[2];
  assert.ok(Math.abs(Math.abs(dotU) - 1) < 1e-6, "DIM-2 up-vector must align with a1");
  // PERSPECTIVE toggle PRESERVES the pose (the user's ruling): position + target byte-equal
  const p0 = { x: pos.x, y: pos.y, z: pos.z };
  const t0 = { x: C.controls.target.x, y: C.controls.target.y, z: C.controls.target.z };
  const box = window.document.getElementById("a-persp");
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  const C2 = window.__atoms.cam();
  assert.equal(C2.camera, cams.persp, "toggle must activate the perspective camera");
  assert.equal(C2.camera.isPerspectiveCamera, true, "toggled camera not the perspective mock");
  assert.equal(C2.camera.position.x, p0.x, "position x changed on projection toggle");
  assert.equal(C2.camera.position.y, p0.y, "position y changed on projection toggle");
  assert.equal(C2.camera.position.z, p0.z, "position z changed on projection toggle");
  assert.equal(C2.controls.target.x, t0.x, "target changed on projection toggle");
  assert.equal(C2.controls.object, cams.persp, "controls.object not rewired to perspective");
  // toggle back -> ortho, same pose
  box.checked = false; box.dispatchEvent(new window.Event("change"));
  const C3 = window.__atoms.cam();
  assert.equal(C3.camera, cams.ortho, "toggle back must restore ortho");
  assert.equal(C3.camera.position.x, p0.x, "position not preserved toggling back");
  assert.equal(C3.camera.position.z, p0.z, "position not preserved toggling back");
});

test("23: DIM-3 iso fit; Reset restores camHome; gizmo toggles; perspective survives a retile", () => {
  const { window, navData, threw } = loadPage({ navData: synthNavData3D() });
  assert.equal(threw, null, threw && threw.stack);
  const C = window.__atoms.cam();
  // DIM-3: iso view — direction is the AABB diagonal, NOT a3. With this cell
  // the box spans all three axes, so the view dir must have a real z-component
  // and must NOT coincide with normalized a3.
  const L = navData.meta.material.lattice;
  const norm = v => { const n = Math.hypot(v[0], v[1], v[2]); return [v[0]/n, v[1]/n, v[2]/n]; };
  const a3 = norm(L[2]);
  const pos = C.camera.position, plen = Math.hypot(pos.x, pos.y, pos.z);
  const viewDir = [pos.x/plen, pos.y/plen, pos.z/plen];
  const dotA3 = viewDir[0]*a3[0] + viewDir[1]*a3[1] + viewDir[2]*a3[2];
  assert.ok(Math.abs(Math.abs(dotA3) - 1) > 1e-3, "DIM-3 must NOT view straight down a3 (iso, not top-down)");
  assert.ok(Math.abs(viewDir[0]) > 1e-3 && Math.abs(viewDir[1]) > 1e-3,
    "iso view should have in-plane components");
  // Reset restores camHome pos/tgt/up after mutation
  const hx = C.home.pos.x, hy = C.home.pos.y, hz = C.home.pos.z;
  C.camera.position.set(123, 456, 789);
  C.controls.target.set(9, 9, 9);
  window.document.getElementById("a-cam").click();
  assert.equal(C.camera.position.x, hx, "iso reset x");
  assert.equal(C.camera.position.y, hy, "iso reset y");
  assert.equal(C.camera.position.z, hz, "iso reset z");
  assert.equal(C.controls.target.x, 0, "iso reset target x");
  // gizmo toggles the triad object's visibility
  const giz = window.__atoms.gizmo();
  assert.equal(giz.visible, true, "gizmo should default visible");
  const gbox = window.document.getElementById("a-gizmo");
  gbox.checked = false; gbox.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.gizmo().visible, false, "gizmo not hidden");
  gbox.checked = true; gbox.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.gizmo().visible, true, "gizmo not re-shown");
  // perspective choice SURVIVES a retile (rebuild must not reset projection)
  const pbox = window.document.getElementById("a-persp");
  pbox.checked = true; pbox.dispatchEvent(new window.Event("change"));
  const cams = window.__atoms.cameras();
  assert.equal(window.__atoms.cam().camera, cams.persp, "perspective not active before retile");
  const doc = window.document;
  doc.getElementById("a-t1").value = "2"; doc.getElementById("a-t2").value = "2";
  doc.getElementById("a-t3").value = "2";
  doc.getElementById("a-t3").dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.cam().camera, cams.persp, "retile reset the projection (should persist)");
});

test("24: arrow thickness scales radius only (length invariant); custom color is absolute, auto follows theme", () => {
  const { window, dmCalls, realK, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  // ---- THICKNESS: radius is a live per-instance scale; length (scale.y) is untouched
  window.__atoms.applyFrame();
  const disp = realK.displaceMode(...dmCalls[dmCalls.length - 1]);
  // find a moving slot (the default M mode moves several)
  const mag = v => Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  let k = -1;
  for (let s = 0; s < disp.length; s++) if (mag(disp[s]) > 1e-9) { k = s; break; }
  assert.ok(k >= 0, "no moving slot in the default mode — thickness check would be vacuous");
  const a0 = window.__atoms.arrows();
  const shaft0 = a0.shaft.scales[k], head0 = a0.head.scales[k];
  assert.ok(shaft0 && head0, "arrow instance scales not recorded at the moving slot");
  const sX0 = shaft0.x, sY0 = shaft0.y, hX0 = head0.x, hY0 = head0.y;
  assert.ok(sX0 > 0 && hX0 > 0, "base radii must be positive at thickness 1");
  // bump thickness to 2 -> radius doubles on BOTH shaft and head, lengths unchanged
  const tb = window.document.getElementById("a-athick");
  tb.value = "2"; tb.dispatchEvent(new window.Event("change"));
  window.__atoms.applyFrame();
  const a1 = window.__atoms.arrows();
  assert.equal(a1.thick, 2, "arrowThick not updated");
  const shaft1 = a1.shaft.scales[k], head1 = a1.head.scales[k];
  assert.ok(Math.abs(shaft1.x - 2 * sX0) < 1e-12, "shaft radius did not double");
  assert.ok(Math.abs(head1.x - 2 * hX0) < 1e-12, "head radius did not double");
  assert.ok(Math.abs(shaft1.y - sY0) < 1e-12, "shaft LENGTH changed under thickness (must not)");
  assert.ok(Math.abs(head1.y - hY0) < 1e-12, "head LENGTH changed under thickness (must not)");
  // clamp writes back
  tb.value = "99"; tb.dispatchEvent(new window.Event("change"));
  assert.equal(tb.value, "5", "thickness clamp not written back");
  // ---- COLOR semantics: auto re-themes, custom is absolute
  const doc = window.document;
  // default auto: toggle to DARK -> arrow color becomes the dark fg
  doc.getElementById("theme-toggle").click();
  const darkFg = window.getComputedStyle(doc.documentElement).getPropertyValue("--fg").trim() ||
                 "#e8e8e8";   // fallback to the known dark fg if jsdom drops the var
  let mat = window.__atoms.arrows().mat;
  assert.ok(mat.color._v, "auto-mode theme toggle did not set an arrow color");
  const autoDark = mat.color._v;
  // pick a custom color -> Auto auto-unchecks, color is now ABSOLUTE
  const cp = doc.getElementById("a-arrowcol");
  cp.value = "#ff0000"; cp.dispatchEvent(new window.Event("input"));
  assert.equal(doc.getElementById("a-arrowauto").checked, false, "Auto did not auto-uncheck on pick");
  assert.equal(window.__atoms.arrows().auto, false, "arrowColorAuto not cleared on pick");
  assert.equal(window.__atoms.arrows().mat.color._v, "#ff0000", "picked color not applied");
  // toggle back to LIGHT -> the custom red is NOT re-themed (the absolute guarantee)
  doc.getElementById("theme-toggle").click();
  assert.equal(window.__atoms.arrows().mat.color._v, "#ff0000",
    "custom color was overwritten on theme toggle (must be absolute)");
  // --- the 0.6.24 toggle-memory fix (the bug): CHECK Automatic -> theme fg;
  // UNCHECK -> the personalized colour comes BACK (not stranded at theme fg).
  const ab = doc.getElementById("a-arrowauto");
  ab.checked = true; ab.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.arrows().auto, true, "Automatic not re-enabled on check");
  const themedFg = window.__atoms.arrows().mat.color._v;
  assert.notEqual(themedFg, "#ff0000", "checking Automatic must switch to the theme fg");
  ab.checked = false; ab.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.arrows().auto, false, "Automatic not cleared on uncheck");
  assert.equal(window.__atoms.arrows().mat.color._v, "#ff0000",
    "UNCHECKING Automatic must RESTORE the personalized colour, not strand at theme fg");
  // and a theme toggle while unchecked still leaves the personalized colour put
  doc.getElementById("theme-toggle").click();
  assert.equal(window.__atoms.arrows().mat.color._v, "#ff0000",
    "personalized colour must survive a theme toggle while Automatic is off");
});

test("25: cell modes as lit cylinders — counts, envelope, live edge thickness (geometric)", () => {
  const { window, navData, threw } = loadPage();            // DIM-2 synthetic
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // the cell is now an InstancedMesh of cylinders: one per EDGE (12 per box)
  const ecount = () => { const c = window.__atoms.cell(); return c ? c.count : 0; };
  const midXspan = () => {
    // in-plane x extent of the cylinder MIDPOINTS: super stretches x by t1,
    // primitive does not. (max|coord| is swamped by the shared 20 Å c-axis at
    // t3=1, so it can't distinguish them — the 0.6.25 lesson, on midpoints.)
    const ps = window.__atoms.cell().positions;
    let lo = Infinity, hi = -Infinity;
    for (const p of ps) { if (!p) continue; lo = Math.min(lo, p.x); hi = Math.max(hi, p.x); }
    return hi - lo;
  };
  // set a 2x2x1 tiling for round edge counts
  doc.getElementById("a-t1").value = "2";
  doc.getElementById("a-t2").value = "2";
  doc.getElementById("a-t3").value = "1";
  doc.getElementById("a-t3").dispatchEvent(new window.Event("change"));
  // 0.6.32 default is "super"; this test exercises every mode, so start from
  // primitive explicitly (the on-load default is checked in the rail-sections test).
  const sel = doc.getElementById("a-cell");
  sel.value = "primitive"; sel.dispatchEvent(new window.Event("change"));
  assert.equal(ecount(), 12, "primitive must be 12 edge cylinders");
  const primSpan = midXspan();
  // grid: 12 edges per replica -> 12 * (2*2*1) = 48
  sel.value = "grid"; sel.dispatchEvent(new window.Event("change"));
  assert.equal(ecount(), 12 * 4, "grid must be 12 edges per replica (2x2x1 -> 48)");
  // super: 12 edges, but the midpoint x-span is strictly larger than primitive
  sel.value = "super"; sel.dispatchEvent(new window.Event("change"));
  assert.equal(ecount(), 12, "super must be a single 12-edge box");
  assert.ok(midXspan() > primSpan + 1e-9,
    "super envelope midpoints must span strictly wider in-plane than primitive");
  // off: no mesh
  sel.value = "off"; sel.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.cell(), null, "off must remove the cell mesh");
  // grid tracks the tiling: retile to 3x2x1 -> 12 * 6 = 72
  sel.value = "grid"; sel.dispatchEvent(new window.Event("change"));
  doc.getElementById("a-t1").value = "3";
  doc.getElementById("a-t1").dispatchEvent(new window.Event("change"));
  assert.equal(ecount(), 12 * 6, "grid must re-grid with the tiling (3x2x1 -> 72)");
  // THICKNESS: radius is a live per-instance scale; edge LENGTH (scale.y) is invariant.
  sel.value = "primitive"; sel.dispatchEvent(new window.Event("change"));
  const s0 = window.__atoms.cell().scales.slice().map(s => ({ x: s.x, y: s.y }));
  assert.ok(s0.length === 12 && s0[0].x > 0, "cell cylinder scales not recorded");
  const eb = doc.getElementById("a-celledge");
  eb.value = "0.06"; eb.dispatchEvent(new window.Event("change"));
  const s1 = window.__atoms.cell().scales;
  for (let e = 0; e < 12; e++) {
    assert.ok(Math.abs(s1[e].x - 2 * s0[e].x) < 1e-12, "edge radius.x did not double at edge " + e);
    assert.ok(Math.abs(s1[e].z - 2 * s0[e].x) < 1e-12, "edge radius.z did not double at edge " + e);
    assert.ok(Math.abs(s1[e].y - s0[e].y) < 1e-12, "edge LENGTH changed under thickness at edge " + e);
  }
  // clamp writes back
  eb.value = "99"; eb.dispatchEvent(new window.Event("change"));
  assert.equal(eb.value, "0.3", "cell edge clamp not written back");
});

test("26: amplitude slider is peak displacement in Å (engine gain back-solved by linearity)", () => {
  const { window, dmCalls, realK, threw } = loadPage();   // default-on-load moving mode
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const masses = window.NAV_DATA.meta.material.masses;
  const slots = window.__atoms.getSlots();
  // the args the engine last received (curER/curEI/curQF + the back-solved amp);
  // we re-sweep at THAT amp and confirm the realized peak equals the slider Å.
  const lastArgs = dmCalls[dmCalls.length - 1];
  const curER = lastArgs[0], curEI = lastArgs[1], curQF = lastArgs[4];
  const peakAt = ampUsed => {
    let peak = 0;
    for (let i = 0; i < 24; i++) {
      const d = realK.displaceMode(curER, curEI, masses, slots, curQF, ampUsed, 2*Math.PI*i/24, true);
      for (const v of d) peak = Math.max(peak, Math.hypot(v[0], v[1], v[2]));
    }
    return peak;
  };
  const ampNow = () => dmCalls[dmCalls.length - 1][5];   // amplitude arg of the latest call
  const ampEl = doc.getElementById("a-amp");
  // set 0.2 Å -> the realized peak |u| must equal 0.2 Å
  ampEl.value = "0.2"; ampEl.dispatchEvent(new window.Event("input"));
  window.__atoms.applyFrame();
  assert.ok(Math.abs(peakAt(ampNow()) - 0.2) < 1e-6, "peak |u| != 0.2 Å at slider 0.2");
  // pick a fixed MOVING slot and record its TOTAL arrow extent. The shaft alone
  // is affine (shaftLen = GAIN*|disp| - headH), so it does NOT scale ~2x; the
  // full arrow length shaftLen + headH = GAIN*|disp| IS proportional to |disp|.
  const sc02 = window.__atoms.arrows().shaft.scales;
  let kslot = -1;
  for (let s = 0; s < sc02.length; s++) if (sc02[s] && sc02[s].y > 1e-3) { kslot = s; break; }
  assert.ok(kslot >= 0, "no moving arrow slot — ratio check would be vacuous");
  const headY02 = window.__atoms.arrows().head.scales[kslot].y;   // == arrowHeadH (head LENGTH)
  const total02 = sc02[kslot].y + headY02;                        // shaftLen + headH = GAIN*|disp|
  // set 0.4 Å -> realized peak |u| must be 0.4 (linearity), readout ends in Å
  ampEl.value = "0.4"; ampEl.dispatchEvent(new window.Event("input"));
  window.__atoms.applyFrame();
  assert.ok(Math.abs(peakAt(ampNow()) - 0.4) < 1e-6, "peak |u| != 0.4 Å at slider 0.4");
  const valTxt = doc.getElementById("a-amp-val").textContent;
  assert.equal(valTxt, "peak 0.40 Å", "readout should read exactly 'peak 0.40 Å'");
  assert.match(valTxt, /Å$/, "readout must still end in Å (assertion 6 contract)");
  // total arrow extent on the SAME slot scales ~2x from 0.2 -> 0.4 Å
  const sc04 = window.__atoms.arrows().shaft.scales;
  const total04 = sc04[kslot].y + window.__atoms.arrows().head.scales[kslot].y;
  assert.ok(Math.abs(total04 / total02 - 2) < 1e-6, "arrow length did not scale ~2x with peak-Å");
});

test("27: E1 retires the 2D group-A colour language (colorbar, hover scalar, char-grp prose all gone)", () => {
  // Phase E1 replaced the 2D panel's group-A CHAR_SCALE projection with a
  // per-atom |e|² blend, so the colorbar carrier, the hover "<group>=%{...}"
  // scalar, and the intro .char-grp spans no longer exist. (The label
  // derivation this assertion previously guarded is superseded; CHAR_GROUP/
  // CHAR_TITLE remain only for the 3D surface, which itself carries no visible
  // scale until E2.) Verify the 2D panel is fully off the old scale regardless
  // of the payload's character_group_a.
  const navSe = synthNavData();
  navSe.meta.conventions.character_group_a = ["Se"];
  const { window, state, threw } = loadPage({ navData: navSe });
  assert.equal(threw, null, threw && threw.stack);
  const bandData = state.newPlot.find(p => p.id === "band2d").data;
  // no colorbar anywhere on band2d
  assert.ok(!bandData.some(t => t.marker && t.marker.colorbar),
    "band2d must carry no colorbar in E1");
  // the carrier (index nB) is present but hidden
  const nB = navSe.band.frequency[0].length;
  assert.equal(bandData[nB].marker.showscale, false, "carrier must be showscale:false");
  // the branch hover no longer carries the group-A scalar readout
  const branch = bandData.find(t => t.hovertemplate && /branch 0/.test(t.hovertemplate));
  assert.ok(branch, "a branch trace with a hovertemplate must exist");
  assert.ok(!/Se=|Sn=|=%\{marker\.color/.test(branch.hovertemplate),
    "branch hover must not carry the retired group-A scalar readout");
  // the fatband marker is an RGB-string array (the blend), not a scalar+colorscale
  assert.ok(Array.isArray(branch.marker.color) && /^rgb/.test(branch.marker.color[0]),
    "fatband marker.color must be an rgb() blend array");
  assert.ok(!branch.marker.colorscale, "fatband marker must have no colorscale in E1");
  // the intro prose no longer has .char-grp spans
  assert.equal(window.document.querySelectorAll(".char-grp").length, 0,
    "intro prose must have no .char-grp spans in E1");
});

test("28: Reset amplitude restores the default slider value and re-applies the back-solve", () => {
  const { window, dmCalls, realK, threw } = loadPage();   // default-on-load moving mode
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const masses = window.NAV_DATA.meta.material.masses;
  const slots = window.__atoms.getSlots();
  const lastArgs = dmCalls[dmCalls.length - 1];
  const curER = lastArgs[0], curEI = lastArgs[1], curQF = lastArgs[4];
  const ampNow = () => dmCalls[dmCalls.length - 1][5];
  const peakAt = a => {
    let peak = 0;
    for (let i = 0; i < 24; i++) {
      const d = realK.displaceMode(curER, curEI, masses, slots, curQF, a, 2*Math.PI*i/24, true);
      for (const v of d) peak = Math.max(peak, Math.hypot(v[0], v[1], v[2]));
    }
    return peak;
  };
  // drive the slider to 0.8, confirm it took (the back-solve realizes 0.8)
  const ampEl = doc.getElementById("a-amp");
  ampEl.value = "0.8"; ampEl.dispatchEvent(new window.Event("input"));
  window.__atoms.applyFrame();
  assert.ok(Math.abs(peakAt(ampNow()) - 0.8) < 1e-6, "precondition: slider 0.8 should realize peak 0.8");
  // click Reset amplitude
  doc.getElementById("a-amp-reset").dispatchEvent(new window.Event("click"));
  // slider snaps back to the HTML default
  assert.equal(ampEl.value, "0.2", "slider not restored to defaultValue 0.2");
  // readout rewritten to the default, still ends in Å
  const valTxt = doc.getElementById("a-amp-val").textContent;
  assert.equal(valTxt, "peak 0.20 Å", "readout not reset to 'peak 0.20 Å'");
  assert.match(valTxt, /Å$/, "readout must still end in Å");
  // the GAIN was re-back-solved (not stranded at the 0.8 value): realized peak == 0.2
  window.__atoms.applyFrame();
  assert.ok(Math.abs(peakAt(ampNow()) - 0.2) < 1e-6, "reset did not re-apply the back-solve to 0.2");
});

test("29: left rail — DOM structure, collapse toggle, and plot re-fit on collapse", () => {
  const { window, state, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // structure: the new containers exist and the moved nodes are descendants
  const layout = doc.getElementById("layout");
  const rail = doc.getElementById("left-rail");
  const main = doc.getElementById("main");
  const toggle = doc.getElementById("rail-toggle");
  assert.ok(layout && rail && main && toggle, "layout/left-rail/main/rail-toggle must all exist");
  assert.ok(main.contains(doc.getElementById("atoms3d")), "#atoms3d must be inside #main");
  assert.ok(main.contains(doc.getElementById("panels")), "#panels must be inside #main");
  assert.ok(rail.contains(doc.getElementById("atoms-controls")), "#atoms-controls must be inside #left-rail");
  // a couple of moved control ids are still reachable (behavior preservation spot-check)
  assert.ok(rail.contains(doc.getElementById("a-amp")), "a-amp must have moved into the rail");
  assert.ok(rail.contains(doc.getElementById("a-cam")), "a-cam must have moved into the rail");
  // collapse toggle flips the .collapsed class; the chevron points to the ACTION
  // (0.6.32): « collapse-left when open, » expand-right when collapsed.
  assert.equal(rail.classList.contains("collapsed"), false, "rail should start expanded");
  assert.equal(toggle.textContent, "«", "chevron should read « when expanded");
  toggle.dispatchEvent(new window.Event("click"));
  assert.equal(rail.classList.contains("collapsed"), true, "first click must collapse the rail");
  assert.equal(toggle.textContent, "»", "chevron should flip to » when collapsed");
  toggle.dispatchEvent(new window.Event("click"));
  assert.equal(rail.classList.contains("collapsed"), false, "second click must expand the rail");
  assert.equal(toggle.textContent, "«", "chevron should flip back to «");
  // the collapse re-fits BOTH plots (responsive:true does not fire on rail toggle)
  const n0 = state.recResize.length;
  toggle.dispatchEvent(new window.Event("click"));
  const fired = state.recResize.slice(n0);
  assert.ok(fired.includes("band2d"), "rail collapse must resize band2d");
  assert.ok(fired.includes("surf3d"), "rail collapse must resize surf3d");
});

test("30: rail chrome — #rail-hint present in the rail; motion cluster moved below display controls", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // CHANGE A: the open-affordance hint exists and lives in the rail
  const hint = doc.getElementById("rail-hint");
  assert.ok(hint, "#rail-hint must exist");
  assert.ok(doc.getElementById("left-rail").contains(hint), "#rail-hint must be inside #left-rail");
  // CHANGE C: within #atoms-controls, a-play now sits AFTER the display controls
  // (a-cell, a-gizmo) — confirms the motion cluster moved to the bottom.
  const kids = Array.from(doc.getElementById("atoms-controls").children);
  const idxOfDesc = id => {
    // index of the child that IS or CONTAINS the element with this id
    const el = doc.getElementById(id);
    return kids.findIndex(k => k === el || k.contains(el));
  };
  const iPlay = idxOfDesc("a-play"), iCell = idxOfDesc("a-cell"), iGizmo = idxOfDesc("a-gizmo");
  assert.ok(iPlay >= 0 && iCell >= 0 && iGizmo >= 0, "a-play/a-cell/a-gizmo must all be in #atoms-controls");
  assert.ok(iPlay > iCell, "a-play must come after a-cell (motion moved below display controls)");
  assert.ok(iPlay > iGizmo, "a-play must come after a-gizmo (motion moved below display controls)");
  // a-play still reachable + still the play/pause button (listener preserved by the move)
  assert.equal(doc.getElementById("a-play").tagName, "BUTTON", "a-play must still be the play button");
});

test("31: rail sections, Reset speed, Supercell default, chevron direction", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // CHANGE A: chevron reads « when expanded (the action: collapse-left)
  assert.equal(doc.getElementById("rail-toggle").textContent, "«", "chevron must read « on load (expanded)");
  // CHANGE D: default Cell display is Supercell box
  assert.equal(doc.getElementById("a-cell").value, "super", "default cell mode must be super");
  // CHANGE C: Reset speed exists and restores the default
  const spReset = doc.getElementById("a-speed-reset");
  assert.ok(spReset, "#a-speed-reset must exist");
  const sp = doc.getElementById("a-speed");
  sp.value = "2"; sp.dispatchEvent(new window.Event("input"));
  spReset.dispatchEvent(new window.Event("click"));
  assert.equal(sp.value, "0.7", "Reset speed must restore the default 0.7");
  // CHANGE E: exactly four .rail-section boxes, titled in order
  const sections = Array.from(doc.querySelectorAll(".rail-section"));
  assert.equal(sections.length, 5, "there must be five rail sections (Modes added in Phase D)");
  const titles = sections.map(s => s.querySelector(".rail-section-title").textContent);
  assert.deepEqual(titles, ["Modes", "Arrows", "Bonds", "Cell", "Atoms"], "section titles wrong or out of order");
  // each control lands in the right section
  const sectionTitleOf = id =>
    doc.getElementById(id).closest(".rail-section").querySelector(".rail-section-title").textContent;
  assert.equal(sectionTitleOf("a-arrows"), "Arrows", "a-arrows not in Arrows");
  assert.equal(sectionTitleOf("a-bonds"), "Bonds", "a-bonds not in Bonds");
  assert.equal(sectionTitleOf("a-cell"), "Cell", "a-cell not in Cell");
  assert.equal(sectionTitleOf("a-play"), "Atoms", "a-play not in Atoms");
  assert.equal(sectionTitleOf("a-amp"), "Atoms", "a-amp not in Atoms");
  assert.equal(sectionTitleOf("a-info"), "Modes", "a-info moved to the Modes section in Phase D");
  // within the Atoms box: a-amp precedes the "Atom motion" label precedes a-play
  const atomsBox = doc.getElementById("a-play").closest(".rail-section");
  const flat = Array.from(atomsBox.querySelectorAll("*"));
  const idxOf = el => flat.indexOf(el);
  const ampEl = doc.getElementById("a-amp");
  const motionLabel = Array.from(atomsBox.querySelectorAll(".label")).find(e => /Atom motion/.test(e.textContent));
  assert.ok(motionLabel, "Atom motion label missing from the Atoms box");
  assert.ok(idxOf(ampEl) < idxOf(motionLabel), "a-amp must precede the Atom motion label");
  assert.ok(idxOf(motionLabel) < idxOf(doc.getElementById("a-play")), "Atom motion label must precede a-play");
});

test("32: draggable vertical splitter — DOM order, proportional resize, re-fit on drag", () => {
  const { window, state, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // structure: #vsplit between #panels and #atoms3d, all inside #main
  const split = doc.getElementById("vsplit");
  const panels = doc.getElementById("panels");
  const atoms = doc.getElementById("atoms3d");
  const main = doc.getElementById("main");
  assert.ok(split && panels && atoms && main, "vsplit/panels/atoms3d/main must all exist");
  const kids = Array.from(main.children);
  const iPanels = kids.indexOf(panels), iSplit = kids.indexOf(split), iAtoms = kids.indexOf(atoms);
  assert.ok(iPanels >= 0 && iSplit >= 0 && iAtoms >= 0, "all three must be direct children of #main");
  assert.ok(iPanels < iSplit && iSplit < iAtoms, "DOM order must be panels, vsplit, atoms3d");
  // mock #main geometry so the drag math runs (jsdom has no layout)
  main.getBoundingClientRect = () => ({ top: 0, left: 0, width: 1000, height: 1000, right: 1000, bottom: 1000 });
  // drag the splitter to 60% down
  const n0 = state.recResize.length;
  split.dispatchEvent(new window.MouseEvent("pointerdown", { clientY: 0, bubbles: true }));
  split.dispatchEvent(new window.MouseEvent("pointermove", { clientY: 600, bubbles: true }));
  split.dispatchEvent(new window.MouseEvent("pointerup", { clientY: 600, bubbles: true }));
  // proportional flex-grow: 60 / 40. (jsdom's CSSOM normalizes the written
  // "60.00"/"40.00" to "60"/"40"; assert the numeric value, not the literal.)
  assert.equal(parseFloat(panels.style.flexGrow), 60, "panels flex-grow should be 60 at 60% drag");
  assert.equal(parseFloat(atoms.style.flexGrow), 40, "atoms3d flex-grow should be 40 at 60% drag");
  // pointerup's synchronous refit re-fits BOTH plots (rAF need not be flushed)
  const fired = state.recResize.slice(n0);
  assert.ok(fired.includes("band2d"), "splitter drag must resize band2d");
  assert.ok(fired.includes("surf3d"), "splitter drag must resize surf3d");
});

test("33: Modes section — steppers (clamped), jump buttons, selection state", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // a Modes section exists and is FIRST; #a-info lives in it
  const sections = Array.from(doc.querySelectorAll(".rail-section"));
  assert.equal(sections[0].querySelector(".rail-section-title").textContent, "Modes",
    "Modes must be the first rail section");
  assert.ok(sections[0].contains(doc.getElementById("a-info")), "#a-info must be in the Modes box");
  // default-on-load selection (branch 0)
  assert.equal(window.__atoms.sel().branch, 0, "default-on-load branch should be 0");
  // branch stepper: next -> 1, prev twice -> clamp at 0
  doc.getElementById("mode-branch-next").dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().branch, 1, "branch-next should advance to 1");
  assert.equal(doc.getElementById("mode-branch-cur").textContent, "1", "branch readout should track to 1");
  doc.getElementById("mode-branch-prev").dispatchEvent(new window.Event("click"));
  doc.getElementById("mode-branch-prev").dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().branch, 0, "branch-prev must clamp at 0 (not -1)");
  assert.equal(doc.getElementById("mode-branch-cur").textContent, "0", "branch readout should track back to 0");
  // q stepper. The synth has only 2 q-points (0,1) and the default-on-load
  // lands on q=1 (= tick_positions[1], the MAX index), so q-next there is
  // already clamped. Jump to Γ (q=0) first to test stepping from a known end.
  const jbtns0 = Array.from(doc.querySelectorAll("#mode-jump-buttons button"));
  const gamma0 = jbtns0.find(b => /Γ/.test(b.textContent));
  assert.ok(gamma0, "a Γ jump button should exist");
  gamma0.dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().q, 0, "Γ jump should set q to 0");
  doc.getElementById("mode-q-next").dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().q, 1, "q-next should increment q from 0 to 1");
  doc.getElementById("mode-q-prev").dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().q, 0, "q-prev should return to q=0");
  doc.getElementById("mode-q-prev").dispatchEvent(new window.Event("click"));   // already at 0
  assert.ok(window.__atoms.sel().q >= 0, "q-prev must clamp at 0 (never negative)");
  // jump buttons: one per UNIQUE high-symmetry point (synth base has Γ + M = 2)
  const jbtns = Array.from(doc.querySelectorAll("#mode-jump-buttons button"));
  const uniqHSP = new Set(window.NAV_DATA.band.high_symmetry_points.map(
    h => Math.round(h.q_cart[0]*1e6) + "," + Math.round(h.q_cart[1]*1e6)));
  assert.equal(jbtns.length, uniqHSP.size, "one jump button per unique high-symmetry point");
  // move away then jump back to Γ; #a-info reflects the jumped mode
  doc.getElementById("mode-q-next").dispatchEvent(new window.Event("click"));
  jbtns.find(b => /Γ/.test(b.textContent)).dispatchEvent(new window.Event("click"));
  assert.equal(window.__atoms.sel().q, 0, "Γ jump should set q to its index (0 in the synth)");
  assert.match(doc.getElementById("a-info").textContent, /ω = .* THz/,
    "#a-info should show the mode readout (ω = … THz) after the jump");
});

test("34: Atom-blend mode -> per-atom |e|² species blend (effective restyle); carrier hidden", () => {
  const { window, state, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const nB = navData.band.frequency[0].length;
  const SYMS = navData.meta.material.symbols;
  const STY = navData.meta.material.species_style;
  const ER = navData.band.eigenvectors_real, EI = navData.band.eigenvectors_imag;
  // independent reference blend (same convention as the viewer: w_a = |e_a|²,
  // f_a = w_a/Σw, RGB = Σ f_a · species_color, rounded to 0-255).
  const refBlend = (qi, b) => {
    const er = ER[qi][b], ei = EI[qi][b];
    const w = SYMS.map((_, a) => {
      let m = 0; for (let d = 0; d < 3; d++) m += er[a][d]*er[a][d] + ei[a][d]*ei[a][d];
      return m;
    });
    const tot = w.reduce((s, x) => s + x, 0);
    let r = 0, g = 0, bl = 0;
    SYMS.forEach((s, a) => {
      const f = tot > 0 ? w[a] / tot : 1 / SYMS.length;
      const c = (STY[s] || {}).color || [0.7, 0.7, 0.7];
      r += f*c[0]; g += f*c[1]; bl += f*c[2];
    });
    return "rgb(" + Math.round(r*255) + "," + Math.round(g*255) + "," + Math.round(bl*255) + ")";
  };
  // the 2-species synth defaults to "species"; SELECT "Atom blend" first.
  const rb = doc.getElementById("a-cmode-blend");
  rb.checked = true; rb.dispatchEvent(new window.Event("change"));
  // effective branch-0 marker is the blend RGB array; spot-check q0 and q1
  const m0 = effMarker(state, 0);
  assert.ok(Array.isArray(m0.color), "blend marker.color must be a per-point array");
  assert.equal(m0.color[0], refBlend(0, 0), "blend mismatch at (q0, branch0)");
  assert.equal(m0.color[1], refBlend(1, 0), "blend mismatch at (q1, branch0)");
  assert.ok(/^rgb\(/.test(m0.color[0]), "blend marker colour must be an rgb() string");
  assert.ok(!m0.colorscale, "blend marker must carry no colorscale");
  assert.equal(effMarker(state, nB).showscale, false, "carrier must be showscale:false in blend mode");
  // worked example from the task spec: 3-atom Sn/Se/Se,
  // er [[.2,.1,0],[.1,.1,0],[.1,0,0]], ei 0; |e|²=.05/.02/.01 tot .08 ->
  // f=.625/.25/.125; Sn=[.4,.5,.5] Se=[1,.6,0] -> rgb(159,137,80). (2 species ->
  // defaults to species, so select Atom blend.)
  {
    const nav = synthNavData();
    nav.band.eigenvectors_real[0][0] = [[.2,.1,0],[.1,.1,0],[.1,0,0]];
    nav.band.eigenvectors_imag[0][0] = [[0,0,0],[0,0,0],[0,0,0]];
    nav.meta.material.symbols = ["Sn", "Se", "Se"];
    nav.meta.material.species_style = { Sn: { color: [.4,.5,.5], radius: 1 },
                                        Se: { color: [1,.6,0], radius: 1 } };
    const r = loadPage({ navData: nav });
    const rb2 = r.window.document.getElementById("a-cmode-blend");
    rb2.checked = true; rb2.dispatchEvent(new r.window.Event("change"));
    assert.equal(effMarker(r.state, 0).color[0], "rgb(159,137,80)", "worked-example blend must be rgb(159,137,80)");
  }
});

test("35: colour-mode selector — default species (2 sp.), blend/plain switch, theme re-flip, exclusivity, 1-sp. disable", () => {
  const { window, state, navData, threw } = loadPage();   // base synth = Sn/Se/Se = 2 species
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const nB = navData.band.frequency[0].length;
  const CHAR = navData.band.character;
  const rb = doc.getElementById("a-cmode-blend");
  const rs = doc.getElementById("a-cmode-species");
  const rp = doc.getElementById("a-cmode-plain");

  // --- default-on-load: 2 species -> "species" checked, branch-0 effective
  //     marker is the baked char scalar on a colorscale; CBAR shows the colorbar.
  assert.equal(rs.checked, true, "2-species default must be 'species'");
  assert.equal(rb.checked, false, "blend must not be the 2-species default");
  const sp0 = effMarker(state, 0);
  assert.ok(Array.isArray(sp0.color), "species marker.color must be the per-point char array");
  assert.equal(sp0.color[0], CHAR[0][0], "species marker[0] must be the baked char scalar");
  assert.ok(sp0.colorscale, "species marker must carry a colorscale");
  const cbar = effMarker(state, nB);
  assert.equal(cbar.showscale, true, "species CBAR carrier must show the colorbar");
  assert.ok(cbar.colorbar && cbar.colorbar.title && /character/.test(cbar.colorbar.title.text),
    "species colorbar title.text must be CHAR_TITLE (object title, fix B)");

  // --- switch to Atom blend: branch-0 effective marker is the RGB blend; CBAR hidden
  rb.checked = true; rb.dispatchEvent(new window.Event("change"));
  const bl0 = effMarker(state, 0);
  assert.ok(Array.isArray(bl0.color) && /^rgb\(/.test(bl0.color[0]), "blend marker must be an rgb() array");
  assert.ok(!bl0.colorscale, "blend marker must drop the colorscale");
  assert.equal(effMarker(state, nB).showscale, false, "blend CBAR must hide the colorbar");

  // --- switch to Plain: branch-0 effective marker === themeColors().fg (light #1a1a1a), no colorscale
  rp.checked = true; rp.dispatchEvent(new window.Event("change"));
  assert.equal(effMarker(state, 0).color, "#1a1a1a", "Plain marker must be the light theme fg");
  assert.ok(!effMarker(state, 0).colorscale, "Plain marker must have no colorscale");
  assert.equal(effMarker(state, nB).showscale, false, "Plain CBAR must hide the colorbar");

  // --- Plain re-flips on theme toggle: #1a1a1a -> #e8e8e8
  doc.getElementById("theme-toggle").dispatchEvent(new window.Event("click"));
  assert.equal(effMarker(state, 0).color, "#e8e8e8", "Plain bands must re-flip to the dark theme fg");
  doc.getElementById("theme-toggle").dispatchEvent(new window.Event("click"));   // back to light

  // --- mutual exclusivity: selecting one radio leaves the others unchecked
  rs.checked = true; rs.dispatchEvent(new window.Event("change"));
  assert.ok(rs.checked && !rb.checked && !rp.checked, "radios must be mutually exclusive");

  // --- 1-species: blend & species disabled, plain checked + active (marker == fg)
  const nav1 = synthNavData();
  nav1.meta.material.symbols = ["C", "C", "C"];
  nav1.meta.material.species_style = { C: { color: [0.3, 0.3, 0.3], radius: 1 } };
  const r1 = loadPage({ navData: nav1 });
  const d1 = r1.window.document;
  assert.equal(d1.getElementById("a-cmode-blend").disabled, true, "1-species: blend radio disabled");
  assert.equal(d1.getElementById("a-cmode-species").disabled, true, "1-species: species radio disabled");
  assert.equal(d1.getElementById("a-cmode-plain").checked, true, "1-species: plain checked");
  assert.equal(effMarker(r1.state, 0).color, "#1a1a1a", "1-species: bands are the plain theme fg");
});

test("36: 0.6.37 dim=2 regression guard — species default doesn't abort the on-load UI; theme-toggle title restyle is safe", () => {
  // base synth = 2 species -> defaults to "species" (the regression trigger).
  // Under the stricter mock (which throws on a string-title sub-property
  // restyle), the FULL on-load sequence must complete: jump buttons built,
  // default selection fired, and a theme toggle in species mode must not throw.
  const { window, state, navData, threw } = loadPage();
  assert.equal(threw, null, "on-load sequence must not throw (regression: it did): " + (threw && threw.stack));
  const doc = window.document;
  // (i) the jump-button builder ran (downstream of the would-be abort)
  const jb = doc.getElementById("mode-jump-buttons");
  assert.ok(jb && jb.children.length > 0, "jump buttons must be present (builder ran past the would-be abort)");
  // (ii) the default selection fired on load — a-info is the mode line, not the prompt
  assert.match(doc.getElementById("a-info").textContent, /ω = .* THz/,
    "a mode must be selected on load (a-info shows the ω readout, not the click prompt)");
  // (iii) the species default is active after the FULL sequence (end-of-IIFE apply)
  assert.equal(doc.getElementById("a-cmode-species").checked, true, "species must be the 2-species default");
  const cbar = effMarker(state, navData.band.frequency[0].length);
  assert.equal(cbar.showscale, true, "species colorbar must be live after the on-load sequence");
  assert.equal(typeof cbar.colorbar.title, "object", "colorbar title must be an OBJECT (fix B), not a string");
  // (iv) toggling the theme in species mode does NOT throw under the strict mock
  //      (the title.font.color restyle now lands on an object title)
  assert.doesNotThrow(() => doc.getElementById("theme-toggle").dispatchEvent(new window.Event("click")),
    "theme toggle in species mode must not throw (object title makes the font restyle safe)");
});

test("37: Plain bands are thinner (marker.size ~3) while blend/species keep the default size", () => {
  const { window, state, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // species default -> size 5
  assert.equal(effMarker(state, 0).size, 5, "species markers keep the construction default size 5");
  // blend -> size 5
  doc.getElementById("a-cmode-blend").checked = true;
  doc.getElementById("a-cmode-blend").dispatchEvent(new window.Event("change"));
  assert.equal(effMarker(state, 0).size, 5, "blend markers keep the construction default size 5");
  // plain -> thinner
  doc.getElementById("a-cmode-plain").checked = true;
  doc.getElementById("a-cmode-plain").dispatchEvent(new window.Event("change"));
  assert.equal(effMarker(state, 0).size, 3, "Plain markers are thinner (size 3)");
});

test("38: labeled cell-vectors matrix + Lattice-parameters tables + Atom-coordinates (frac/cart toggle)", () => {
  // orthogonal lattice a=2, b=3, c=4 -> A/B/C exact, α=β=γ=90.00
  const nav = synthNavData();
  nav.meta.material.lattice = [[2, 0, 0], [0, 3, 0], [0, 0, 4]];
  const { window, navData, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const M = navData.meta.material;
  // titles renamed
  const titles = Array.from(doc.querySelectorAll(".rail-section-title")).map(t => t.textContent.replace(/\s+/g, " ").trim());
  assert.ok(titles.some(t => /Primitive cell vectors \(Å\)/.test(t)), "cell-vectors title renamed");
  assert.ok(titles.some(t => /Lattice parameters/.test(t)), "Lattice parameters title present");
  assert.ok(titles.some(t => /Atom coordinates/.test(t)), "Atom coordinates title present");
  // matrix: 4-col grid with header [v,x,y,z] + v₁/v₂/v₃ row labels. The 9 numeric
  // values are the non-header, non-rowlabel cells.
  const allCells = Array.from(doc.querySelectorAll("#lattice-matrix span"));
  const nums = allCells.filter(s => !s.classList.contains("lm-head") && !s.classList.contains("lm-rowlabel"))
                       .map(s => s.textContent);
  assert.equal(nums.length, 9, "matrix must have 9 numeric cells past the labels");
  let k = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    assert.equal(nums[k++], M.lattice[i][j].toFixed(4), "matrix cell mismatch at " + i + "," + j);
  // header + row labels present
  assert.ok(allCells.some(s => s.classList.contains("lm-head") && s.textContent === "x"), "matrix header x");
  assert.ok(allCells.some(s => s.classList.contains("lm-rowlabel") && /v\u2081/.test(s.textContent)), "v₁ row label");
  // params: read from #lattice-abc and #lattice-angles (numeric cells past the heads)
  const abcNums = Array.from(doc.querySelectorAll("#lattice-abc span"))
    .filter(s => !s.classList.contains("lp-head") && !s.classList.contains("lp-unit")).map(s => s.textContent);
  assert.deepEqual(abcNums, ["2.0000", "3.0000", "4.0000"], "abc lengths wrong: " + abcNums.join());
  const angNums = Array.from(doc.querySelectorAll("#lattice-angles span"))
    .filter(s => !s.classList.contains("lp-head") && !s.classList.contains("lp-unit")).map(s => s.textContent);
  assert.deepEqual(angNums, ["90.00", "90.00", "90.00"], "orthogonal angles must be 90.00: " + angNums.join());
  // positions table: Na rows, NO swatch, symbol + x/y/z, fractional default
  const rows = Array.from(doc.querySelectorAll("#atom-pos-table .atom-pos-row"));
  assert.equal(rows.length, M.symbols.length, "one position row per basis atom");
  rows.forEach((row, i) => {
    assert.equal(row.querySelector(".atom-pos-swatch"), null, "row " + i + " must NOT have a swatch (0.6.39 C)");
    assert.match(row.querySelector(".atom-pos-sym").textContent, new RegExp(M.symbols[i]), "row " + i + " symbol");
    const vals = Array.from(row.querySelectorAll("span")).slice(-3).map(s => s.textContent);
    assert.equal(vals[0], M.fractional_coords[i][0].toFixed(4), "row " + i + " frac x");
    assert.equal(vals[2], M.fractional_coords[i][2].toFixed(4), "row " + i + " frac z");
  });
  // cart toggle: button reads "Fractional" -> click -> "Cartesian (Å)" + frac·lattice values
  const btn = doc.getElementById("pos-units");
  assert.equal(btn.textContent, "Fractional", "button starts on Fractional");
  btn.dispatchEvent(new window.Event("click"));
  assert.equal(btn.textContent, "Cartesian (Å)", "button reads Cartesian (Å) after toggle");
  const r0 = doc.querySelectorAll("#atom-pos-table .atom-pos-row")[0];
  const cx = Array.from(r0.querySelectorAll("span")).slice(-3).map(s => s.textContent);
  const exp0 = [0, 0, 0];
  for (let j = 0; j < 3; j++) for (let c = 0; c < 3; c++) exp0[c] += M.fractional_coords[0][j] * M.lattice[j][c];
  assert.equal(cx[0], exp0[0].toFixed(4), "cart x mismatch");
  assert.equal(cx[2], exp0[2].toFixed(4), "cart z mismatch");
  btn.dispatchEvent(new window.Event("click"));
  assert.equal(btn.textContent, "Fractional", "button returns to Fractional");
});

test("38b: non-orthogonal lattice -> at least one angle != 90 (negative control on the angle calc)", () => {
  // the base synth lattice is hexagonal-ish (gamma=120) -> not all 90
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const angNums = Array.from(window.document.querySelectorAll("#lattice-angles span"))
    .filter(s => !s.classList.contains("lp-head") && !s.classList.contains("lp-unit"))
    .map(s => parseFloat(s.textContent));
  assert.equal(angNums.length, 3, "three angles must be shown");
  assert.ok(angNums.some(a => Math.abs(a - 90) > 0.01), "a non-orthogonal lattice must show an angle != 90: " + angNums.join());
});

test("39: Reset atoms colors — clears overrides and restores each picker to its baked hex", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // the reset button exists in the Atoms section
  const btn = doc.getElementById("a-colreset");
  assert.ok(btn, "Reset atoms colors button must exist");
  assert.equal(btn.closest(".rail-section").querySelector(".rail-section-title").textContent, "Atoms",
    "reset button must live in the Atoms section");
  // legend pickers (one color input per unique species)
  const pickers = Array.from(doc.querySelectorAll("#atoms-legend input[type=color]"));
  assert.ok(pickers.length >= 2, "expected a colour picker per species (Sn, Se)");
  // baked hex from the synth species_style (mirror rgb01ToHex)
  const STY = navData.meta.material.species_style;
  const hexOf = c => "#" + c.map(x =>
    Math.round(Math.max(0, Math.min(1, x)) * 255).toString(16).padStart(2, "0")).join("");
  const p0 = pickers[0];
  const sym0 = p0.title.replace(/^color:\s*/, "");
  const baked0 = hexOf(STY[sym0].color);
  assert.equal(p0.value, baked0, "picker starts at the baked species hex");
  // change the picker (fires the input handler -> sets colorOverride[sym0])
  p0.value = "#123456";
  p0.dispatchEvent(new window.Event("input"));
  assert.equal(p0.value, "#123456", "picker reflects the changed value before reset");
  // reset -> picker value restored to the baked hex
  btn.dispatchEvent(new window.Event("click"));
  assert.equal(p0.value, baked0, "Reset atoms colors must restore the picker to its baked hex");
});

test("40: Atom size — slider + reset restore value to 1.0 and label to ×1.00", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // slider + reset exist in the Atoms section
  const sl = doc.getElementById("a-atomscale");
  const btn = doc.getElementById("a-atomscale-reset");
  const val = doc.getElementById("a-atomscale-val");
  assert.ok(sl && btn && val, "atom-size slider, reset, and readout must exist");
  assert.equal(sl.closest(".rail-section").querySelector(".rail-section-title").textContent, "Atoms",
    "atom-size controls must live in the Atoms section");
  assert.equal(sl.value, "1.0", "atom-size slider defaults to 1.0");
  // change the slider -> label updates to ×N.NN
  sl.value = "2.5"; sl.dispatchEvent(new window.Event("input"));
  assert.equal(val.textContent, "×2.50", "label tracks the slider value");
  // reset -> slider back to 1.0, label back to ×1.00
  btn.dispatchEvent(new window.Event("click"));
  assert.equal(sl.value, "1.0", "Reset atom size must restore the slider to 1.0");
  assert.equal(val.textContent, "×1.00", "Reset atom size must restore the label to ×1.00");
});

test("41: per-species radius inputs default to 0.34 × Cordero in Å + Reset restores the Å defaults", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const style = navData.meta.material.species_style;
  const defA = sym => (Math.round(0.34 * style[sym].radius * 100) / 100).toFixed(2);
  // a radius number input per species, each defaulting to its Å value
  const radInputs = Array.from(doc.querySelectorAll("#atoms-legend input[type=number]"));
  assert.ok(radInputs.length >= 2, "expected a radius input per species (Sn, Se)");
  const bySym = sym => radInputs.find(r => r.title === "display radius (Å): " + sym);
  assert.equal(bySym("Sn").value, defA("Sn"), "Sn defaults to 0.34 × Cordero = 0.47 Å");
  assert.equal(bySym("Se").value, defA("Se"), "Se defaults to 0.34 × Cordero = 0.41 Å");
  // the reset-all button exists in the Atoms section
  const btn = doc.getElementById("a-radreset");
  assert.ok(btn, "Reset atoms radii button must exist");
  assert.equal(btn.closest(".rail-section").querySelector(".rail-section-title").textContent, "Atoms",
    "Reset atoms radii must live in the Atoms section");
  // change two species' radii in Å (fire input handlers -> set radiusOverride)
  bySym("Sn").value = "0.80"; bySym("Sn").dispatchEvent(new window.Event("input"));
  bySym("Se").value = "0.20"; bySym("Se").dispatchEvent(new window.Event("input"));
  assert.equal(bySym("Sn").value, "0.80", "first radius reflects the change");
  assert.equal(bySym("Se").value, "0.20", "second radius reflects the change");
  // reset-all -> every radius input back to its Å default
  btn.dispatchEvent(new window.Event("click"));
  assert.equal(bySym("Sn").value, defA("Sn"), "Reset restores Sn to its 0.34 × Cordero Å default");
  assert.equal(bySym("Se").value, defA("Se"), "Reset restores Se to its 0.34 × Cordero Å default");
});

test("42: bonds-legend lists exactly the distinct canonical species-pairs in bonds[]", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const symbols = navData.meta.material.symbols;
  const slots = window.__atoms.getSlots();
  const list = window.__atoms.bonds().list;
  // independent reference: walk the bond list, map endpoints via
  // symbols[slots.j[...]], canonicalize with sort + EN-DASH (U+2013), dedupe.
  const ref = new Set();
  list.forEach(([i, jj]) => {
    const s1 = symbols[slots.j[i]], s2 = symbols[slots.j[jj]];
    ref.add([s1, s2].sort().join("\u2013"));
  });
  assert.ok(ref.size > 0, "reference pair set must be non-empty (the synth has bonds)");
  // 0.6.54: the pair key is now its own .bonds-legend-pair grid cell (the per-pair
  // <label> was replaced by the 5-column grid restructure).
  const rendered = Array.from(window.document.querySelectorAll("#bonds-legend .bonds-legend-pair"))
    .map(c => c.textContent);
  // exactly the distinct pairs, no dupes
  assert.equal(rendered.length, ref.size, "bonds-legend must render exactly the deduped pair count");
  assert.deepEqual(new Set(rendered), ref, "bonds-legend labels must equal the canonical pair set");
  // every label uses the EN-DASH, not an ASCII hyphen
  rendered.forEach(t => assert.ok(/\u2013/.test(t) && !/-/.test(t),
    "pair label must use EN-DASH U+2013, not ASCII '-': " + JSON.stringify(t)));
});

test("43: per-pair bond radius is ABSOLUTE \u00c5 \u2014 edited pairs pin, unedited pairs track the global \u00c5 default", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const slots = window.__atoms.getSlots();
  const list = window.__atoms.bonds().list;
  const keyOf = ([i, jj]) => [symbols[slots.j[i]], symbols[slots.j[jj]]].sort().join("\u2013");
  // independent per-bond key reference
  const bondKeys = list.map(keyOf);
  const distinct = [...new Set(bondKeys)].sort();
  assert.ok(list.length >= 1 && distinct.length >= 1, "synth must have >=1 bond / >=1 distinct pair");
  // 0.6.76: the per-pair input is an ABSOLUTE cylinder radius in \u00c5 (was a
  // \u00d7-multiplier); it shows the global \u00c5 default (0.15) until edited.
  const pairCells = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-pair"));
  const radInpFor = key => Array.from(doc.querySelectorAll("#bonds-legend input"))
    .find(i => i.title === "bond cylinder radius (\u00c5): " + key);
  assert.equal(pairCells.length, distinct.length, "one legend cell per distinct pair");
  distinct.forEach(key => {
    const inp = radInpFor(key);
    assert.ok(inp, "each pair must have a radius number input");
    assert.equal(inp.value, "0.15", "pair radius input defaults to the global \u00c5 default");
  });
  // baseline: every effective radius equals the global \u00c5 default (0.15)
  const bondR0 = 0.15;
  let rad = window.__atoms.bonds().rad;
  bondKeys.forEach((k, b) => assert.ok(Math.abs(rad[b] - bondR0) < 1e-9,
    "baseline bond radius must equal the global \u00c5 default"));
  // pin the FIRST distinct pair at 0.5 \u00c5 via its input
  const targetKey = distinct[0];
  const targetInp = radInpFor(targetKey);
  targetInp.value = "0.5"; targetInp.dispatchEvent(new window.Event("input"));
  rad = window.__atoms.bonds().rad;
  bondKeys.forEach((k, b) => {
    const want = (k === targetKey ? 0.5 : bondR0);
    assert.ok(Math.abs(rad[b] - want) < 1e-9,
      "bond " + b + " (" + k + ") radius must be " + want + " \u00c5");
  });
  // change the global \u00c5 default -> UNEDITED pairs follow it (and their inputs
  // re-display it); the pinned pair keeps its absolute 0.5 \u00c5
  const g = doc.getElementById("a-bondr");
  g.value = "0.30"; g.dispatchEvent(new window.Event("change"));
  const newBondR = 0.30;
  rad = window.__atoms.bonds().rad;
  bondKeys.forEach((k, b) => {
    const want = (k === targetKey ? 0.5 : newBondR);
    assert.ok(Math.abs(rad[b] - want) < 1e-9,
      "after global change, bond " + b + " (" + k + ") must be " + want + " \u00c5");
  });
  distinct.forEach(key => {
    assert.equal(radInpFor(key).value, key === targetKey ? "0.5" : "0.3",
      "input display: pinned pair keeps its \u00c5; unedited pairs show the new default");
  });
  // EN-DASH check on the pair cells
  pairCells.forEach(c => assert.ok(/\u2013/.test(c.textContent) && !/-/.test(c.textContent),
    "pair label must use EN-DASH U+2013"));
});

test("44: per-pair bond colour — split default, custom override, survives species change", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const slots = window.__atoms.getSlots();
  const list = window.__atoms.bonds().list;
  const keyOf = ([i, jj]) => [symbols[slots.j[i]], symbols[slots.j[jj]]].sort().join("\u2013");
  const bondKeys = list.map(keyOf);
  // a HETERO pair (two different symbols) must exist for the split test
  const heteroKey = bondKeys.find(k => { const [a, b] = k.split("\u2013"); return a !== b; });
  assert.ok(heteroKey, "synth must have >=1 hetero bond pair (two different symbols)");
  // each pair row has a checkbox (default unchecked) and a colour input
  // 0.6.54: per-pair controls are grid cells now; find colour controls by key-suffixed title.
  const pairCells = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-pair"));
  const chkFor = key => Array.from(doc.querySelectorAll("#bonds-legend input"))
    .find(i => i.title === "use custom color: " + key);
  const colInpFor = key => Array.from(doc.querySelectorAll("#bonds-legend input"))
    .find(i => i.title === "custom bond color: " + key);
  pairCells.forEach(c => {
    const key = c.textContent;
    assert.ok(chkFor(key), "each pair must have a checkbox");
    assert.ok(colInpFor(key), "each pair must have a colour input");
    assert.equal(chkFor(key).checked, false, "checkbox defaults unchecked (split)");
  });
  const eq = (c, v) => c && Math.abs(c[0]-v[0]) < 1e-6 && Math.abs(c[1]-v[1]) < 1e-6 && Math.abs(c[2]-v[2]) < 1e-6;
  // SPLIT default: a hetero pair's two halves differ (two endpoint colours)
  let colors = window.__atoms.bonds().colors;
  const heteroBonds = bondKeys.map((k, b) => k === heteroKey ? b : -1).filter(b => b >= 0);
  const hb = heteroBonds[0];
  assert.ok(!eq(colors[2*hb], colors[2*hb+1]), "split: a hetero pair's two halves must differ");
  // pick a DIFFERENT pair to confirm it stays untouched when we customise heteroKey
  const otherBonds = bondKeys.map((k, b) => k !== heteroKey ? b : -1).filter(b => b >= 0);
  const ob = otherBonds.length ? otherBonds[0] : null;
  const obBefore = ob != null ? [colors[2*ob], colors[2*ob+1]] : null;
  // set heteroKey custom -> both halves red
  const colInp = colInpFor(heteroKey);
  colInp.value = "#ff0000"; colInp.dispatchEvent(new window.Event("input"));
  colors = window.__atoms.bonds().colors;
  heteroBonds.forEach(b => {
    assert.ok(eq(colors[2*b], [1,0,0]) && eq(colors[2*b+1], [1,0,0]),
      "custom: both halves of every hetero-pair bond must be red");
  });
  if (ob != null) {
    assert.ok(eq(colors[2*ob], obBefore[0]) && eq(colors[2*ob+1], obBefore[1]),
      "a different pair must be unchanged when heteroKey is customised");
  }
  // uncheck -> back to split (halves differ again)
  const chk = chkFor(heteroKey);
  chk.checked = false; chk.dispatchEvent(new window.Event("change"));
  colors = window.__atoms.bonds().colors;
  assert.ok(!eq(colors[2*hb], colors[2*hb+1]), "uncheck: hetero pair returns to split (halves differ)");
  // CUSTOM SURVIVES SPECIES: re-enable custom, then change an endpoint species colour
  colInp.value = "#00ff00"; colInp.dispatchEvent(new window.Event("input"));   // heteroKey custom green
  const [sa] = heteroKey.split("\u2013");
  // find sa's atoms-legend colour picker and change it
  const atomPicker = Array.from(doc.querySelectorAll("#atoms-legend input[type=color]"))
    .find(p => p.title === "color: " + sa);
  assert.ok(atomPicker, "atoms-legend must have a picker for " + sa);
  atomPicker.value = "#0000ff"; atomPicker.dispatchEvent(new window.Event("input"));   // recolorSpecies(sa)
  colors = window.__atoms.bonds().colors;
  heteroBonds.forEach(b => {
    assert.ok(eq(colors[2*b], [0,1,0]) && eq(colors[2*b+1], [0,1,0]),
      "custom pair must KEEP its custom colour through a species change");
  });
  // a SPLIT pair containing sa did change to the new species colour
  const splitWithSa = bondKeys.map((k, b) => (k !== heteroKey && k.split("\u2013").includes(sa)) ? b : -1)
                              .filter(b => b >= 0);
  if (splitWithSa.length) {
    const sb = splitWithSa[0];
    const saSlotIsFirst = symbols[slots.j[list[sb][0]]] === sa;
    const half = colors[saSlotIsFirst ? 2*sb : 2*sb+1];
    assert.ok(eq(half, [0,0,1]), "a split pair's " + sa + " half must pick up the new species colour");
  }
  // EN-DASH on labels
  pairCells.forEach(c => assert.ok(/\u2013/.test(c.textContent) && !/-/.test(c.textContent), "label uses EN-DASH"));
});

test("45: live bond breaking — opt-in toggle, per-pair max distance, stateless re-form (visibility-only over the rest set)", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const style = navData.meta.material.species_style;
  const slots = window.__atoms.getSlots();
  const list = window.__atoms.bonds().list;
  const keyOf = ([i, jj]) => [symbols[slots.j[i]], symbols[slots.j[jj]]].sort().join("\u2013");
  const bondKeys = list.map(keyOf);
  const distinct = [...new Set(bondKeys)].sort();
  assert.ok(list.length >= 1 && distinct.length >= 1, "synth must have >=1 bond / >=1 distinct pair");

  const BOND_TOL = 1.2;   // must match index.html's const

  // (1) toggle exists and defaults UNCHECKED (breaking OFF = the static 0.6.47 set)
  const tog = doc.getElementById("a-bondbreak");
  assert.ok(tog, "#a-bondbreak toggle must exist");
  assert.equal(tog.checked, false, "#a-bondbreak must default unchecked (breaking OFF)");
  assert.equal(window.__atoms.bonds().breakOn, false, "bonds().breakOn must default false");

  // (2) each pair row has a distance input defaulting to the pair's TYPE max, which
  //     at seed equals the COVALENT cutoff BOND_TOL*(rcov_a+rcov_b) — computed
  //     independently from the navData species_style radii here.
  // 0.6.54: per-pair controls are grid cells now; find inputs by their key-suffixed title.
  const pairCells = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-pair"));
  assert.equal(pairCells.length, distinct.length, "one legend cell per distinct pair");
  const inpByTitle = title => Array.from(doc.querySelectorAll("#bonds-legend input")).find(i => i.title === title);
  pairCells.forEach(c => {
    const key = c.textContent;
    const mx = inpByTitle("bond distance (\u00c5): " + key);
    assert.ok(mx, "each pair must have a bond-distance number input");
    const [sa, sb] = key.split("\u2013");
    const want = (BOND_TOL * (style[sa].radius + style[sb].radius)).toFixed(2);
    assert.equal(mx.value, want, "distance default must equal the covalent type cutoff for " + key);
  });
  // the per-pair radius input still defaults to the global Å default (0.6.76)
  pairCells.forEach(c => assert.equal(inpByTitle("bond cylinder radius (Å): " + c.textContent).value, "0.15",
    "radius input still defaults to the 0.15 Å global default"));

  // (3) breaking OFF (default): every bond visible (static set); topology count unchanged
  window.__atoms.applyFrame();
  let vis = window.__atoms.bonds().visible;
  assert.equal(window.__atoms.bonds().list.length, list.length, "bond count (topology) unchanged by F2d");
  bondKeys.forEach((k, b) => assert.equal(vis[b], true, "OFF: bond " + b + " (" + k + ") must be visible"));

  // (4) F2d-2b: bondMax[b] (the live-break cutoff) is now SOURCED from the pair's
  //     TYPE max (recomputeBondMax reads typeMaxOf); at seed it equals the covalent
  //     cutoff. Because the type max ALSO gates detection, a sub-rest type max REMOVES
  //     the bond (covered by assertion 47) rather than hiding it — so here we prove the
  //     SOURCE, then exercise the UNCHANGED per-frame break by driving bondMax[b]
  //     directly (bonds[] untouched; applyFrame reads bondMax[b]).
  const targetKey = distinct[0];
  const tgtBonds = bondKeys.map((k, b) => k === targetKey ? b : -1).filter(b => b >= 0);
  const othBonds = bondKeys.map((k, b) => k !== targetKey ? b : -1).filter(b => b >= 0);
  const [ta, tb] = targetKey.split("\u2013");
  const typeCut = BOND_TOL * (style[ta].radius + style[tb].radius);
  tgtBonds.forEach(b => assert.ok(Math.abs(window.__atoms.bonds().max[b] - typeCut) < 1e-9,
    "SOURCE: bondMax[b] must come from the type max (= covalent default) for " + targetKey));

  const scaleOf = b => window.__atoms.bonds().mesh.scales[2*b];
  const bm = window.__atoms.bonds().max;          // the per-frame cutoff array (live ref)

  // drive the target bonds' cutoff below rest + breaking ON -> hide (present-but-hidden)
  tgtBonds.forEach(b => { bm[b] = 0.1; });        // below any chemical bond length
  tog.checked = true; tog.dispatchEvent(new window.Event("change"));
  assert.equal(window.__atoms.bonds().breakOn, true, "breakOn true after check");
  window.__atoms.applyFrame();
  vis = window.__atoms.bonds().visible;
  tgtBonds.forEach(b => {
    assert.equal(vis[b], false, "ON + tiny bondMax: target bond " + b + " must be BROKEN (visible=false)");
    const s = scaleOf(b);
    assert.ok(s.x === 0 && s.y === 0 && s.z === 0, "broken bond " + b + " must be ZERO-SCALED in the mesh");
  });
  othBonds.forEach(b => {
    assert.equal(vis[b], true, "other pairs keep their (covalent) cutoff -> stay visible");
    assert.ok(scaleOf(b).x > 0 && scaleOf(b).y > 0, "unbroken bond " + b + " must remain drawn (scale > 0)");
  });

  // raise the SAME bonds' cutoff above rest -> re-form (still ON, same phi)
  tgtBonds.forEach(b => { bm[b] = 9; });
  window.__atoms.applyFrame();
  vis = window.__atoms.bonds().visible;
  tgtBonds.forEach(b => {
    assert.equal(vis[b], true, "raised cutoff: target bond " + b + " must RE-FORM (visible=true)");
    assert.ok(scaleOf(b).x > 0 && scaleOf(b).y > 0, "re-formed bond " + b + " must be drawn (scale > 0)");
  });

  // (5) tiny cutoff again but UNCHECK breaking -> off = always shown
  tgtBonds.forEach(b => { bm[b] = 0.1; });
  tog.checked = false; tog.dispatchEvent(new window.Event("change"));
  window.__atoms.applyFrame();
  vis = window.__atoms.bonds().visible;
  tgtBonds.forEach(b => assert.equal(vis[b], true, "OFF (even with tiny cutoff): bond " + b + " must be visible"));

  // EN-DASH on labels (silent-mismatch guard, like 42/43/44)
  pairCells.forEach(c => assert.ok(/\u2013/.test(c.textContent) && !/-/.test(c.textContent), "label uses EN-DASH U+2013"));
});

test("46: bond detection is bondTypes-driven — default table reproduces auto-detection; zeroing a type removes its bonds", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const style = navData.meta.material.species_style;
  const BOND_TOL = 1.2;                       // must match index.html's const
  const rc = s => ((style[s] || {}).radius) || 1.0;
  const canon = (s1, s2) => [s1, s2].sort().join("\u2013");

  // (1) bonds().types is seeded with EVERY unordered species-pair (a<=b sorted),
  //     each max === BOND_TOL*(rc(a)+rc(b)) — the SAME covalent cutoff the old
  //     auto-detection used, computed independently from navData here.
  const uniq = [...new Set(symbols)];
  const expectedTypes = [];
  for (let x = 0; x < uniq.length; x++)
    for (let y = x; y < uniq.length; y++) {
      const pair = [uniq[x], uniq[y]].sort();
      expectedTypes.push({ a: pair[0], b: pair[1], max: BOND_TOL * (rc(pair[0]) + rc(pair[1])) });
    }
  const types = window.__atoms.bonds().types;
  assert.equal(types.length, expectedTypes.length, "bondTypes must seed one row per unordered species-pair");
  const keyOfType = t => canon(t.a, t.b);
  const gotMap = new Map(types.map(t => [keyOfType(t), t]));
  expectedTypes.forEach(e => {
    const g = gotMap.get(keyOfType(e));
    assert.ok(g, "missing seeded type " + keyOfType(e));
    assert.ok(g.a <= g.b, "type pair must be stored a<=b: " + keyOfType(e));
    assert.ok(Math.abs(g.max - e.max) < 1e-12, "seeded max must equal covalent cutoff for " + keyOfType(e));
  });

  // (2) bonds().list is byte-identical to the pre-refactor auto-detection.
  //     Independent reference: brute-force the SAME criterion (d^2 <= cut^2,
  //     cut = BOND_TOL*(rcov_i+rcov_j)) over the rest positions, like assertion 18.
  const slots0 = window.__atoms.getSlots();
  const r0 = window.__atoms.getR0c();
  const rcSlot = slots0.j.map(a => rc(symbols[a]));
  let expectedCount = 0; const expectedPairs = new Set();
  for (let i = 0; i < r0.length; i++) for (let jj = i + 1; jj < r0.length; jj++) {
    const dx = r0[i][0]-r0[jj][0], dy = r0[i][1]-r0[jj][1], dz = r0[i][2]-r0[jj][2];
    const cut = BOND_TOL * (rcSlot[i] + rcSlot[jj]);
    if (dx*dx + dy*dy + dz*dz <= cut*cut) {
      expectedCount++; expectedPairs.add(canon(symbols[slots0.j[i]], symbols[slots0.j[jj]]));
    }
  }
  assert.ok(expectedCount > 0, "synth must produce bonds (vacuity guard)");
  const list = window.__atoms.bonds().list;
  assert.equal(list.length, expectedCount, "type-driven default count must equal auto-detection count");
  const gotPairs = new Set(list.map(([i, jj]) => canon(symbols[slots0.j[i]], symbols[slots0.j[jj]])));
  assert.deepEqual(gotPairs, expectedPairs, "canonical pair set must be unchanged by the refactor");

  // (3) TYPE-DRIVEN proof: zero a PRESENT pair's type max, retile -> that pair's
  //     bonds vanish; restore -> they return. Proves detection reads the table,
  //     not a fixed cutoff. (This sub-assertion is what the negative control bites.)
  const targetKey = [...gotPairs][0];
  const tgt = gotMap.get(targetKey);                       // live bondTypes element
  const t = ["a-t1", "a-t2", "a-t3"].map(id => parseInt(doc.getElementById(id).value) || 1);
  const countTarget = () => {
    const sj = window.__atoms.getSlots().j;
    return window.__atoms.bonds().list
      .filter(([i, jj]) => canon(symbols[sj[i]], symbols[sj[jj]]) === targetKey).length;
  };
  const beforeTarget = countTarget();
  const beforeTotal = window.__atoms.bonds().list.length;
  assert.ok(beforeTarget > 0, "target pair must have bonds to remove");
  const savedMax = tgt.max;
  tgt.max = 0;                                              // zero the live type cutoff
  window.__atoms.buildTiling(t[0], t[1], t[2]);
  assert.equal(countTarget(), 0, "zeroing the type max must remove that pair's bonds (type-driven)");
  assert.ok(window.__atoms.bonds().list.length < beforeTotal, "total bond count must drop");
  tgt.max = savedMax;                                       // restore
  window.__atoms.buildTiling(t[0], t[1], t[2]);
  assert.equal(countTarget(), beforeTarget, "restoring the max must return that pair's bonds");
});

test("47: F2d-2b — distance drives detection (single source), per-type Remove, Reset-bonds restores defaults", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const A = window.__atoms;
  const canon = (s1, s2) => [s1, s2].sort().join("\u2013");
  const keyOfType = t => canon(t.a, t.b);

  // live re-readers (buildTiling rebuilds slots / list / legend each time)
  const sj = () => A.getSlots().j;
  const listKeys = () => A.bonds().list.map(([i, jj]) => canon(symbols[sj()[i]], symbols[sj()[jj]]));
  const typeKeys = () => A.bonds().types.map(keyOfType);
  const countOf = key => listKeys().filter(k => k === key).length;
  const idxOf = key => listKeys().findIndex(k => k === key);
  // 0.6.54: per-pair controls are grid cells; locate by key-suffixed title.
  const hasPair = key => Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-pair")).some(c => c.textContent === key);
  const inpByTitle = title => Array.from(doc.querySelectorAll("#bonds-legend input")).find(i => i.title === title);
  const rmBtnFor = key => Array.from(doc.querySelectorAll("#bonds-legend button")).find(b => b.title === "remove bond type: " + key);

  const fullDefault = typeKeys().slice().sort();             // seeded default type set
  const present0 = [...new Set(listKeys())];
  assert.ok(present0.length >= 1, "synth must have >=1 bonded pair");
  const targetKey = present0[0];
  const before = countOf(targetKey);
  assert.ok(before > 0, "target pair must have bonds at default");

  // (i) DISTANCE is the single source that drives DETECTION. Lower the present pair's
  //     distance input below its rest length + dispatch "change" -> the rebuild
  //     re-detects and the pair's bonds (and its legend row) vanish. Raising the type
  //     max back + rebuild restores them. (The legend row is gone once the bonds
  //     vanish — buildBondsLegend iterates bonds[] — so we raise via the table, which
  //     IS the same single source the input writes to.)
  const di = inpByTitle("bond distance (\u00c5): " + targetKey);
  const t0 = A.bonds().types.find(x => keyOfType(x) === targetKey);
  const savedMax = t0.max;
  di.value = "0.1"; di.dispatchEvent(new window.Event("change"));
  assert.equal(countOf(targetKey), 0, "lowering the distance below rest removes the pair's bonds (re-detect)");
  assert.ok(!hasPair(targetKey), "the legend row disappears when the pair has no bonds");
  t0.max = savedMax; A.buildTiling();
  assert.equal(countOf(targetKey), before, "raising the type max back restores the pair's bonds");

  // (ii)+(iii) prep: edit the target pair's RADIUS up (factor 2.5) while it's present,
  //     so Reset has a non-default radius to clear. (The synth has a single bonded
  //     pair, so we operate on the target throughout rather than a second pair.)
  const radDefault = A.bonds().rad[idxOf(targetKey)];        // the global Å default (per-pair Å unset)
  const radInp = inpByTitle("bond cylinder radius (Å): " + targetKey);   // per-pair ABSOLUTE Å (0.6.76)
  radInp.value = "0.5"; radInp.dispatchEvent(new window.Event("input"));
  assert.ok(A.bonds().rad[idxOf(targetKey)] > radDefault * 1.5, "radius edit scales the pair's bondRad up");

  // (ii) per-type Remove: the row's Remove button drops the pair from bonds().types
  //      AND its bonds from bonds().list.
  assert.ok(typeKeys().includes(targetKey), "target type present before Remove");
  const rm = rmBtnFor(targetKey);
  assert.ok(rm, "Remove button must exist on the target row");
  rm.dispatchEvent(new window.Event("click"));
  assert.ok(!typeKeys().includes(targetKey), "Remove deletes the pair from bonds().types");
  assert.equal(countOf(targetKey), 0, "Remove deletes the pair's bonds from bonds().list");

  // (iii) Reset-bonds: #a-bondreset restores the FULL default type set (the removed
  //       pair returns) AND clears the per-pair Å back to the global default.
  doc.getElementById("a-bondreset").dispatchEvent(new window.Event("click"));
  assert.deepEqual(typeKeys().slice().sort(), fullDefault, "Reset restores bonds().types to the full default set");
  assert.equal(countOf(targetKey), before, "Reset re-detects the removed pair's bonds");
  assert.ok(Math.abs(A.bonds().rad[idxOf(targetKey)] - radDefault) < 1e-9,
    "Reset clears the per-pair Å (bondRad back to the global default)");
});

test("48: bond-edit rebuilds preserve the live camera view (Reset / Remove don't re-fit)", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const A = window.__atoms;
  const { camera, controls, home: camHome } = A.cam();
  assert.ok(camHome && camHome.pos, "camHome (framed default) must exist after init");
  const near = (a, b) => Math.abs(a - b) < 1e-6;

  // set a NON-default sentinel pose (as if the user rotated/zoomed)
  const S = { x: 123.5, y: -77.25, z: 41.0 }, T = { x: 5.5, y: -2.5, z: 9.0 };
  camera.position.set(S.x, S.y, S.z);
  controls.target.set(T.x, T.y, T.z);
  // non-vacuity: the sentinel must differ from the framed-default camHome we'd snap to
  assert.ok(Math.abs(camHome.pos.x - S.x) > 1 || Math.abs(camHome.pos.y - S.y) > 1 || Math.abs(camHome.pos.z - S.z) > 1,
    "sentinel pose must differ from the framed-default camHome (non-vacuous)");

  // (a) Reset rebuild must KEEP the view (rebuildKeepView), not re-fit to camHome
  window.document.getElementById("a-bondreset").dispatchEvent(new window.Event("click"));
  assert.ok(near(camera.position.x, S.x) && near(camera.position.y, S.y) && near(camera.position.z, S.z),
    "camera.position must be PRESERVED across the Reset rebuild (not re-fit to camHome)");
  assert.ok(near(controls.target.x, T.x) && near(controls.target.y, T.y) && near(controls.target.z, T.z),
    "controls.target must be PRESERVED across the Reset rebuild");

  // (b) a row Remove must preserve a (different) sentinel too
  const S2 = { x: -210.0, y: 88.0, z: -15.5 };
  camera.position.set(S2.x, S2.y, S2.z);
  const firstKey = (window.document.querySelector("#bonds-legend .bonds-legend-pair") || {}).textContent;
  assert.ok(firstKey, "a legend pair cell must exist after Reset");
  const rm = Array.from(window.document.querySelectorAll("#bonds-legend button")).find(b => b.title === "remove bond type: " + firstKey);
  assert.ok(rm, "the row must have a Remove button");
  rm.dispatchEvent(new window.Event("click"));
  assert.ok(near(camera.position.x, S2.x) && near(camera.position.y, S2.y) && near(camera.position.z, S2.z),
    "camera.position must be PRESERVED across a Remove rebuild too");
});

test("49: Reset bond radius button restores the global a-bondr to its default (0.15)", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const A = window.__atoms;
  const bondrIn = doc.getElementById("a-bondr");
  const resetBtn = doc.getElementById("a-bondr-reset");
  assert.ok(resetBtn, "#a-bondr-reset button must exist");
  assert.equal(bondrIn.value, "0.15", "a-bondr default value is 0.15");

  // bondRad[0] = global bondR * pair-factor[0]. At default (bondR=0.15, factor unset=1.0)
  // recover factor0 so the assertions are robust to whatever the pair factor is.
  const rad0def = A.bonds().rad[0];
  const factor0 = rad0def / 0.15;
  assert.ok(factor0 > 0, "bond 0 must have a positive radius (sanity)");

  // raise the GLOBAL radius to 0.4 -> rad[0] tracks it (0.4 * factor0)
  bondrIn.value = "0.4"; bondrIn.dispatchEvent(new window.Event("change"));
  assert.ok(Math.abs(0.4 - 0.15) > 1e-9, "non-vacuity: 0.4 differs from the default 0.15");
  assert.ok(Math.abs(A.bonds().rad[0] - 0.4 * factor0) < 1e-9, "rad[0] tracks the global radius at 0.4");

  // click "Reset bond radius" -> input back to 0.15 AND rad[0] restored (re-dispatch ran the handler)
  resetBtn.dispatchEvent(new window.Event("click"));
  assert.equal(bondrIn.value, "0.15", "Reset restores #a-bondr.value to 0.15");
  assert.ok(Math.abs(A.bonds().rad[0] - 0.15 * factor0) < 1e-9, "rad[0] restored to 0.15 * factor (global reset applied)");
});

test("50: F2d-2c — the Add row forms a species-pair bond type (update + push paths)", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const A = window.__atoms;
  const symbols = navData.meta.material.symbols;
  const style = navData.meta.material.species_style;
  const BOND_TOL = 1.2;
  const canon = (s1, s2) => [s1, s2].sort().join("\u2013");
  const keyOfType = t => canon(t.a, t.b);
  const species = [...new Set(symbols)];
  const rc = s => ((style[s] || {}).radius) || 1.0;

  // (1) Add row exists; selects populated 1:1 with distinct species; distance defaults
  //     to the covalent cutoff of the initial A/B selection.
  const selA = doc.getElementById("a-bond-add-a");
  const selB = doc.getElementById("a-bond-add-b");
  const dInp = doc.getElementById("a-bond-add-d");
  const addBtn = doc.getElementById("a-bond-add");
  assert.ok(selA && selB && dInp && addBtn, "Add row controls must exist");
  assert.equal(selA.options.length, species.length, "#a-bond-add-a has one option per distinct species");
  assert.equal(selB.options.length, species.length, "#a-bond-add-b has one option per distinct species");
  assert.equal(dInp.value, (BOND_TOL * (rc(selA.value) + rc(selB.value))).toFixed(2),
    "distance defaults to the covalent cutoff of the initial A/B selection");

  // (2) FORM proof on a pair NOT bonded at default: Se–Se (beyond its covalent cutoff; cf. 47).
  const sj = () => A.getSlots().j;
  const keys = () => A.bonds().list.map(([i, jj]) => canon(symbols[sj()[i]], symbols[sj()[jj]]));
  const countOf = key => keys().filter(k => k === key).length;
  const targetKey = "Se\u2013Se";
  assert.ok(species.includes("Se"), "synth must contain Se");
  assert.equal(countOf(targetKey), 0, "non-vacuity: Se–Se has 0 bonds at default (beyond its covalent cutoff)");

  const addPair = (a, b, d) => {
    selA.value = a; selA.dispatchEvent(new window.Event("change"));
    selB.value = b; selB.dispatchEvent(new window.Event("change"));
    dInp.value = String(d);                       // set AFTER the change handlers (which reset to covalent default)
    addBtn.dispatchEvent(new window.Event("click"));
  };
  addPair("Se", "Se", 5.0);                       // Se–Se IS in the table by default -> UPDATE path
  const tSeSe = A.bonds().types.find(t => keyOfType(t) === targetKey);
  assert.ok(tSeSe && Math.abs(tSeSe.max - 5.0) < 1e-9, "Add sets the Se–Se type max to 5.0");
  assert.ok(countOf(targetKey) >= 1, "Se–Se bonds FORM at the 5.0 distance (>=1, was 0)");

  // (3) PUSH path: Remove Se–Se (now bonded -> has a legend row), then Add it back at 5.0.
  const rmBtnFor = key => Array.from(doc.querySelectorAll("#bonds-legend button")).find(b => b.title === "remove bond type: " + key);
  const seRm = rmBtnFor(targetKey);
  assert.ok(seRm, "Se–Se has a legend Remove button once it is bonded");
  seRm.dispatchEvent(new window.Event("click"));
  assert.ok(!A.bonds().types.some(t => keyOfType(t) === targetKey), "Remove drops Se–Se from bonds().types");
  assert.equal(countOf(targetKey), 0, "Remove clears Se–Se bonds");
  addPair("Se", "Se", 5.0);                       // Se–Se absent -> PUSH path
  assert.ok(A.bonds().types.some(t => keyOfType(t) === targetKey && Math.abs(t.max - 5.0) < 1e-9),
    "Add (push) re-adds the Se–Se type at 5.0");
  assert.ok(countOf(targetKey) >= 1, "Add (push) re-forms Se–Se bonds");
});

test("51: F-polish — bonds-legend header row aligns guide words over the grid columns", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const A = window.__atoms;
  const symbols = navData.meta.material.symbols;
  const canon = (a, b) => [a, b].sort().join("\u2013");
  // the feature: the first four .bonds-legend-head cells read the guide words IN ORDER
  const heads = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-head")).map(h => h.textContent);
  assert.ok(heads.length >= 4, "header row must have >=4 cells (got " + heads.length + ")");
  assert.deepEqual(heads.slice(0, 4), ["Pairs", "Bond radius", "Color", "Bond distance"],
    "the first four header cells must read Pairs / Bond radius / Color / Bond distance in order");
  // pairs still listed: every distinct bonded pair has a .bonds-legend-pair cell (non-vacuity)
  const slots = A.getSlots();
  const ref = new Set(A.bonds().list.map(([i, jj]) => canon(symbols[slots.j[i]], symbols[slots.j[jj]])));
  assert.ok(ref.size >= 1, "synth must have >=1 bonded pair");
  const pairCells = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-pair")).map(c => c.textContent);
  assert.deepEqual(new Set(pairCells), ref, ".bonds-legend-pair cells must still list every distinct bonded pair");
  assert.equal(pairCells.length, ref.size, "one pair cell per distinct pair (no dupes/omissions)");
});

test("52: F-polish — Remove × docks into the distance cell (shared parent); header is 4 cells", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const A = window.__atoms;
  const symbols = navData.meta.material.symbols;
  const canon = (a, b) => [a, b].sort().join("\u2013");
  // pick a bonded pair
  const slots = A.getSlots();
  const present = [...new Set(A.bonds().list.map(([i, jj]) => canon(symbols[slots.j[i]], symbols[slots.j[jj]])))];
  assert.ok(present.length >= 1, "synth must have >=1 bonded pair");
  const key = present[0];
  // the Remove button (by title) and the distance input (by title) share ONE parent cell
  const rm = Array.from(doc.querySelectorAll("#bonds-legend button")).find(b => b.title === "remove bond type: " + key);
  const maxInp = Array.from(doc.querySelectorAll("#bonds-legend input")).find(i => i.title === "bond distance (\u00c5): " + key);
  assert.ok(rm, "Remove button must exist for " + key);
  assert.ok(maxInp, "distance input must exist for " + key);
  assert.equal(rm.textContent, "\u00d7", "Remove button is the compact × glyph");
  assert.strictEqual(rm.parentElement, maxInp.parentElement, "Remove and distance input share one parent cell");
  assert.ok(rm.parentElement.classList.contains("bonds-legend-distcell"), "the shared cell is .bonds-legend-distcell");
  // header row is now exactly 4 cells reading the four guide words
  const heads = Array.from(doc.querySelectorAll("#bonds-legend .bonds-legend-head")).map(h => h.textContent);
  assert.equal(heads.length, 4, "header row has exactly 4 cells (5->4)");
  assert.deepEqual(heads, ["Pairs", "Bond radius", "Color", "Bond distance"],
    "the 4 header cells read Pairs / Bond radius / Color / Bond distance");
});

test("53: F1c-polish — atoms legend is a de-labeled swatch|symbol|radius grid (cells survive)", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const symbols = navData.meta.material.symbols;
  const uniq = [...new Set(symbols)];
  assert.ok(uniq.length >= 1, "synth must have >=1 species");
  // symbol cells: exactly one .atoms-legend-sym per unique species, reading the symbols
  const symCells = Array.from(doc.querySelectorAll("#atoms-legend .atoms-legend-sym")).map(c => c.textContent);
  assert.equal(symCells.length, uniq.length, "one .atoms-legend-sym cell per unique species");
  assert.deepEqual(new Set(symCells), new Set(uniq), "symbol cells read the actual species symbols");
  // regression: pickers + radius inputs survive the de-labeling (one per species, by title)
  const pickers = Array.from(doc.querySelectorAll("#atoms-legend input[type=color]"));
  const radInputs = Array.from(doc.querySelectorAll("#atoms-legend input[type=number]"));
  assert.equal(pickers.length, uniq.length, "one colour picker per species (survives de-labeling)");
  assert.equal(radInputs.length, uniq.length, "one radius input per species (survives de-labeling)");
  uniq.forEach(sym => {
    assert.ok(pickers.find(p => p.title === "color: " + sym), "colour picker present for " + sym);
    assert.ok(radInputs.find(r => r.title === "display radius (Å): " + sym), "radius input present for " + sym);
  });
});

test("54: F1c-polish — atoms radius track is auto (no fixed 4em overflow into the next swatch)", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const css = Array.from(window.document.querySelectorAll("style")).map(s => s.textContent).join("\n");
  assert.ok(css.length > 0, "stylesheet text must be present");
  // the atoms grid radius track is now auto (grows to fit the input box), not a fixed 4em
  assert.ok(css.includes("repeat(2, auto minmax(2.5em, auto) auto)"),
    "atoms grid must use an auto radius track");
  assert.ok(!css.includes("minmax(2.5em, auto) 4em"),
    "atoms grid must NOT keep the old fixed 4em radius track");
  // the radius input's margin-left (which spilled into the next track) is gone
  assert.ok(!css.includes("#atoms-legend input[type=number]"),
    "the #atoms-legend input[type=number] margin-left rule must be removed");
});

test("55: panel cleanup — dead #atoms-legend label CSS rule removed", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const css = Array.from(window.document.querySelectorAll("style")).map(s => s.textContent).join("\n");
  assert.ok(css.length > 0, "stylesheet text must be present");
  assert.ok(!css.includes("#atoms-legend label"),
    "the dead '#atoms-legend label' rule (de-labeled since 0.6.56) must be gone");
});

test("56: panel cleanup — bonds Remove × carries aria-label === title (accessible name)", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const rmButtons = Array.from(doc.querySelectorAll("#bonds-legend button"))
    .filter(b => b.title.startsWith("remove bond type: "));
  assert.ok(rmButtons.length >= 1, "at least one Remove × button present");
  for (const b of rmButtons) {
    assert.equal(b.textContent, "\u00d7", "Remove is the × glyph");
    assert.equal(b.getAttribute("aria-label"), b.title,
      "aria-label must equal title (== 'remove bond type: ' + key) — the × needs an explicit accessible name");
  }
});

test("57: bond-panel spelling — #a-bondreset title is American 'colors' (no 'colour')", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const btn = window.document.getElementById("a-bondreset");
  assert.ok(btn, "#a-bondreset button present");
  assert.ok(/reset .*radii and colors to default/.test(btn.title),
    "reset-bonds title reads '...radii and colors to default' (got: " + btn.title + ")");
  assert.ok(!/colour/i.test(btn.title), "reset-bonds title must NOT contain British 'colour'");
});

// --- 0.6.66 XSS: payload-derived strings reach innerHTML only after esc() ---
// The navigator payload is baked from a user-supplied phonopy file (and, via
// pv_serve, a browser upload), so atom symbols and high-symmetry labels are
// UNTRUSTED. Audit finding #1 (HIGH): three sinks wrote them to innerHTML raw
//   :567  atom-position table  <- SYMS[i]
//   :2131 status line          <- symbols.join("")
//   :2115 jump button          <- tickLabel(label)
// A symbol/label like <img src=x onerror=...> was stored XSS in the shared page.
// jsdom runs with runScripts:"outside-only" — it PARSES injected markup into live
// DOM nodes (an <img>/<script> element) but never fires their handlers. So the
// load-bearing, revert-sensitive assertion is DOM STRUCTURE: an escaped payload
// yields ZERO injected elements and an entity-encoded string; an un-escaped one
// yields a live <img>/<script> node. window.__pwned is a belt-and-suspenders
// check (never fires under outside-only) and is not what the negative control
// hangs on.
const XSS_SYM = '<img src=x onerror="window.__pwned=1">';
const XSS_LBL = '<img src=y onerror="window.__pwnedLabel=1">';

test("58: XSS — malicious atom symbol is entity-escaped in the positions table, not parsed as an element", () => {
  const nav = synthNavData();
  nav.meta.material.symbols = [XSS_SYM, "Se", "Se"];   // symbol[0] carries the payload
  const { window, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;

  // sink #1 — atom-position table (:567): the payload rendered as INERT text.
  const table = doc.getElementById("atom-pos-table");
  assert.ok(table, "#atom-pos-table present");
  assert.equal(table.querySelector("img"), null,
    "a live <img> was injected into the positions table — symbol reached innerHTML un-escaped");
  assert.match(table.innerHTML, /&lt;img/,
    "the malicious symbol must appear entity-encoded (&lt;img), proving esc() ran");
  // the first row's symbol cell shows the literal payload as text, markup intact-as-text
  const sym0 = table.querySelector(".atom-pos-row .atom-pos-sym");
  assert.equal(sym0.textContent, XSS_SYM,
    "the symbol cell's textContent is the raw payload string (rendered, not executed)");
  assert.equal(sym0.querySelector("img"), null, "no live <img> inside the symbol cell");

  // the onerror never ran (defensive; jsdom outside-only would not fire it anyway)
  assert.equal(window.__pwned, undefined, "the injected onerror payload must not have executed");

  // Sink #3 (status line, :2131) shares this exact taint (symbols.join) and the
  // same esc() fix, but is NOT separately asserted here: the "Loaded …" text is
  // overwritten synchronously on load by the default mode selection (see test 8),
  // so it is not observable in the settled DOM. esc() coverage of #3 rides on the
  // same helper proven revert-sensitive by the table assertions above.
});

test("59: XSS — malicious high-symmetry label is escaped in the jump button (tickLabel routes through esc first)", () => {
  const nav = synthNavData();
  // first HSP label carries the payload; tickLabel() is the sink at :2115.
  nav.band.high_symmetry_points[0] = { label: XSS_LBL, q_cart: [0, 0], distance: 0 };
  const { window, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const jb = window.document.getElementById("mode-jump-buttons");
  assert.ok(jb, "#mode-jump-buttons present");
  assert.ok(jb.querySelectorAll("button").length >= 1, "at least one jump button built");
  assert.equal(jb.querySelector("img"), null,
    "a live <img> was injected via the high-symmetry label — tickLabel reached innerHTML un-escaped");
  assert.match(jb.innerHTML, /&lt;img/,
    "the malicious label must appear entity-encoded (&lt;img) in the jump button");
  assert.equal(window.__pwnedLabel, undefined, "the injected label payload must not have executed");
});

test("60: XSS-fix is behavior-preserving — a legitimate TeX label still renders its <sup>/prime (esc did not over-sanitize)", () => {
  const nav = synthNavData();
  nav.band.high_symmetry_points[0] = { label: "$X^{\\prime}$", q_cart: [0, 0], distance: 0 };
  const { window, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const jb = window.document.getElementById("mode-jump-buttons");
  const btn = jb.querySelector("button");
  assert.ok(btn, "jump button for the TeX label present");
  // esc() escapes < > & " ' but NOT \ { } ^ _ $, so tickLabel's own prime/sup
  // transforms still fire: "$X^{\prime}$" -> "X" + "′" (or a <sup>).
  assert.ok(btn.querySelector("sup") !== null || /′/.test(btn.textContent),
    "legitimate TeX markup must survive escaping (got: " + btn.innerHTML + ")");
  assert.ok(!/[<>]/.test(btn.textContent.replace(/′/g, "")),
    "no raw angle brackets left in the rendered label");
});

// --- 0.6.69: bond-settings side-car (docs/bonds_sidecar.md) ---
// serializeBondSettings()/applyBondSettings() are pure functions on the curated
// bond state, exposed via window.__atoms; the Export/Import controls are thin
// wrappers. File pair keys are lexicographically sorted, ASCII-hyphen joined
// ("Se-Sn"); internal legend keys use the en-dash. Deletion round-trips because
// apply REPLACES the bond-type table (absent pair = deleted).
const EN_DASH = "–";

// objects returned from the jsdom page realm carry that realm's Object.prototype,
// so node:assert/strict deepStrictEqual (which compares prototypes) treats them as
// unequal to test-realm literals even when structurally identical. Normalize
// page-produced values through JSON so both operands live in the test realm.
const plain = o => JSON.parse(JSON.stringify(o));

test("61: side-car round-trip — curated radius/color and a DELETED pair survive serialize→reset→apply", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, A = window.__atoms;
  const inpByTitle = t => Array.from(doc.querySelectorAll("#bonds-legend input")).find(i => i.title === t);
  // curate the visible pair Se–Sn: absolute radius 0.30 Å + custom color #aa33cc
  const radInp = inpByTitle("bond cylinder radius (Å): Se" + EN_DASH + "Sn");
  assert.ok(radInp, "Se–Sn radius input present");
  radInp.value = "0.30"; radInp.dispatchEvent(new window.Event("input"));
  const colInp = inpByTitle("custom bond color: Se" + EN_DASH + "Sn");
  colInp.value = "#aa33cc"; colInp.dispatchEvent(new window.Event("input"));
  // DELETE the default Sn–Sn type: Add it (4.0 Å forms its bonds → legend row), then Remove it
  doc.getElementById("a-bond-add-a").value = "Sn";
  doc.getElementById("a-bond-add-b").value = "Sn";
  doc.getElementById("a-bond-add-d").value = "4.0";
  doc.getElementById("a-bond-add").dispatchEvent(new window.Event("click"));
  const rm = Array.from(doc.querySelectorAll("#bonds-legend button"))
    .find(b => b.title === "remove bond type: Sn" + EN_DASH + "Sn");
  assert.ok(rm, "Sn–Sn legend row appeared after Add");
  rm.dispatchEvent(new window.Event("click"));
  // snapshot: deleted pair ABSENT, curation present, format header correct
  const S = A.serializeBondSettings();
  assert.equal(S.format, "phonon-vision-bonds");
  assert.equal(S.version, 1);
  assert.equal(S.units, "angstrom");
  assert.deepEqual(plain(S.material), { formula: "SnSeSe", natoms: 3 });
  assert.equal(S.bonds["Sn-Sn"], undefined, "DELETED pair must be absent from the side-car");
  // 0.6.76: radius_A (absolute Å) is authoritative; radius_scale is DERIVED
  // (= radius_A / 0.15, the navigator's default-radius constant) for 0.7.9-Blender.
  assert.equal(S.bonds["Se-Sn"].radius_A, 0.30, "curated absolute Å radius serialized");
  assert.equal(S.bonds["Se-Sn"].radius_scale, 2, "derived legacy radius_scale = 0.30/0.15");
  assert.equal(S.bonds["Se-Sn"].color, "#aa33cc", "curated custom color serialized");
  assert.ok(S.bonds["Se-Se"] && typeof S.bonds["Se-Se"].max === "number",
    "untouched default pair serialized with its absolute-Å max");
  // reset restores defaults (Sn–Sn back, curation cleared) — the apply below must redo everything
  doc.getElementById("a-bondreset").dispatchEvent(new window.Event("click"));
  const D0 = A.serializeBondSettings();
  assert.ok(D0.bonds["Sn-Sn"], "reset restores the default Sn–Sn type");
  // both radius keys are ALWAYS emitted (0.6.76 contract): defaults after reset
  assert.equal(D0.bonds["Se-Sn"].radius_A, 0.15, "reset returns the pair to the Å default");
  assert.equal(D0.bonds["Se-Sn"].radius_scale, 1, "derived legacy scale is 1 at the default");
  assert.equal(D0.bonds["Se-Sn"].color, undefined, "reset clears the custom color");
  // apply the snapshot → state deep-equals; the deleted pair stays deleted
  assert.equal(A.applyBondSettings(S), true);
  assert.deepEqual(plain(A.serializeBondSettings()), plain(S), "serialize(apply(S)) must deep-equal S");
  assert.ok(!A.bonds().types.some(t => t.a === "Sn" && t.b === "Sn"),
    "the DELETED pair stays deleted after apply (override semantics)");
});

test("62: side-car canonicalization — 'Sn-Se'-keyed and 'Se-Sn'-keyed files produce identical state", () => {
  const mk = key => ({ format: "phonon-vision-bonds", version: 1, units: "angstrom",
                       bonds: { [key]: { max: 2.5, radius_scale: 1.4, color: "#112233" } } });
  const p1 = loadPage(); assert.equal(p1.threw, null, p1.threw && p1.threw.stack);
  assert.equal(p1.window.__atoms.applyBondSettings(mk("Se-Sn")), true);
  const s1 = p1.window.__atoms.serializeBondSettings();
  const t1 = p1.window.__atoms.bonds().types.map(t => [t.a, t.b, t.max]);
  const p2 = loadPage(); assert.equal(p2.threw, null, p2.threw && p2.threw.stack);
  assert.equal(p2.window.__atoms.applyBondSettings(mk("Sn-Se")), true);
  const s2 = p2.window.__atoms.serializeBondSettings();
  const t2 = p2.window.__atoms.bonds().types.map(t => [t.a, t.b, t.max]);
  assert.deepEqual(plain(s1), plain(s2), "serialized state must not depend on the file's key order");
  assert.deepEqual(plain(t1), plain(t2), "live bondTypes rows must not depend on the file's key order");
  assert.deepEqual(Object.keys(s1.bonds), ["Se-Sn"], "file keys re-canonicalize to sorted order");
  for (const t of p2.window.__atoms.bonds().types)
    assert.ok(t.a <= t.b, "bondTypes rows keep a<=b, got " + t.a + "," + t.b);
});

test("63: side-car ordering independence — one file applies identically to atom-permuted payloads", () => {
  // same material, permuted atom order: synth is [Sn,Se,Se] → [Se,Se,Sn]
  const permuted = () => {
    const d = synthNavData(), perm = [1, 2, 0];   // new[i] = old[perm[i]]
    const M = d.meta.material;
    M.symbols = perm.map(i => M.symbols[i]);
    M.masses = perm.map(i => M.masses[i]);
    M.fractional_coords = perm.map(i => M.fractional_coords[i]);
    d.band.eigenvectors_real = d.band.eigenvectors_real.map(q => q.map(b => perm.map(i => b[i])));
    d.band.eigenvectors_imag = d.band.eigenvectors_imag.map(q => q.map(b => perm.map(i => b[i])));
    return d;
  };
  const F = { format: "phonon-vision-bonds", version: 1,
              bonds: { "Sn-Se": { max: 2.6, radius_scale: 1.4, color: "#336699" },  // deliberately NON-canonical key
                       "Se-Se": { max: 2.9 } } };
  const read = pg => {
    const A = pg.window.__atoms;
    assert.equal(A.applyBondSettings(F), true);
    for (const t of A.bonds().types) assert.ok(t.a <= t.b, "canonical row shape a<=b, got " + t.a + "," + t.b);
    const s = A.serializeBondSettings();
    assert.equal(s.bonds["Se-Sn"].radius_scale, 1.4, "radius_scale must land under the CANONICAL key");
    assert.equal(s.bonds["Se-Sn"].color, "#336699", "color must land under the CANONICAL key");
    return { bonds: s.bonds, types: A.bonds().types.map(t => [t.a, t.b, t.max]).sort() };
  };
  const p1 = loadPage(); assert.equal(p1.threw, null, p1.threw && p1.threw.stack);
  const p2 = loadPage({ navData: permuted() }); assert.equal(p2.threw, null, p2.threw && p2.threw.stack);
  const r1 = read(p1), r2 = read(p2);
  assert.deepEqual(plain(r1.bonds), plain(r2.bonds), "identical per-pair file state on both payloads");
  assert.deepEqual(plain(r1.types), plain(r2.types), "identical bondTypes rows on both payloads");
});

test("64: side-car loose binding — formula/natoms mismatch WARNS in #status but applies", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const A = window.__atoms;
  const F = { format: "phonon-vision-bonds", version: 1,
              material: { formula: "WSe2", natoms: 12 },
              bonds: { "Se-Sn": { max: 2.5 } } };
  assert.equal(A.applyBondSettings(F), true);
  const st = window.document.getElementById("status").textContent;
  assert.match(st, /WARNING/, "mismatch warning shown in #status");
  assert.match(st, /WSe2\/12/, "warning names the file's material");
  assert.match(st, /does not match loaded SnSeSe\/3/, "warning names the loaded material");
  assert.match(st, /applied anyway/, "warning states the settings were applied");
  assert.deepEqual(plain(A.bonds().types), [{ a: "Se", b: "Sn", max: 2.5 }],
    "settings applied despite the mismatch (loose binding)");
});

test("65: side-car rejection — malformed JSON / wrong format / missing table leave state byte-unchanged", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const A = window.__atoms, doc = window.document;
  const before = JSON.stringify(A.serializeBondSettings());
  const typesRef = A.bonds().types;                      // array-identity guard: no partial apply
  assert.equal(A.importBondSettingsText("{ not json"), false);
  assert.match(doc.getElementById("status").textContent, /malformed JSON — ignored/);
  assert.equal(A.applyBondSettings({ format: "some-other-tool", version: 1, bonds: {} }), false);
  assert.match(doc.getElementById("status").textContent, /wrong "format" — ignored/);
  assert.equal(A.applyBondSettings(null), false);
  assert.equal(A.applyBondSettings({ format: "phonon-vision-bonds", version: 1 }), false);
  assert.match(doc.getElementById("status").textContent, /no bonds table — ignored/);
  assert.equal(JSON.stringify(A.serializeBondSettings()), before,
    "state byte-unchanged after every rejection");
  assert.equal(A.bonds().types, typesRef, "bondTypes array identity untouched (validation precedes any write)");
});

test("66: side-car controls — Export downloads <formula>_bonds.json; Import file input applies", async () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, A = window.__atoms;
  // Export: jsdom lacks URL.createObjectURL — polyfill and capture the Blob
  let blobSeen = null, revoked = 0;
  window.URL.createObjectURL = b => { blobSeen = b; return "blob:pv-test"; };
  window.URL.revokeObjectURL = () => { revoked++; };
  doc.getElementById("a-bondexport").dispatchEvent(new window.Event("click"));
  assert.ok(blobSeen, "Export built a Blob");
  assert.equal(revoked, 1, "object URL revoked after the click-through");
  assert.match(doc.getElementById("status").textContent, /exported SnSeSe_bonds\.json/,
    "status names the <formula>_bonds.json download");
  const parsed = JSON.parse(await blobSeen.text());
  assert.deepEqual(parsed, plain(A.serializeBondSettings()), "exported JSON equals serializeBondSettings()");
  // Import: drive the real file-input change path (FileReader is async in jsdom)
  const F = { format: "phonon-vision-bonds", version: 1, bonds: { "Se-Sn": { max: 2.44 } } };
  const file = new window.File([JSON.stringify(F)], "SnSeSe_bonds.json", { type: "application/json" });
  const inp = doc.getElementById("a-bondimport");
  Object.defineProperty(inp, "files", { configurable: true, value: [file] });
  inp.dispatchEvent(new window.Event("change"));
  for (let i = 0; i < 40 && A.bonds().types.length !== 1; i++)
    await new Promise(r => setTimeout(r, 25));
  assert.deepEqual(plain(A.bonds().types), [{ a: "Se", b: "Sn", max: 2.44 }],
    "import applied through the file input + FileReader path");
});

// --- 0.6.71 E2 viewer swap: mesh3d + per-vertex RGB blend on the 3D surface ---
// Payload WITH the 0.6.70 weight grids. Deliberate stress choices:
//  * species_order is PERMUTED vs first-appearance (["Se","Sn"], symbols are
//    Sn,Se,Se) — a viewer that pairs weight columns by payload symbol order
//    instead of species_order mis-colours (negative control (a) bites here);
//  * opts.normalized=false scales all weights by 0.5 — a viewer that skips the
//    per-vertex renormalization mis-colours (negative control (b) bites here);
//  * a 3x3 grid with one masked (null) corner exercises the triangulation.
function synthNavDataWeights(opts = {}) {
  const normalized = opts.normalized !== false;
  const d = synthNavData();
  const kx = [-0.1, 0, 0.1], ky = [-0.1, 0, 0.1];
  d.surface.kx = kx; d.surface.ky = ky;
  d.surface.species_order = ["Se", "Sn"];
  const nB = d.surface.bands.length;                 // 9, matches the band block
  d.surface.bands = [];
  for (let b = 0; b < nB; b++) {
    const z = [], ch = [], wSe = [], wSn = [];
    for (let i = 0; i < 3; i++) {
      const zr = [], cr = [], se = [], sn = [];
      for (let j = 0; j < 3; j++) {
        if (i === 0 && j === 0) { zr.push(null); cr.push(null); se.push(null); sn.push(null); continue; }
        zr.push(b + 0.1 * i + 0.01 * j);
        const fSn = (i + j) / 4;                     // Sn fraction, 0.25..1
        cr.push(fSn);                                // character = group-A(Sn) sum
        const scale = normalized ? 1 : 0.5;
        se.push((1 - fSn) * scale); sn.push(fSn * scale);
      }
      z.push(zr); ch.push(cr); wSe.push(se); wSn.push(sn);
    }
    d.surface.bands.push({ z: z, character: ch, weights: [wSe, wSn] }); // species_order order
  }
  return d;
}
const SE_COL = [1, 0.63, 0], SN_COL = [0.4, 0.5, 0.5];   // synth species_style colours
// live surf3d trace array (mutated by the delete+add replacement toggle)
const surfData = state => state.newPlot.find(p => p.id === "surf3d").data;
// 0.6.74: a BINARY payload defaults to Projection (PART 2), so blend-value tests
// switch to Blend first; after the replacement toggle, surfData() holds the fresh
// blend traces. Returns the loadPage result.
function loadWeightsBlend(opts) {
  const p = loadPage({ navData: synthNavDataWeights(opts) });
  const sel = p.window.document.getElementById("surf-colormode");
  if (sel.value !== "blend") { sel.value = "blend"; sel.dispatchEvent(new p.window.Event("change")); }
  return p;
}

test("67: E2 fallback — weightless payload builds today's scalar surface traces; toggle shown minus Blend (0.6.96)", () => {
  const { window, state, threw } = loadPage();            // synthNavData: NO weights key
  assert.equal(threw, null, threw && threw.stack);
  const sp = state.newPlot.find(p => p.id === "surf3d");
  const nB = 9;
  for (let b = 0; b < nB; b++) {
    assert.equal(sp.data[b].type, "surface", "branch trace " + b + " stays type surface");
    assert.ok(sp.data[b].surfacecolor, "scalar surfacecolor present");
    assert.ok(sp.data[b].colorscale, "CHAR_SCALE colorscale present");
    assert.equal(sp.data[b].vertexcolor, undefined, "no vertexcolor on the fallback path");
  }
  assert.equal(sp.data[nB].type, "mesh3d", "BZ prism still at index nB");
  // 0.6.96 DELIBERATE CONTRACT CHANGE: the dropdown now shows for every
  // surface payload (Plain is universal); only Blend needs weights.
  const wrap = window.document.getElementById("surf-colormode-wrap");
  assert.notEqual(wrap.style.display, "none", "colour-mode toggle SHOWN for weightless payloads");
  const values = [...window.document.getElementById("surf-colormode").options].map(o => o.value);
  assert.ok(!values.includes("blend"), "Blend removed without weights");
  assert.ok(values.includes("plain"), "Plain offered");
});

test("68: E2 blend path — mesh3d at 0..nB-1, shared consistent i/j/k, per-vertex colour/hover; indices preserved", () => {
  const { window, state, threw } = loadWeightsBlend();     // binary defaults to projection -> switch to blend
  assert.equal(threw, null, threw && threw.stack);
  const sp = surfData(state);                              // LIVE array (post-replacement blend traces)
  const nB = 9, nV = 8;                                    // 3x3 grid, one null corner
  for (let b = 0; b < nB; b++) {
    const t = sp[b];
    assert.equal(t.type, "mesh3d", "branch trace " + b + " is mesh3d");
    assert.equal(t.x.length, nV, "vertex count");
    assert.equal(t.vertexcolor.length, nV, "vertexcolor per vertex");
    assert.equal(t.text.length, nV, "hover text per vertex");
    // KEY-PRESENCE — the blend trace must carry NO intensity/colorscale KEYS
    // (present-but-null poisons the vertexcolor branch in supplyDefaults).
    assert.ok(!("intensity" in t), "no intensity KEY (present-null would eat the blend)");
    assert.ok(!("colorscale" in t), "no colorscale KEY (PoC vertexcolor-only shape)");
    assert.ok(t.i === sp[0].i && t.j === sp[0].j && t.k === sp[0].k,
      "triangulation arrays shared across branches");
  }
  // PoC invariants ported: count, range, integrality, no degenerates
  const { i: TI, j: TJ, k: TK } = sp[0];
  assert.equal(TI.length, 6, "3x3 grid with one masked corner -> 6 triangles");
  assert.ok(TJ.length === 6 && TK.length === 6, "i/j/k equal lengths");
  for (let t = 0; t < TI.length; t++) {
    for (const v of [TI[t], TJ[t], TK[t]])
      assert.ok(Number.isInteger(v) && v >= 0 && v < nV, "index in [0,nV)");
    assert.ok(TI[t] !== TJ[t] && TJ[t] !== TK[t] && TI[t] !== TK[t], "no degenerate triangle");
  }
  assert.equal(sp[nB].type, "mesh3d", "BZ prism still at index nB (bookkeeping preserved)");
  assert.equal(sp[nB].color, "#9467bd", "BZ prism colour untouched");
  assert.match(sp[0].text[0], /Se 0\.\d\d · Sn 0\.\d\d/, "terse per-species hover");
  const wrap = window.document.getElementById("surf-colormode-wrap");
  assert.notEqual(wrap.style.display, "none", "colour-mode toggle shown");
});

test("69: E2 palette consistency — vertexcolor == the shared mixer on species_order-paired, RENORMALIZED weights", () => {
  // unnormalized payload (all weights x0.5) + permuted species_order: the viewer
  // must (i) pair columns by species_order and (ii) renormalize per vertex.
  const { window, state, threw } = loadWeightsBlend({ normalized: false });
  assert.equal(threw, null, threw && threw.stack);
  const A = window.__atoms;
  const vc = surfData(state)[0].vertexcolor;
  // vertex 7 = grid (2,2): fSn=1 -> fr=[0,1] over ["Se","Sn"] -> pure Sn colour
  assert.equal(vc[7], A.mixRGB([0, 1], [SE_COL, SN_COL]), "pure-Sn vertex == mixer output");
  // vertex 2 = grid (1,0): fSn=0.25 -> fr=[0.75,0.25] AFTER renormalization
  assert.equal(vc[2], A.mixRGB([0.75, 0.25], [SE_COL, SN_COL]), "mixed vertex == mixer output");
  // and the mixer is the SAME single source the 2D blend uses (exposed helper)
  assert.equal(typeof A.mixRGB, "function");
});

test("70: E2 toggle = FULL REBUILD through the init path — exactly one purge + one newPlot on surf3d; fresh array, presence correct, BZ prism at nB", () => {
  const { window, state, threw } = loadWeightsBlend();       // start in blend
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, nB = 9;
  const sp = surfData(state);
  const beforeRefs = sp.slice(0, nB);                        // capture branch-trace refs
  const sel = doc.getElementById("surf-colormode");
  const np0 = state.newPlotCalls.filter(id => id === "surf3d").length;
  const pg0 = state.purge.filter(id => id === "surf3d").length;
  const d0 = state.deleteTraces.length + state.addTraces.length;
  sel.value = "projection"; sel.dispatchEvent(new window.Event("change"));
  // the mode flip goes through the SINGLE init path: one purge + one newPlot; NEVER
  // delete+add or restyle (those are the compositing-order / GL-staleness traps).
  assert.equal(state.purge.filter(id => id === "surf3d").length, pg0 + 1, "exactly one surf3d purge");
  assert.equal(state.newPlotCalls.filter(id => id === "surf3d").length, np0 + 1, "exactly one surf3d newPlot");
  assert.equal(state.deleteTraces.length + state.addTraces.length, d0, "no deleteTraces/addTraces (not a replacement)");
  const sp2 = surfData(state);                               // fresh plot data
  assert.equal(sp2.length, 2 * nB + 1, "full array rebuilt (nB branches + 9 fixed + the 0.7.2 placeholder)");
  for (let b = 0; b < nB; b++) {
    assert.notEqual(sp2[b], beforeRefs[b], "branch trace " + b + " is a NEW object (rebuild)");
    assert.ok(!("vertexcolor" in sp2[b]) && Array.isArray(sp2[b].intensity) && "colorscale" in sp2[b],
      "projection presence (trace " + b + ")");
  }
  assert.equal(sp2[nB].type, "mesh3d", "BZ prism still at index nB");
  assert.equal(sp2[nB].color, "#9467bd", "BZ prism colour intact");
});

test("71: E2 recolor propagation — species picker restyles the 3D blend in blend mode (no newPlot)", () => {
  const { window, state, threw } = loadWeightsBlend();       // blend mode (so recolor engages)
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const nPlots = state.newPlot.length;
  const picker = Array.from(doc.querySelectorAll("#atoms-legend input[type=color]"))
    .find(i => i.title === "color: Sn");
  assert.ok(picker, "Sn picker present");
  const r0 = state.restyle.length;
  picker.value = "#000000"; picker.dispatchEvent(new window.Event("input"));
  const rs = state.restyle.slice(r0).filter(r => r.id === "surf3d" && r.update && "vertexcolor" in r.update);
  assert.equal(rs.length, 1, "recolor propagates to the 3D blend via ONE restyle");
  // vertex 7 is pure Sn: overridden to black -> rgb(0,0,0)
  assert.equal(rs[0].update.vertexcolor[0][7], "rgb(0,0,0)", "blend uses the LIVE override colour");
  assert.equal(state.newPlot.length, nPlots, "restyle only — no newPlot");
});

// --- 0.6.74: mode class flips by trace REPLACEMENT (the gl-mesh3d staleness fix) ---
// The vendored 2.35.2 mesh3d supplyDefaults selects the colour pipeline by KEY
// PRESENCE ("intensity"in t), and its gl-mesh3d update() does NOT clear the scalar
// colormap/vertexIntensity when those keys are later deleted via restyle — so a
// restyle/react mode flip leaves a stale colormap that overrides fresh
// vertexColors. The fix replaces the traces (deleteTraces+addTraces of fresh
// objects). jsdom never renders, so the harness reads the LIVE post-replacement
// gd.data directly (the mock mutates it exactly as Plotly does).

test("72: E2 KEY-PRESENCE across modes (live gd.data) — blend vertexcolor-only, projection intensity-only; round-trip deep-equals build", () => {
  const { window, state, threw } = loadWeightsBlend();       // blend (binary default is projection)
  assert.equal(threw, null, threw && threw.stack);
  const nB = 9, nV = 8, sel = window.document.getElementById("surf-colormode");
  const checkBlend = () => { const sp = surfData(state); for (let b = 0; b < nB; b++) {
    assert.ok(!("intensity" in sp[b]), "blend: no intensity KEY (trace " + b + ")");
    assert.ok(!("colorscale" in sp[b]), "blend: no colorscale KEY (trace " + b + ")");
    assert.ok(Array.isArray(sp[b].vertexcolor) && sp[b].vertexcolor.length === nV, "blend: vertexcolor arrays (trace " + b + ")");
  } };
  const checkProj = () => { const sp = surfData(state); for (let b = 0; b < nB; b++) {
    assert.ok(!("vertexcolor" in sp[b]), "projection: no vertexcolor KEY (trace " + b + ")");
    assert.ok(Array.isArray(sp[b].intensity) && sp[b].intensity.length === nV, "projection: intensity arrays (trace " + b + ")");
    assert.ok("colorscale" in sp[b], "projection: colorscale KEY present (trace " + b + ")");
  } };
  const built = surfData(state).slice(0, nB).map(plain);    // freshly-built blend traces
  checkBlend();
  sel.value = "projection"; sel.dispatchEvent(new window.Event("change")); checkProj();
  sel.value = "blend"; sel.dispatchEvent(new window.Event("change")); checkBlend();
  const after = surfData(state).slice(0, nB).map(plain);
  for (let b = 0; b < nB; b++) assert.deepEqual(after[b], built[b], "round-trip trace " + b + " deep-equals build");
});

test("73: E2 recolor while in projection — no surf3d restyle, vertexcolor KEY absent; blend re-entry rebuilds with the live palette", () => {
  const { window, state, threw } = loadPage({ navData: synthNavDataWeights() });   // binary -> defaults to PROJECTION
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, sel = doc.getElementById("surf-colormode");
  assert.equal(sel.value, "projection", "binary payload defaults to projection");
  const r0 = state.restyle.length;
  const picker = Array.from(doc.querySelectorAll("#atoms-legend input[type=color]"))
    .find(i => i.title === "color: Sn");
  picker.value = "#000000"; picker.dispatchEvent(new window.Event("input"));
  assert.equal(state.restyle.slice(r0).filter(r => r.id === "surf3d").length, 0,
    "recolor during projection issues NO surf3d restyle");
  assert.ok(!("vertexcolor" in surfData(state)[0]), "vertexcolor KEY still absent after the recolor");
  sel.value = "blend"; sel.dispatchEvent(new window.Event("change"));
  const t = surfData(state)[0];
  assert.equal(t.vertexcolor[7], "rgb(0,0,0)", "blend re-entry paints with the LIVE override (pure-Sn vertex black)");
  assert.ok(!("intensity" in t), "and no intensity KEY in the blend trace");
});

// independent RGB recompute (does NOT call the viewer's mixRGB) — pins the
// pairing (species_order) + per-vertex renormalization + palette to arithmetic
// outside the code under test.
function expectRGB(fr, cols) {
  let r = 0, g = 0, b = 0;
  for (let i = 0; i < fr.length; i++) { r += fr[i]*cols[i][0]; g += fr[i]*cols[i][1]; b += fr[i]*cols[i][2]; }
  return "rgb(" + Math.round(r*255) + "," + Math.round(g*255) + "," + Math.round(b*255) + ")";
}

test("74: E2 numeric blend probe (synthetic) — vertexcolor == independent recompute on species_order-paired renormalized weights", () => {
  // unnormalized (x0.5) + species_order ["Se","Sn"] permuted vs symbols[Sn,Se,Se]
  const { window, state, threw } = loadWeightsBlend({ normalized: false });
  assert.equal(threw, null, threw && threw.stack);
  const vc = surfData(state)[0].vertexcolor;
  // vertex 7 = grid(2,2): fSn=1 -> renorm fr=[Se 0, Sn 1]; vertex 2 = grid(1,0): fSn=0.25 -> [0.75,0.25]
  assert.equal(vc[7], expectRGB([0, 1], [SE_COL, SN_COL]), "pure-Sn vertex matches independent recompute");
  assert.equal(vc[2], expectRGB([0.75, 0.25], [SE_COL, SN_COL]), "mixed vertex matches (pairing + renorm)");
});

test("75: E2 numeric blend probe (REAL payload) — Γ-acoustic w_Sn ≈ mass ratio, vertexcolor matches recompute", { skip: !existsSync(REAL_NAV) }, () => {
  const D = JSON.parse(readFileSync(REAL_NAV, "utf-8"));
  const { window, state, threw } = loadPage({ navData: D });
  assert.equal(threw, null, threw && threw.stack);
  // real SnSe2 is binary -> defaults to projection; switch to blend for vertexcolor
  const selR = window.document.getElementById("surf-colormode");
  selR.value = "blend"; selR.dispatchEvent(new window.Event("change"));
  const S = D.surface, SPO = S.species_order, kx = S.kx, ky = S.ky;
  const sn = SPO.indexOf("Sn"), se = SPO.indexOf("Se");
  const M = D.meta.material, style = M.species_style;
  const snCol = style.Sn.color, seCol = style.Se.color;
  // masses (payload order Sn,Se,Se): acoustic-Γ mass-weighted eigenvectors give
  // |e_a|² ∝ m_a, so w_Sn = m_Sn/(m_Sn+2 m_Se) (Dove eq. 4.16, rigid-translation limit)
  const mass = {}; M.symbols.forEach((s, a) => { mass[s] = M.masses[a]; });
  const wSnMass = mass.Sn / (mass.Sn + 2 * mass.Se);
  // vertex nearest Γ=(0,0) on branch 0 (acoustic): scan the non-null grid
  const z0 = S.bands[0].z; let bi = -1, bj = -1, bd = Infinity;
  for (let i = 0; i < ky.length; i++) for (let j = 0; j < kx.length; j++) {
    if (z0[i][j] == null) continue;
    const d = kx[j]*kx[j] + ky[i]*ky[i];
    if (d < bd) { bd = d; bi = i; bj = j; }
  }
  const wSn = S.bands[0].weights[sn][bi][bj], wSe = S.bands[0].weights[se][bi][bj];
  assert.ok(Math.abs(wSn - wSnMass) < 0.03,
    "Γ-acoustic w_Sn " + wSn.toFixed(4) + " ≈ mass ratio " + wSnMass.toFixed(4) + " (Dove 4.16)");
  // and the built (blend) vertexcolor at that vertex == independent recompute from the payload
  const vidx = surfData(state)[0].vertexcolor;
  // map (bi,bj) -> vertex index: count non-null cells before it in row-major order
  let vpos = 0;
  outer: for (let i = 0; i < ky.length; i++) for (let j = 0; j < kx.length; j++) {
    if (i === bi && j === bj) break outer;
    if (z0[i][j] != null) vpos++;
  }
  const tot = wSn + wSe, fr = SPO.map((s, k) => (S.bands[0].weights[k][bi][bj] || 0) / (tot || 1));
  const cols = SPO.map(s => style[s].color);
  assert.equal(vidx[vpos], expectRGB(fr, cols),
    "real-payload vertexcolor matches independent (species_order-paired, renormalized) recompute");
});

test("76: E2 full-rebuild STATE CARRY-OVER — hidden IBZ + hidden branch 3 + a dropped marker all survive the toggle", () => {
  const { window, state, threw } = loadWeightsBlend();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, nB = 9;
  const IBZ_MESH = nB + 3;
  // (a) hide the IBZ zone
  const ibz = doc.getElementById("t-ibz"); ibz.checked = false; ibz.dispatchEvent(new window.Event("change"));
  // (b) hide branch 3
  const box3 = Array.from(doc.querySelectorAll("#branches input[type=checkbox]"))
    .find(cb => cb.dataset && cb.dataset.b === "3");
  box3.checked = false; box3.dispatchEvent(new window.Event("change"));
  // (c) drop a marker (select a mode; PICK/DROP get positioned + carried in surfPick)
  window.__atoms && window.__atoms.select ? window.__atoms.select(1, 0) : null;
  const before = surfData(state);
  const pickBefore = before[nB + 6];   // PICK trace has a position now
  assert.ok(pickBefore.x.length >= 1, "a marker was dropped (PICK populated)");
  // toggle -> full rebuild
  const sel = doc.getElementById("surf-colormode");
  sel.value = "projection"; sel.dispatchEvent(new window.Event("change"));
  const sp = surfData(state);
  assert.equal(sp[3].visible, false, "branch 3 stays hidden through the rebuild");
  for (const b of [0, 1, 2, 4, 8]) assert.equal(sp[b].visible, true, "other branches visible (" + b + ")");
  assert.equal(sp[IBZ_MESH].visible, false, "IBZ prism stays hidden through the rebuild");
  assert.equal(sp[nB].visible, true, "BZ prism stays visible (not hidden)");
  const pickAfter = sp[nB + 6];
  assert.deepEqual(plain(pickAfter.x), plain(pickBefore.x), "dropped marker x survives the rebuild");
  assert.deepEqual(plain(pickAfter.z), plain(pickBefore.z), "dropped marker z survives the rebuild");
});

test("76b: E2 full-rebuild CAMERA carry-over — the user-orbited camera is captured and re-injected into the rebuilt layout", () => {
  const { window, state, threw } = loadWeightsBlend();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // simulate a user orbit: real Plotly writes the live camera to gd._fullLayout.scene.camera
  const ORBIT = { eye: { x: 2.7, y: -0.4, z: 0.9 }, projection: { type: "orthographic" } };
  doc.getElementById("surf3d")._fullLayout = { scene: { camera: ORBIT } };
  const sel = doc.getElementById("surf-colormode");
  sel.value = "projection"; sel.dispatchEvent(new window.Event("change"));
  const layout = state.newPlot.find(p => p.id === "surf3d").layout;   // the rebuilt layout
  assert.deepEqual(plain(layout.scene.camera), plain(ORBIT), "rebuilt layout carries the captured (orbited) camera");
});

test("76c: E2 full-rebuild FULL-ARRAY IDENTITY — post-toggle gd.data deep-equals a fresh init build in the same mode", () => {
  // toggle a blend-started page to projection; compare to a page that INITS in
  // projection (binary default). Same code path -> identical full trace array.
  const p1 = loadWeightsBlend();
  const sel = p1.window.document.getElementById("surf-colormode");
  sel.value = "projection"; sel.dispatchEvent(new p1.window.Event("change"));
  const toggled = plain(surfData(p1.state));                          // whole array -> test realm
  const p2 = loadPage({ navData: synthNavDataWeights() });            // inits in projection
  const fresh = plain(surfData(p2.state));                            // (p1/p2 are distinct jsdom realms)
  assert.equal(toggled.length, fresh.length, "same trace count");
  assert.deepEqual(toggled.map(t => t.type), fresh.map(t => t.type), "same type sequence (mesh3d x nB, then prisms/labels/marker)");
  // exclude the PICK/DROP marker slots (nB+6, nB+7): those hold RUNTIME selection
  // state (the default-on-load marker is carried into the rebuilt plot's data but
  // not into a fresh init's data until it too rebuilds) — carry-over is test 76.
  const nB = 9, skip = new Set([nB + 6, nB + 7]);
  const struct = a => a.filter((_, i) => !skip.has(i));
  assert.deepEqual(struct(toggled), struct(fresh),
    "the mode-dependent structure (branches + zones + labels) deep-equals a fresh init in the same mode");
});

test("77: E2 defaults — a BINARY (nS==2) payload -> 3D Projection AND 2D Species radio", () => {
  const { window, state, threw } = loadPage({ navData: synthNavDataWeights() });   // species_order length 2
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  assert.equal(doc.getElementById("surf-colormode").value, "projection", "3D select defaults to Projection for a binary");
  // and the initial 3D build is the projection shape (intensity, no vertexcolor)
  assert.ok(!("vertexcolor" in surfData(state)[0]) && Array.isArray(surfData(state)[0].intensity),
    "initial 3D traces are projection-shaped");
  assert.ok(doc.getElementById("a-cmode-species").checked, "2D radio defaults to Species");
  assert.ok(!doc.getElementById("a-cmode-blend").checked, "2D Atom-blend not the default");
  // both controls remain independent: switching 3D to blend does not touch the 2D radio
  const sel = doc.getElementById("surf-colormode");
  sel.value = "blend"; sel.dispatchEvent(new window.Event("change"));
  assert.ok(doc.getElementById("a-cmode-species").checked, "2D radio unaffected by the 3D toggle");
});

test("78: E2 defaults — a TERNARY (nS==3) payload -> 3D Blend AND 2D Atom-blend radio", () => {
  const nav = synthNavDataWeights();
  // promote to 3 species: add a 3rd species column + relabel one atom
  nav.meta.material.symbols = ["Sn", "Se", "Te"];
  nav.meta.material.species_style.Te = { color: [0.5, 0.2, 0.6], radius: 1.4 };
  nav.surface.species_order = ["Sn", "Se", "Te"];
  nav.surface.bands.forEach(bd => {
    bd.weights = bd.weights.concat([bd.weights[0].map(row => row.map(v => v == null ? null : 0))]);  // Te weight 0
  });
  const { window, state, threw } = loadPage({ navData: nav });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  assert.equal(doc.getElementById("surf-colormode").value, "blend", "3D select defaults to Blend for nS>=3");
  assert.ok(Array.isArray(surfData(state)[0].vertexcolor) && !("intensity" in surfData(state)[0]),
    "initial 3D traces are blend-shaped");
  assert.ok(doc.getElementById("a-cmode-blend").checked, "2D radio defaults to Atom blend");
  assert.ok(!doc.getElementById("a-cmode-species").checked, "2D Species not the default for a ternary");
});

// --- 0.6.76 absolute-Å display geometry + side-car radius_A ---
// Cross-tool convention: ball display radius = 0.34 × Cordero covalent (the
// Blender extension's value — Sn 0.47 Å, Se 0.41 Å); cylinder radii are
// absolute Å with DEFAULT_BOND_RADIUS_A = 0.15 Å (the 0.6.75 visual default,
// preserved). The side-car gains radius_A (authoritative); radius_scale stays
// as the DERIVED legacy key (= radius_A / 0.15) for 0.7.9-Blender consumers.

test("79: Å round-trip — a 0.30 Å pair exports radius_A 0.30 + derived radius_scale 2, and re-imports to 0.30 Å", () => {
  const { window, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, A = window.__atoms;
  const inpByTitle = t => Array.from(doc.querySelectorAll("#bonds-legend input")).find(i => i.title === t);
  const radInp = inpByTitle("bond cylinder radius (Å): Se" + EN_DASH + "Sn");
  assert.ok(radInp, "Se–Sn radius input present");
  radInp.value = "0.30"; radInp.dispatchEvent(new window.Event("input"));
  const S = A.serializeBondSettings();
  assert.equal(S.bonds["Se-Sn"].radius_A, 0.30, "export: radius_A is the absolute Å value");
  assert.equal(S.bonds["Se-Sn"].radius_scale, 2, "export: radius_scale = radius_A / 0.15 (derived)");
  // reset -> back to defaults; import the snapshot -> 0.30 Å restored
  doc.getElementById("a-bondreset").dispatchEvent(new window.Event("click"));
  assert.ok(Math.abs(A.bonds().rad[0] - 0.15) < 1e-9, "reset returns bonds to the 0.15 Å default");
  assert.equal(A.applyBondSettings(S), true);
  const rad = A.bonds().rad;
  for (let b = 0; b < rad.length; b++)
    assert.ok(Math.abs(rad[b] - 0.30) < 1e-9, "re-import restores 0.30 Å on bond " + b);
  const inp2 = inpByTitle("bond cylinder radius (Å): Se" + EN_DASH + "Sn");
  assert.equal(inp2.value, "0.3", "the rebuilt per-pair input displays the imported Å value");
});

test("80: legacy import — radius_scale-only applies scale × 0.15 Å; radius_A WINS when both present", () => {
  const p1 = loadPage(); assert.equal(p1.threw, null, p1.threw && p1.threw.stack);
  const A1 = p1.window.__atoms;
  // (a) a 0.7.9-era file with ONLY radius_scale
  const legacy = { format: "phonon-vision-bonds", version: 1,
                   bonds: { "Se-Sn": { max: 3.11, radius_scale: 2.0 } } };
  assert.equal(A1.applyBondSettings(legacy), true);
  let rad = A1.bonds().rad;
  assert.ok(rad.length > 0, "legacy import must keep the Se–Sn bonds");
  for (let b = 0; b < rad.length; b++)
    assert.ok(Math.abs(rad[b] - 0.30) < 1e-9, "legacy radius_scale 2.0 -> 2.0 × 0.15 = 0.30 Å");
  // (b) BOTH keys: radius_A is authoritative (a file that disagrees on purpose)
  const p2 = loadPage(); assert.equal(p2.threw, null, p2.threw && p2.threw.stack);
  const A2 = p2.window.__atoms;
  const both = { format: "phonon-vision-bonds", version: 1,
                 bonds: { "Se-Sn": { max: 3.11, radius_A: 0.25, radius_scale: 3.0 } } };
  assert.equal(A2.applyBondSettings(both), true);
  rad = A2.bonds().rad;
  for (let b = 0; b < rad.length; b++)
    assert.ok(Math.abs(rad[b] - 0.25) < 1e-9,
      "radius_A (0.25 Å) must win over radius_scale (which would give 0.45 Å)");
});

test("81: ball convention pin — per-species defaults are round2(0.34 × Cordero): Sn 0.47 Å, Se 0.41 Å", () => {
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, A = window.__atoms;
  const style = navData.meta.material.species_style;
  // the pin: derive from the PAYLOAD's Cordero radii with the shared constant
  const wantSn = Math.round(0.34 * style.Sn.radius * 100) / 100;   // 0.47
  const wantSe = Math.round(0.34 * style.Se.radius * 100) / 100;   // 0.41
  assert.equal(wantSn, 0.47, "0.34 × 1.39 rounds to 0.47 Å (the Blender ball value)");
  assert.equal(wantSe, 0.41, "0.34 × 1.20 rounds to 0.41 Å (the Blender ball value)");
  const inputs = Array.from(doc.querySelectorAll("#atoms-legend input[type=number]"));
  assert.equal(inputs.find(r => r.title === "display radius (Å): Sn").value, "0.47");
  assert.equal(inputs.find(r => r.title === "display radius (Å): Se").value, "0.41");
  // and the RENDERED per-slot display radii equal those values (scene units are Å)
  const symbols = navData.meta.material.symbols;
  const slots = A.getSlots(), rad = A.getRad();
  for (let k = 0; k < rad.length; k++) {
    const want = symbols[slots.j[k]] === "Sn" ? wantSn : wantSe;
    assert.ok(Math.abs(rad[k] - want) < 1e-12,
      "slot " + k + " (" + symbols[slots.j[k]] + ") display radius must be " + want + " Å");
  }
});

test("82: default-view pin — cylinders stay at the 0.6.75 default (0.15 Å); spheres sit at the sanctioned 0.34 × Cordero", () => {
  // Architect ruling (0.6.76, option A): the BOND default is invariant vs
  // 0.6.75 (0.15 Å); the BALL default deliberately converges to the Blender
  // convention (0.47/0.41 Å, ~32% smaller than 0.6.75's 0.5 × Cordero) so the
  // web view predicts the Blender render. This test pins BOTH.
  const { window, navData, threw } = loadPage();
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document, A = window.__atoms;
  assert.equal(doc.getElementById("a-bondr").value, "0.15", "global bond radius default is 0.15 Å");
  const rad = A.bonds().rad;
  assert.ok(rad.length > 0, "synth must have bonds");
  for (let b = 0; b < rad.length; b++)
    assert.ok(Math.abs(rad[b] - 0.15) < 1e-9, "default cylinder radius must be 0.15 Å (bond " + b + ")");
  const style = navData.meta.material.species_style, symbols = navData.meta.material.symbols;
  const slots = A.getSlots(), arad = A.getRad();
  for (let k = 0; k < arad.length; k++) {
    const want = Math.round(0.34 * style[symbols[slots.j[k]]].radius * 100) / 100;
    assert.ok(Math.abs(arad[k] - want) < 1e-12,
      "default sphere radius must be round2(0.34 × Cordero) for slot " + k);
  }
});

// ---- 0.6.77: imaginary-badge threshold parity + honest label (REAL payload) ----
// Mirrors the Blender add-on (state.py imag_freq_threshold): the badge fires iff
// ν<0 AND |ν|>thr, so SnSe2's Γ-acoustic numerical dust (ν≈−0.00248 THz, exactly
// zero by the acoustic sum rule) is NOT flagged an instability at the default
// 0.01; the ω readout stays sign-honest at every threshold, and the label no
// longer implies |ν| is a physical oscillation rate. Skips in a bare-zip audit
// (the real payload lives outside the src tree, exactly like test 75).
// 0.6.84: the bundled payload is engine-exact + ASR-fixed. Tests 83/84 no
// longer hardcode the artifact value (-0.00248 was the OLD payload's broken
// sum rule) — they READ the payload's Γ branch-0 ν and assert AS A
// PRECONDITION that it is negative dust (|ν| < 0.01); if a future payload
// regenerates with ν exactly 0 or positive, the precondition failure says
// "move this test", instead of a silent vacuous pass.
test("83: imaginary badge — default threshold gates the Γ-acoustic dust (REAL payload)",
     { skip: !existsSync(REAL_NAV) }, () => {
  const D = JSON.parse(readFileSync(REAL_NAV, "utf-8"));
  const nu = D.band.frequency[0][0];                     // Γ, branch 0
  assert.ok(nu < 0 && Math.abs(nu) < 0.01,
    "PRECONDITION: Γ branch-0 must be sub-threshold negative dust; got " + nu +
    " — payload changed, migrate this test");
  const { window, state, threw } = loadPage({ navData: D });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  // (a) the input exists with the Blender-mirroring default.
  assert.equal(doc.getElementById("a-imagthr").value, "0.01",
    "imaginary threshold input default must be 0.01 THz (Blender parity)");
  // (b) Γ acoustic (qi=0, branch=0) — sub-threshold, NOT badged.
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 0, pointNumber: 0 }] });
  assert.equal(doc.getElementById("a-badge").style.display, "none",
    "sub-threshold negative (Γ-acoustic dust) must NOT be badged at default 0.01");
  const info = doc.getElementById("a-info").textContent;
  // the expected prefix is DERIVED from the payload via the viewer's OWN
  // toFixed(3) formatting — never a hardcoded artifact value again.
  assert.ok(info.startsWith("ω = " + nu.toFixed(3) + " THz"),
    "ω readout is sign-honest even sub-threshold: " + info);
  assert.ok(!/imaginary/.test(info), "sub-threshold mode must not say 'imaginary': " + info);
});

test("84: imaginary badge — threshold 0 is a pure sign test with the honest label (REAL payload)",
     { skip: !existsSync(REAL_NAV) }, () => {
  const D = JSON.parse(readFileSync(REAL_NAV, "utf-8"));
  const nu = D.band.frequency[0][0];                     // Γ, branch 0
  assert.ok(nu < 0 && Math.abs(nu) < 0.01,
    "PRECONDITION: Γ branch-0 must be sub-threshold negative dust; got " + nu +
    " — payload changed, migrate this test");
  const { window, state, threw } = loadPage({ navData: D });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 0, pointNumber: 0 }] });  // Γ acoustic
  // set threshold 0 -> any ν<0 is flagged (pure sign test); re-gate via "input".
  const thr = doc.getElementById("a-imagthr");
  thr.value = "0"; thr.dispatchEvent(new window.Event("input"));
  assert.equal(doc.getElementById("a-badge").style.display, "",
    "at threshold 0 the sub-threshold negative IS flagged (pure sign test)");
  const info = doc.getElementById("a-info").textContent;
  assert.ok(/instability direction/.test(info),
    "honest label names the instability DIRECTION: " + info);
  assert.ok(/oscillation rate not physical/.test(info),
    "honest label disclaims the oscillation rate: " + info);
  assert.ok(!/\|ω\| shown/.test(info),
    "the over-promising '|ω| shown' label must be gone: " + info);
});

// 0.6.84 MIGRATION: this case lived on the REAL payload's qi=30 ν=−0.118 THz —
// which the ASR-fixed regeneration PROVED to be a sum-rule artifact (SnSe2 is
// dynamically stable; the engine-exact payload has ZERO modes < −0.01 THz, so
// the old test correctly FAILED against it — transcript in the 0.6.84
// CHANGELOG). The above-threshold behavior now lives on the SYNTHETIC payload,
// whose branch 4 carries −0.3 THz by construction; the honest-label
// assertions are kept verbatim.
test("85: imaginary badge — above-threshold mode flagged with the honest label (synthetic)", () => {
  const { window, state, threw } = loadPage();          // synthetic: branch 4 = −0.3 THz
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 4, pointNumber: 0 }] });
  assert.equal(doc.getElementById("a-imagthr").value, "0.01", "default threshold unchanged");
  assert.equal(doc.getElementById("a-badge").style.display, "",
    "above-threshold negative must be badged imaginary");
  const info = doc.getElementById("a-info").textContent;
  assert.ok(info.startsWith("ω = -0.300 THz"),
    "ω readout is sign-honest for the instability: " + info);
  assert.ok(/instability direction/.test(info), "honest label present: " + info);
  assert.ok(!/\|ω\| shown/.test(info), "no over-promising '|ω| shown': " + info);
});

// ---- 0.6.82 Stage 3: mesh_modes viewer wiring (click-anywhere) --------------
// Synthetic engine payload: the standard synthetic + a mesh_modes block
// (6 in-BZ nodes, the same 9 branches / 3 atoms). Node 0 branch 0 is a strong
// imaginary mode (-0.5 THz); node 1 branch 0 is numerical dust (-0.005 THz) —
// the 0.6.77 threshold cases, now reached through a mesh click.
function synthNavDataMeshModes() {
  const d = synthNavData();
  const Nn = 6, Nb = 9, Na = 3;
  const q_cart = [], q_frac = [], fr = [], er = [], ei = [];
  for (let n = 0; n < Nn; n++) {
    q_cart.push([0.01 * n - 0.025, 0.005 * n - 0.0125]);   // inside the hex BZ
    q_frac.push([0.05 * n, 0.02 * n, 0]);
    const frn = [], ern = [], ein = [];
    for (let b = 0; b < Nb; b++) {
      frn.push(b + 0.1 * n + 0.5);
      const erb = [], eib = [];
      for (let a = 0; a < Na; a++) {
        erb.push([0.1 * (a + 1) + 0.01 * n, 0.2, 0.0]);
        eib.push([0.0, 0.0, 0.05 * (b + 1)]);
      }
      ern.push(erb); ein.push(eib);
    }
    fr.push(frn); er.push(ern); ei.push(ein);
  }
  fr[0][0] = -0.5;     // above-threshold imaginary -> badge via mesh click
  fr[1][0] = -0.005;   // sub-threshold dust -> NO badge at the default 0.01
  d.mesh_modes = { grid: [4, 4], q_frac: q_frac, q_cart: q_cart, frequency: fr,
                   eigenvectors_real: er, eigenvectors_imag: ei };
  return d;
}
const NB_SYN = 9;                 // synthetic branch count (see synthNavData)
const MM0_SYN = NB_SYN + 9;       // first pick layer: after the 9 fixed traces

test("86: LEGACY invariance — no mesh_modes -> no pick traces, trace count/indices unchanged", () => {
  const { state } = loadPage();                       // standard synthetic payload
  const gd = state.newPlot.find(p => p.id === "surf3d");
  assert.equal(gd.data.length, NB_SYN + 10,
    "legacy surf3d must keep exactly nB+10 traces (no pick layers; incl. the 0.7.2 placeholder)");
  assert.ok(!gd.data.some(t => /mesh modes/.test(t.name || "")),
    "no mesh-modes traces on a legacy payload");
  assert.equal(gd.data[NB_SYN + 6].name, "selected mode",
    "PICK diamond still at nB+6");
});

test("87: mesh_modes pick layers — one per branch at the right xyz; branch toggle + marker regression", () => {
  const navData = synthNavDataMeshModes();
  const { window, state } = loadPage({ navData });
  const doc = window.document;
  const gd = state.newPlot.find(p => p.id === "surf3d");
  // (i) one pick layer per branch, appended LAST
  assert.equal(gd.data.length, NB_SYN + 9 + NB_SYN + 1, "nB extra pick traces + the 0.7.2 placeholder");
  for (let b = 0; b < NB_SYN; b++) {
    const t = gd.data[MM0_SYN + b];
    assert.equal(t.mode, "markers", `pick layer ${b} is a marker trace`);
    assert.ok(!("colorscale" in (t.marker || {})) && !("colorscale" in t),
      "pick layers carry NO colorscale (CHAR_SCALE guard)");
    assert.equal(t.x[0], navData.mesh_modes.q_cart[0][0]);
    assert.equal(t.y[2], navData.mesh_modes.q_cart[2][1]);
    assert.equal(t.z[3], navData.mesh_modes.frequency[3][b]);
  }
  // (ii) existing indices unchanged with pick layers present
  assert.equal(gd.data[NB_SYN + 6].name, "selected mode");
  // (iii) branch checkbox toggles surface AND pick layer together
  const box0 = doc.querySelectorAll("#branches input")[0];
  box0.checked = false;
  box0.dispatchEvent(new window.Event("change"));
  const surfR = state.restyle.filter(r => r.id === "surf3d" && r.update
                                     && "visible" in r.update);
  const branchIdx = surfR.find(r => Array.isArray(r.idx) && r.idx.includes(0));
  const pickIdx = surfR.find(r => Array.isArray(r.idx) && r.idx.includes(MM0_SYN));
  assert.ok(branchIdx && branchIdx.update.visible[0] === false,
    "surface trace 0 hidden");
  assert.ok(pickIdx && pickIdx.update.visible[0] === false,
    "pick layer 0 hidden by the SAME checkbox");
  // (iv) marker-restyle regression: a 2D click still targets the PICK diamond
  state.restyle.length = 0;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const pickR = state.restyle.find(r => r.id === "surf3d"
    && Array.isArray(r.idx) && r.idx.length === 1 && r.idx[0] === NB_SYN + 6);
  assert.ok(pickR, "PICK diamond restyle still targets trace nB+6");
});

test("88: mesh click -> honest readout (q r.l.u., mesh-node tag) + 2D marker cleared", async () => {
  const navData = synthNavDataMeshModes();
  const { window, state } = loadPage({ navData });
  const doc = window.document;
  state.restyle.length = 0;
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0_SYN + 2,
                                               pointNumber: 3 }] }); await flushClick();
  const info = doc.getElementById("a-info").textContent;
  assert.ok(/mesh node \(exact engine mode\)/.test(info), "mesh-node tag: " + info);
  assert.ok(/r\.l\.u\./.test(info), "r.l.u. units named: " + info);
  const qf = navData.mesh_modes.q_frac[3];
  assert.ok(info.includes("q = (" + qf.map(v => v.toFixed(3)).join(", ") + ")"),
    "snapped q shown honestly: " + info);
  assert.ok(info.includes("ω = " + navData.mesh_modes.frequency[3][2].toFixed(3)),
    "mesh ω through the existing readout: " + info);
  // the 2D selection marker is CLEARED (a mesh node is off the band path)
  const clr = state.restyle.find(r => r.id === "band2d"
    && Array.isArray(r.idx) && r.idx[0] === NB_SYN + 1
    && r.update && Array.isArray(r.update.x) && r.update.x[0][0] === null);
  assert.ok(clr, "PICK2D cleared, not faked");
  // status names the mesh node
  assert.ok(/mesh node/.test(doc.getElementById("status").innerHTML));
});

test("89: mesh click feeds the payload's EXACT eigenvector to kinematics", async () => {
  const navData = synthNavDataMeshModes();
  const { state, dmCalls } = loadPage({ navData });
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0_SYN + 2,
                                               pointNumber: 3 }] }); await flushClick();
  const last = dmCalls[dmCalls.length - 1];
  assert.equal(last[0], navData.mesh_modes.eigenvectors_real[3][2],
    "displaceMode receives the payload eigenvector (real) BY IDENTITY");
  assert.equal(last[1], navData.mesh_modes.eigenvectors_imag[3][2],
    "…and (imag)");
  assert.equal(last[4], navData.mesh_modes.q_frac[3],
    "…at the node's own fractional q");
});

test("90: imaginary badge on mesh modes — 0.6.77 threshold applies for free", async () => {
  const navData = synthNavDataMeshModes();
  const { window, state } = loadPage({ navData });
  const doc = window.document;
  // node 0 branch 0: -0.5 THz, above the default 0.01 threshold -> badged
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0_SYN + 0,
                                               pointNumber: 0 }] }); await flushClick();
  assert.equal(doc.getElementById("a-badge").style.display, "");
  const info = doc.getElementById("a-info").textContent;
  assert.ok(/instability direction/.test(info), "honest 0.6.77 label: " + info);
  assert.ok(/mesh node/.test(info), "…on a mesh mode: " + info);
  // node 1 branch 0: -0.005 THz numerical dust -> NOT badged at default 0.01
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0_SYN + 0,
                                               pointNumber: 1 }] }); await flushClick();
  assert.equal(doc.getElementById("a-badge").style.display, "none");
});

test("91: handover — a 2D click after a mesh click restores band-mode readout", async () => {
  const navData = synthNavDataMeshModes();
  const { window, state } = loadPage({ navData });
  const doc = window.document;
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0_SYN + 1,
                                               pointNumber: 2 }] }); await flushClick();
  assert.ok(/mesh node/.test(doc.getElementById("a-info").textContent));
  state.restyle.length = 0;
  state.cb["band2d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 1 }] });
  const info = doc.getElementById("a-info").textContent;
  assert.ok(!/mesh node/.test(info), "band mode: mesh tag gone: " + info);
  assert.ok(info.startsWith("ω = "), "normal band readout restored: " + info);
  const mark = state.restyle.find(r => r.id === "band2d"
    && Array.isArray(r.idx) && r.idx[0] === NB_SYN + 1
    && r.update && r.update.x && r.update.x[0][0] !== null);
  assert.ok(mark, "PICK2D restored to a real band-path position");
});

// ---- 0.6.84: the BUNDLED payload is engine-exact — click-anywhere is LIVE ----
test("92: bundled payload carries mesh_modes — pick layers, header hint, mesh readout (REAL payload)",
     { skip: !existsSync(REAL_NAV) }, async () => {
  const D = JSON.parse(readFileSync(REAL_NAV, "utf-8"));
  assert.ok(D.mesh_modes && D.provenance,
    "PRECONDITION: the bundled payload must be the engine-exact build");
  const nB = D.band.frequency[0].length;                 // 9 for SnSe2
  const { window, state, threw } = loadPage({ navData: D });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const gd = state.newPlot.find(p => p.id === "surf3d");
  // exactly one pick layer per branch, occupying the LAST nB trace slots
  assert.equal(gd.data.length, nB + 9 + nB + 1, "sheets + 9 fixed + nB pick layers + the 0.7.2 placeholder");
  for (let b = 0; b < nB; b++) {
    // pick layers sit at MM0..MM0+nB-1; the 0.7.2 zone-tiling placeholder
    // is the ONE trace after them
    const t = gd.data[gd.data.length - 1 - nB + b];
    assert.equal(t.mode, "markers", "pick layer " + b + " before the placeholder");
    assert.equal(t.x.length, D.mesh_modes.q_cart.length, "one marker per node");
  }
  assert.equal(gd.data[gd.data.length - 1].name, "zone tiling",
    "the trailing trace is the 0.7.2 placeholder");
  // header: the presence-keyed clickable-nodes sentence is revealed
  const hint = doc.getElementById("mm-hint");
  assert.ok(hint && hint.style.display !== "none", "mm-hint revealed");
  assert.ok(/clickable/.test(hint.textContent), "header names clickable nodes");
  // a simulated pick click shows the honest mesh readout
  const MM0 = nB + 9;
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0 + 3, pointNumber: 7 }] }); await flushClick();
  const info = doc.getElementById("a-info").textContent;
  assert.ok(/mesh node \(exact engine mode\)/.test(info),
    "mesh-node readout live on the bundled payload: " + info);
});

// ---------------------------------------------------------------------------
// 0.6.92 — compact surface storage: the load shim
// ---------------------------------------------------------------------------

// dense/compact twins: same physics, two storage shapes. The compact twin
// stores each band's four in-BZ nodes flat and indexes them via node_map
// (with a c2-style dedup on band-constant fields: 4 grid cells -> 2 nodes).
function synthDenseAndCompact() {
  const dense = synthNavData();                 // 2x2 grids, 9 bands, no weights
  const compact = JSON.parse(JSON.stringify(dense));
  compact.surface = {
    kx: dense.surface.kx, ky: dense.surface.ky,
    node_map: [[0, 1], [1, 0]],                 // c2 orbit: (0,0)~(1,1), (0,1)~(1,0)
    n_nodes: 2,
    sym: { ops: ["c2"], max_dz_THz: 0, max_dchar: 0, nodes_total: 4, nodes_stored: 2 },
    bands: dense.surface.bands.map(bd => ({
      z: [bd.z[0][0], bd.z[0][1]],              // node 0 = (0,0), node 1 = (0,1)
      character: [bd.character[0][0], bd.character[0][1]],
    })),
  };
  return { dense, compact };
}

test("96: 0.6.92 shim — compact surface expands to the dense grids in-place", () => {
  const { dense, compact } = synthDenseAndCompact();
  const res = loadPage({ navData: compact });
  assert.equal(res.threw, null, res.threw && res.threw.stack);
  const S = res.navData.surface;
  assert.ok(!("node_map" in S), "node_map consumed by the shim");
  assert.ok(!("n_nodes" in S), "n_nodes consumed by the shim");
  for (let b = 0; b < dense.surface.bands.length; b++) {
    assert.deepEqual(S.bands[b].z, dense.surface.bands[b].z,
      "band " + b + " z grid reconstructed exactly");
    assert.deepEqual(S.bands[b].character, dense.surface.bands[b].character);
  }
});

test("97: 0.6.92 shim — surf3d traces identical between dense and compact boots", () => {
  const a = loadPage({ navData: synthDenseAndCompact().dense });
  const b = loadPage({ navData: synthDenseAndCompact().compact });
  const surfA = a.state.newPlot.find(p => p.id === "surf3d");
  const surfB = b.state.newPlot.find(p => p.id === "surf3d");
  assert.ok(surfA && surfB, "both boots draw the 3D panel");
  assert.equal(surfA.data.length, surfB.data.length, "same trace count");
  for (let t = 0; t < surfA.data.length; t++) {
    // JSON equality: the traces come from two separate JSDOM realms, where
    // node's deepEqual can reject content-identical arrays; the serialized
    // form IS the contract here.
    assert.equal(JSON.stringify(surfB.data[t].z), JSON.stringify(surfA.data[t].z),
      "trace " + t + " z identical across storage shapes");
  }
});

test("98: 0.6.92 shim — node_map -1 (outside BZ) expands to null and renders as a gap", () => {
  const { compact } = synthDenseAndCompact();
  compact.surface.node_map = [[0, 1], [1, -1]];      // knock out cell (1,1)
  const res = loadPage({ navData: compact });
  assert.equal(res.threw, null, res.threw && res.threw.stack);
  const S = res.navData.surface;
  assert.equal(S.bands[0].z[1][1], null, "outside-BZ cell is null after expansion");
  assert.notEqual(S.bands[0].z[0][0], null, "in-BZ cells keep their values");
});

test("99: 0.6.92 shim — dense (pre-0.6.92) payloads pass through byte-identical", () => {
  const dense = synthDenseAndCompact().dense;
  const before = JSON.stringify(dense.surface);
  const res = loadPage({ navData: dense });
  assert.equal(res.threw, null);
  assert.equal(JSON.stringify(res.navData.surface), before,
    "no node_map -> the shim must not touch the block");
});

// ---------------------------------------------------------------------------
// 0.6.93 — fast marker move (gl object mutation instead of scene replot)
// ---------------------------------------------------------------------------

test("100: 0.6.93 — PICK/DROP boot PARKED (clipped, honest), never empty", () => {
  const { state } = loadPage({ navData: synthDenseAndCompact().compact });
  const surf = state.newPlot.find(p => p.id === "surf3d");
  const pick = surf.data.find(t => t.name === "selected mode");
  const drop = surf.data.find(t => t.name === "drop");
  assert.ok(pick && drop, "pick + drop traces exist at boot");
  assert.equal(pick.x.length, 1, "parked, not empty (gl object must exist)");
  assert.ok(pick.x[0] >= 1e6, "parked OUTSIDE the fixed x-range (gl-clipped => invisible)");
  assert.ok(drop.x.every(v => v >= 1e6), "drop line parked too");
});

function compactWithMeshModes() {
  const d = synthDenseAndCompact().compact;
  const Nb = d.surface.bands.length, Na = 3;
  const eig = Array.from({ length: Nb }, (_, b) =>
    Array.from({ length: Na }, () => [0.1 * (b + 1), 0.05, 0]));
  d.mesh_modes = {
    grid: [2, 2],
    q_frac: [[0, 0, 0], [0.25, 0.1, 0]],
    q_cart: [[0, 0], [0.05, 0.02]],
    frequency: [Array.from({ length: Nb }, (_, b) => b + 0.5),
                Array.from({ length: Nb }, (_, b) => b + 0.7)],
    eigenvectors_real: [eig, eig],
    eigenvectors_imag: [eig, eig],
  };
  return d;
}

test("101: 0.6.93 — fast path: gl objects move via dataScale, no surf3d restyle", async () => {
  const { window, state, navData } = loadPage({ navData: compactWithMeshModes() });
  const nB = navData.surface.bands.length;
  const PICK = nB + 6, DROP = nB + 7, MM0 = nB + 9;
  const gd = window.document.getElementById("surf3d");
  const surf = state.newPlot.find(p => p.id === "surf3d");
  // fake the PINNED vendored internals the fast path probes. The fakes carry
  // marker/line styling like real fullData traces so the FULL-TRACE contract
  // below is a real assertion (the 0.6.94 fix for the 0.6.93 field defect).
  const calls = { pick: [], drop: [], redraw: 0 };
  gd.data = surf.data;
  gd._fullData = surf.data.map((t, i) => Object.assign({ uid: "u" + i }, t));
  gd._fullLayout = { scene: { _scene: {
    dataScale: [2, 3, 4],
    glplot: { redraw: () => { calls.redraw++; } },
    traces: {
      ["u" + PICK]: { scatterPlot: {}, update: o => calls.pick.push(o) },
      ["u" + DROP]: { linePlot: {}, update: o => calls.drop.push(o) },
    },
  } } };
  const before = state.restyle.filter(r => r.id === "surf3d").length;
  // a mesh-node click drives selectMeshMode -> moveSurfMarkers
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0 + 3, pointNumber: 1 }] }); await flushClick();
  assert.equal(calls.pick.length, 1, "pick WRAPPER updated once");
  assert.equal(calls.drop.length, 1, "drop WRAPPER updated once");
  assert.equal(calls.redraw, 2, "GL canvas redrawn (once per moved object)");
  const MM = navData.mesh_modes, qc = MM.q_cart[1];
  const w = MM.frequency[1][3];
  // THE 0.6.94 CONTRACT (fail-then-pass vs the 0.6.93 defect): the wrapper
  // must receive the FULL trace — new coords AND the marker styling — never
  // a position-only object, which is what degraded the diamond to a default
  // dot in the field. A {position: ...} argument fails all three asserts.
  const arg = calls.pick[0];
  assert.equal(JSON.stringify(arg.x), JSON.stringify([qc[0]]), "full trace: new x");
  assert.equal(JSON.stringify(arg.z), JSON.stringify([w]), "full trace: new z");
  assert.ok(arg.marker && arg.marker.symbol === "diamond",
    "full trace: marker styling present (position-only = the 0.6.93 defect)");
  assert.ok(calls.drop[0].line, "drop keeps its line styling too");
  assert.equal(state.restyle.filter(r => r.id === "surf3d").length, before,
    "NO surf3d restyle on the fast path (that replot was the freeze)");
  assert.equal(JSON.stringify(gd.data[PICK].x), JSON.stringify([qc[0]]),
    "gd.data kept in sync for later replots");
});

test("102: 0.6.93 — fallback: without gl internals the slow restyle still runs (correctness)", async () => {
  const { state, navData } = loadPage({ navData: compactWithMeshModes() });
  const nB = navData.surface.bands.length;
  const MM0 = nB + 9;
  // jsdom's mock plotly attaches NO internals -> fastMove3d declines -> restyle
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: MM0, pointNumber: 0 }] }); await flushClick();
  const pickRestyles = state.restyle.filter(r => r.id === "surf3d" && r.update
      && r.update.z && r.update.z[0] && r.update.z[0].length === 1);
  assert.ok(pickRestyles.length >= 1, "PICK moved via restyle when the fast path is unavailable");
});

// ---------------------------------------------------------------------------
// 0.6.95 — sheet clicks snap to the nearest exact mesh node
// ---------------------------------------------------------------------------

test("103: 0.6.95 — clicking a band SHEET selects the nearest mesh node (exact mode, snapped q)", async () => {
  const { window, state, navData } = loadPage({ navData: compactWithMeshModes() });
  const doc = window.document;
  // click sheet (branch 2) at a k-point closest to mesh node 1 (q_cart [0.05, 0.02])
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: 2, pointNumber: 99, x: 0.049, y: 0.021 }] }); await flushClick();
  const info = doc.getElementById("a-info").textContent;
  assert.ok(/mesh node \(exact engine mode\)/.test(info), "snapped to an exact node: " + info);
  assert.ok(info.includes("q = (0.250, 0.100, 0.000)"),
    "the READOUT shows the SNAPPED node's own q, never the clicked k: " + info);
  const w = navData.mesh_modes.frequency[1][2];
  assert.ok(info.includes("ω = " + w.toFixed(3)), "branch = the clicked sheet: " + info);
});

test("104: 0.6.95 — sheet click snaps to a DIFFERENT node when closer; prisms stay inert", async () => {
  const { window, state, navData } = loadPage({ navData: compactWithMeshModes() });
  const doc = window.document;
  const nB = navData.surface.bands.length;
  // near node 0 (q_cart [0,0]) on sheet 0
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: 0, pointNumber: 5, x: 0.001, y: -0.002 }] }); await flushClick();
  const info = doc.getElementById("a-info").textContent;
  assert.ok(info.includes("q = (0.000, 0.000, 0.000)"), "nearest is node 0: " + info);
  // a BZ prism / floor-ring click (nB..nB+5) selects nothing
  const before = doc.getElementById("a-info").textContent;
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: nB + 1, pointNumber: 0, x: 0.05, y: 0.02 }] }); await flushClick();
  assert.equal(doc.getElementById("a-info").textContent, before, "prism clicks stay inert");
  // a malformed sheet click (no x/y) must not throw or select
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: 1, pointNumber: 0 }] }); await flushClick();
  assert.equal(doc.getElementById("a-info").textContent, before, "no coords -> no selection");
});

// ---------------------------------------------------------------------------
// 0.6.96 — 3D display options: plain mode, node toggle, template explainer
// ---------------------------------------------------------------------------

function latestSurfPlot(state) {
  const plots = state.newPlot.filter(p => p.id === "surf3d");
  return plots[plots.length - 1];
}

test("105: 0.6.96 plain mode (weights payload) — one lit translucent colour, no colormap keys", () => {
  const { window, state, navData } = loadPage({ navData: synthNavDataWeights() });
  const doc = window.document;
  const sel = doc.getElementById("surf-colormode");
  assert.ok([...sel.options].some(o => o.value === "plain"), "plain option offered");
  sel.value = "plain"; sel.dispatchEvent(new window.Event("change"));
  const nB = navData.surface.bands.length;
  const surf = latestSurfPlot(state);
  for (let b = 0; b < nB; b++) {
    const t = surf.data[b];
    assert.equal(t.type, "mesh3d", "band " + b + " stays mesh3d");
    assert.ok(t.color, "single colour present");
    assert.ok(!("vertexcolor" in t), "no vertexcolor key (key-presence discipline)");
    assert.ok(!("intensity" in t), "no intensity key (no colormap pipeline)");
    assert.ok(t.opacity < 1, "translucent");
    assert.ok(t.lighting.diffuse > 0, "REAL lighting carries the shape");
  }
});

test("106: 0.6.96 plain mode (weightless payload) — dropdown now shown, blend gone, const colorscale", () => {
  const { window, state, navData } = loadPage({ navData: synthDenseAndCompact().dense });
  const doc = window.document;
  const wrap = doc.getElementById("surf-colormode-wrap");
  assert.notEqual(wrap.style.display, "none", "dropdown visible for weightless payloads (0.6.96)");
  const sel = doc.getElementById("surf-colormode");
  const values = [...sel.options].map(o => o.value);
  assert.ok(!values.includes("blend"), "no Blend without weights");
  assert.ok(values.includes("plain") && values.includes("projection"), "Projection + Plain offered");
  sel.value = "plain"; sel.dispatchEvent(new window.Event("change"));
  const nB = navData.surface.bands.length;
  const surf = latestSurfPlot(state);
  for (let b = 0; b < nB; b++) {
    const t = surf.data[b];
    assert.equal(t.type, "surface", "weightless path keeps the surface type");
    assert.equal(t.colorscale[0][1], t.colorscale[1][1], "constant colorscale = one colour");
    assert.equal(t.showscale, false, "no colorbar in plain mode");
    assert.ok(t.opacity < 1 && t.lighting.diffuse > 0, "translucent + lit");
  }
});

test("107: 0.6.96 node dots — hidden by default, Nodes checkbox shows them", () => {
  const { window, state, navData } = loadPage({ navData: compactWithMeshModes() });
  const doc = window.document;
  const nB = navData.surface.bands.length, MM0 = nB + 9;
  const surf = state.newPlot.find(p => p.id === "surf3d");
  for (let b = 0; b < nB; b++)
    assert.equal(surf.data[MM0 + b].visible, false, "MM layer " + b + " hidden at boot");
  const wrap = doc.getElementById("surf-nodes-wrap");
  assert.notEqual(wrap.style.display, "none", "Nodes toggle offered (payload has mesh_modes)");
  const box = doc.getElementById("surf-nodes");
  assert.equal(box.checked, false, "unchecked by default");
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  const rs = state.restyle.filter(r => r.id === "surf3d" && r.update && r.update.visible);
  const last = rs[rs.length - 1];
  assert.ok(last, "toggle restyles MM visibility");
  assert.ok(last.update.visible.every(v => v === true), "all branches on -> all dots shown");
  assert.equal(JSON.stringify(last.idx), JSON.stringify(
    Array.from({ length: nB }, (_, b) => MM0 + b)), "exactly the MM layers targeted");
});

test("108: 0.6.96 bare template — explains itself instead of a blank page", () => {
  const { window, threw } = loadPage({ navData: "NONE" });
  assert.equal(threw, null, threw && threw.stack);
  const doc = window.document;
  const ex = doc.getElementById("no-payload-explainer");
  assert.ok(ex, "explainer panel injected");
  assert.ok(/navigator_data\.js/.test(ex.innerHTML), "names the missing file");
  assert.ok(/pv_navigator_export\.py/.test(ex.innerHTML), "shows the bake command");
  assert.ok(/pv_serve\.py/.test(ex.innerHTML), "mentions the launcher");
  assert.ok(/template/.test(doc.getElementById("status").textContent), "status says template");
});

// ---------------------------------------------------------------------------
// 0.6.97 — node dots: constant white + border (colormap overlay, not theme fg)
// ---------------------------------------------------------------------------

test("109: 0.6.97 node dots stay white with a dark border across plain->projection and theme toggles", () => {
  const { window, state, navData } = loadPage({ navData: (() => {
    const d = compactWithMeshModes();
    d.surface.species_order = undefined; delete d.surface.species_order;
    return d;
  })() });
  const doc = window.document;
  const nB = navData.surface.bands.length, MM0 = nB + 9;
  const dotColorOf = plot => plot.data[MM0].marker.color;
  const first = state.newPlot.filter(p => p.id === "surf3d").pop();
  assert.equal(dotColorOf(first), "#ffffff", "white at boot");
  assert.equal(first.data[MM0].marker.line.color, "#1a1a1a", "dark border");
  // the field sequence: plain, then projection, plus a theme toggle mixed in
  const sel = doc.getElementById("surf-colormode");
  sel.value = "plain"; sel.dispatchEvent(new window.Event("change"));
  doc.getElementById("theme-toggle").click();
  sel.value = "projection"; sel.dispatchEvent(new window.Event("change"));
  const last = state.newPlot.filter(p => p.id === "surf3d").pop();
  assert.equal(dotColorOf(last), "#ffffff",
    "STILL white after plain -> theme toggle -> projection (no theme coupling, no staleness)");
});

// ---------------------------------------------------------------------------
// 0.7.0 — multi-material manifest + selector
// ---------------------------------------------------------------------------

const DEMO_MANIFEST = { materials: [
  { slug: "snse2",    label: "SnSe2",            file: "navigator_data_snse2.js" },
  { slug: "graphene", label: "Graphene (mp-990448)", file: "navigator_data_graphene.js" },
  { slug: "mos2",     label: "MoS2 2H (mp-2815)", file: "navigator_data_mos2.js" },
] };

test("110: 0.7.0 material selector — present with >=2 entries, active selected, first rail section", () => {
  const { window } = loadPage({ manifest: DEMO_MANIFEST, active: "graphene" });
  const doc = window.document;
  const sel = doc.getElementById("material-select");
  assert.ok(sel, "selector injected");
  const opts = [...sel.options].map(o => o.value);
  assert.equal(JSON.stringify(opts), JSON.stringify(["snse2", "graphene", "mos2"]));
  assert.equal(sel.value, "graphene", "active slug pre-selected");
  const rail = doc.getElementById("atoms-controls");
  assert.ok(rail.firstChild.contains(sel), "Material section sits at the TOP of the rail");
  assert.equal(rail.firstChild.querySelector(".rail-section-title").textContent, "Material");
});

test("111: 0.7.0 selector is presence-keyed — absent without a manifest, absent for single-entry", () => {
  const a = loadPage();
  assert.equal(a.window.document.getElementById("material-select"), null,
    "no manifest -> no selector (every pre-0.7.0 bake unchanged)");
  const b = loadPage({ manifest: { materials: [DEMO_MANIFEST.materials[0]] }, active: "snse2" });
  assert.equal(b.window.document.getElementById("material-select"), null,
    "single-entry manifest (exporter default) -> no selector");
});

// ---------------------------------------------------------------------------
// 0.7.1 Tier 0 — the smooth rim clamp (viewer side)
// ---------------------------------------------------------------------------

test("112: 0.7.1 rim clamp — outside-zone vertices land ON the hexagon; inside ones bitwise unchanged", () => {
  const d = synthNavDataWeights();
  // stretch the grid so corners/edges fall OUTSIDE the hexagon (R = 0.17)
  d.surface.kx = [-0.3, 0, 0.3]; d.surface.ky = [-0.3, 0, 0.3];
  const { state, threw } = loadPage({ navData: d });
  assert.equal(threw, null, threw && threw.stack);
  const surf = state.newPlot.find(p => p.id === "surf3d");
  const t0 = surf.data[0];                       // band 0 mesh3d
  assert.equal(t0.type, "mesh3d");
  const V = d.bz_polygon;
  const inHull = (x, y) => {
    for (let k = 0; k < V.length; k++) {
      const a = V[k], b = V[(k + 1) % V.length];
      if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < -1e-9) return false;
    }
    return true;
  };
  let clamped = 0;
  for (let v = 0; v < t0.x.length; v++) {
    const x = t0.x[v], y = t0.y[v];
    assert.ok(inHull(x, y), `vertex ${v} (${x},${y}) must lie in the closed hexagon`);
    if (Math.hypot(x, y) < 0.299) clamped++;     // moved off its 0.3-grid position
  }
  assert.ok(clamped >= 8, "the outside ring got clamped onto the boundary: " + clamped);
  // the center (0,0) vertex is bitwise untouched
  const c = t0.x.findIndex((x, v) => x === 0 && t0.y[v] === 0);
  assert.ok(c >= 0, "Γ vertex passes through the clamp unchanged");
});

// ---------------------------------------------------------------------------
// 0.7.2 Tier 1 — extended zones by G-tiling
// ---------------------------------------------------------------------------

function weightsWithRecip() {
  const d = synthNavDataWeights();
  // hexagonal reciprocal lattice consistent with the 0.17-circumradius BZ
  d.meta.material.reciprocal_lattice = [[0.3, 0, 0], [0.15, 0.26, 0], [0, 0, 0.025]];
  return d;
}

test("113: 0.7.2 extended zones — toggle tiles every band mesh at the first G-ring", () => {
  const { window, state } = loadPage({ navData: weightsWithRecip() });
  const doc = window.document;
  const wrap = doc.getElementById("ext-zones-wrap");
  assert.notEqual(wrap.style.display, "none", "toggle offered (weights + reciprocal lattice)");
  const before = latestSurfPlot(state);
  const v0 = before.data[0].x.length;
  assert.equal(before.data[before.data.length - 1].visible, false, "placeholder invisible when off");
  const box = doc.getElementById("t-ext");
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  const after = latestSurfPlot(state);
  const t0 = after.data[0];
  assert.equal(t0.x.length % v0, 0, "vertex count = first-zone count x tiles");
  const tiles = t0.x.length / v0;
  assert.ok(tiles >= 5 && tiles <= 9, "first G-ring: 4-8 translates + zone 0, got " + tiles);
  assert.equal(t0.z.length, t0.x.length, "z repeated per tile (the G-periodic licence)");
  // z values repeat verbatim across tiles
  for (let v = 0; v < v0; v++)
    assert.equal(t0.z[v0 + v], t0.z[v], "tile 1 z == zone 0 z at vertex " + v);
  const tiling = after.data[after.data.length - 1];
  assert.equal(tiling.name, "zone tiling");
  assert.equal(tiling.visible, true, "floor tiling shown");
  assert.ok(tiling.x.length > 0, "tiled hexagons present");
  // axis window widened
  const r0 = Math.abs(before.layout.scene.xaxis.range[0]);
  const r1 = Math.abs(after.layout.scene.xaxis.range[0]);
  assert.ok(r1 > 1.5 * r0, "kx range widened for the tiled view");
});

test("114: 0.7.2 fold-click — a click on an extended tile selects the SAME first-zone node", async () => {
  const d = weightsWithRecip();
  const dm = compactWithMeshModes();
  d.mesh_modes = dm.mesh_modes;              // give it clickable nodes
  const { window, state } = loadPage({ navData: d });
  const doc = window.document;
  const box = doc.getElementById("t-ext");
  box.checked = true; box.dispatchEvent(new window.Event("change"));
  const B = d.meta.material.reciprocal_lattice;
  const q1 = d.mesh_modes.q_cart[1];         // node 1 at (0.05, 0.02)
  // click at node 1's position TRANSLATED by G = b1
  state.cb["surf3d:plotly_click"]({ points: [{ curveNumber: 2, pointNumber: 0,
    x: q1[0] + B[0][0], y: q1[1] + B[0][1] }] }); await flushClick();
  const info = doc.getElementById("a-info").textContent;
  assert.ok(info.includes("q = (0.250, 0.100, 0.000)"),
    "extended click folded to the first-zone node: " + info);
});
