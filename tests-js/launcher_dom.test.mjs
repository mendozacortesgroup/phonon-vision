// Independent jsdom harness for web/pv_launcher.html — its OWN JSDOM (NOT the index.html
// harness). jsdom locks window.location (own, non-configurable) so assign() can't be stubbed
// on a live window; we run the launcher's own <script> with a controlled context whose
// `window` facade exposes a spyable location.assign + injected fetch/FormData.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const LAUNCHER = path.resolve(__dirname, "..", "web", "pv_launcher.html");

function loadLauncher() {
  const html = readFileSync(LAUNCHER, "utf-8");
  const dom = new JSDOM(html, { runScripts: "outside-only", url: "http://localhost/" });
  const { window } = dom;
  const doc = window.document;
  const state = { captured: null, assigned: null, fetchResult: { ok: true, url: "/x" } };
  const fetchStub = (url, opts) => {
    state.captured = { url, body: opts && opts.body };
    return Promise.resolve({ json: () => Promise.resolve(state.fetchResult) });
  };
  // post-0.6.61 the launcher no longer navigates; the facade's assign just proves it's NOT called
  const facadeWindow = { location: { assign: (u) => { state.assigned = u; } } };
  const scriptText = doc.querySelector("script").textContent;
  const runner = new Function("window", "document", "fetch", "FormData", scriptText);
  runner(facadeWindow, doc, fetchStub, window.FormData);
  return { window, doc, state };
}

function fire(window, el, type) {
  el.dispatchEvent(new window.Event(type, { bubbles: true }));
}

function stageFile(window, doc, id, name, bytes) {
  const inp = doc.getElementById(id);
  const file = new window.File([bytes], name, { type: "application/octet-stream" });
  Object.defineProperty(inp, "files", { value: [file], configurable: true });
  fire(window, inp, "change");   // a real pick fires change -> refreshEcho -> updateGoState
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("launcher: rail has the 7 file inputs, 3 fields, and #pv-go", () => {
  const { doc } = loadLauncher();
  for (const id of ["pv-band", "pv-mesh", "pv-total-dos", "pv-proj-dos", "pv-pdos-conf",
                    "pv-engine-yaml", "pv-force-constants"]) {   // 0.6.83 engine slots
    const el = doc.getElementById(id);
    assert.ok(el && el.tagName === "INPUT" && el.type === "file", "file input " + id);
  }
  for (const id of ["pv-group-a", "pv-dim", "pv-grid", "pv-go", "pv-status", "pv-results"]) {
    assert.ok(doc.getElementById(id), "field/control " + id);
  }
});

test("launcher: GO logs a #pv-results entry (open link = url, NO navigation) — dim 3", async () => {
  const { window, doc, state } = loadLauncher();
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-total-dos", "total.dat", "TDOS");   // mesh/proj_dos/pdos_conf empty
  doc.getElementById("pv-dim").value = "3";
  state.fetchResult = { ok: true, url: "/pv_serve_out/X/index.html" };

  doc.getElementById("pv-go").click();
  await tick();

  // FormData contract preserved (dim-3: no grid, no mesh)
  assert.deepEqual([...state.captured.body.keys()].sort(), ["band", "dim", "group_a", "total_dos"]);
  assert.equal(state.captured.body.has("grid"), false);

  // a result entry prepended; open link href EXACTLY data.url; summary shows dim + group_a
  const results = doc.getElementById("pv-results");
  assert.equal(results.children.length, 1, "one result entry prepended");
  const link = results.querySelector(".pv-result-link");
  assert.ok(link, "open link present");
  assert.equal(link.getAttribute("href"), "/pv_serve_out/X/index.html", "link href === data.url");
  const text = results.children[0].textContent;
  assert.ok(text.includes("dim=3"), "entry shows the sent dim");
  assert.ok(text.includes("group_a=Sn"), "entry shows group_a");
  assert.equal(state.assigned, null, "location.assign must NOT be called");
});

test("launcher: {ok:false,error} renders a readable error region (pv-error class)", async () => {
  const { window, doc, state } = loadLauncher();
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  state.fetchResult = { ok: false, error: "boom" };

  doc.getElementById("pv-go").click();
  await tick();

  const st = doc.getElementById("pv-status");
  assert.equal(st.textContent, "boom", "error surfaced");
  assert.ok(st.classList.contains("pv-error"), "error region carries the pv-error class");
  assert.equal(doc.getElementById("pv-results").children.length, 0, "no result entry on failure");
});

test("launcher: GO is gated on a band file (disabled at init, enabled after staging band)", () => {
  const { window, doc } = loadLauncher();
  assert.equal(doc.getElementById("pv-go").disabled, true, "GO disabled at init (no band)");
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  assert.equal(doc.getElementById("pv-go").disabled, false, "GO enabled once a band file is staged");
});

test("launcher: dim=3 disables mesh+grid (mesh zone greyed) and the FormData drops both", async () => {
  const { window, doc, state } = loadLauncher();
  doc.getElementById("pv-dim").value = "3";
  fire(window, doc.getElementById("pv-dim"), "change");
  assert.equal(doc.getElementById("pv-mesh").disabled, true, "mesh input disabled for dim 3");
  assert.equal(doc.getElementById("pv-grid").disabled, true, "grid input disabled for dim 3");
  assert.ok(doc.querySelector('.pv-slot[data-slot="mesh"]').classList.contains("pv-disabled"),
    "mesh zone greyed (pv-disabled) for dim 3");

  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-mesh", "mesh.hdf5", "MESH");   // staged, but dropped for dim 3
  state.fetchResult = { ok: true, url: "/pv_serve_out/Y/index.html" };
  doc.getElementById("pv-go").click();
  await tick();

  const fd = state.captured.body;
  assert.deepEqual([...fd.keys()].sort(), ["band", "dim", "group_a"],
    "dim-3: band + group_a + dim only; NO mesh, NO grid");
  assert.equal(fd.has("mesh"), false, "mesh dropped for dim 3");
  assert.equal(fd.has("grid"), false, "grid dropped for dim 3");
  assert.equal(state.assigned, null, "no navigation");
});

test("launcher: dim=2 re-enables mesh+grid; FormData includes both; result entry shows grid", async () => {
  const { window, doc, state } = loadLauncher();
  doc.getElementById("pv-dim").value = "3";
  fire(window, doc.getElementById("pv-dim"), "change");
  doc.getElementById("pv-dim").value = "2";
  fire(window, doc.getElementById("pv-dim"), "change");
  assert.equal(doc.getElementById("pv-mesh").disabled, false, "mesh re-enabled for dim 2");
  assert.equal(doc.getElementById("pv-grid").disabled, false, "grid re-enabled for dim 2");
  assert.equal(doc.querySelector('.pv-slot[data-slot="mesh"]').classList.contains("pv-disabled"), false,
    "mesh zone un-greyed for dim 2");

  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-mesh", "mesh.hdf5", "MESH");
  doc.getElementById("pv-grid").value = "110";
  state.fetchResult = { ok: true, url: "/pv_serve_out/Z/index.html" };
  doc.getElementById("pv-go").click();
  await tick();

  const fd = state.captured.body;
  assert.equal(fd.has("mesh"), true, "dim-2 includes mesh");
  assert.equal(fd.get("grid"), "110", "dim-2 includes grid === '110'");
  assert.equal(fd.get("dim"), "2");

  const results = doc.getElementById("pv-results");
  assert.equal(results.querySelector(".pv-result-link").getAttribute("href"),
    "/pv_serve_out/Z/index.html", "link href === data.url");
  const text = results.children[0].textContent;
  assert.ok(text.includes("dim=2") && text.includes("group_a=Sn") && text.includes("grid=110"),
    "dim-2 entry summary shows dim, group_a, grid");
  assert.equal(state.assigned, null, "no navigation");
});

test("launcher: mesh label reads 2D/dim 2, not the old 3D-surface phrasing", () => {
  const { doc } = loadLauncher();
  const meshTitle = doc.querySelector('.pv-slot[data-slot="mesh"] .pv-slot-title').textContent;
  assert.ok(meshTitle.includes("2D"), "mesh label mentions 2D");
  assert.ok(meshTitle.includes("dim 2"), "mesh label mentions dim 2");
  assert.ok(!meshTitle.includes("3D surface"), "mesh label must not say '3D surface'");
});

// ---- 0.6.83: engine slots (phonopy.yaml + FORCE_CONSTANTS -> exporter --engine) ----

test("launcher: staging engine_yaml + FORCE_CONSTANTS puts both in FormData; entry shows +engine", async () => {
  const { window, doc, state } = loadLauncher();
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-engine-yaml", "phonopy.yaml", "YAML");
  stageFile(window, doc, "pv-force-constants", "FORCE_CONSTANTS", "FC");
  state.fetchResult = { ok: true, url: "/pv_serve_out/E/index.html" };

  doc.getElementById("pv-go").click();
  await tick();

  const fd = state.captured.body;
  assert.deepEqual([...fd.keys()].sort(),
    ["band", "dim", "engine_yaml", "force_constants", "grid", "group_a"],
    "both engine slots posted alongside the standard fields");
  const text = doc.getElementById("pv-results").children[0].textContent;
  assert.ok(text.includes("+engine"), "result entry tags the engine build: " + text);
  assert.ok(!text.includes("engine_yaml") && !text.includes("force_constants"),
    "one clean +engine tag, not raw slot keys: " + text);
});

test("launcher: engine slots are NOT dim-gated (dim 3 still posts engine_yaml)", async () => {
  const { window, doc, state } = loadLauncher();
  doc.getElementById("pv-dim").value = "3";
  fire(window, doc.getElementById("pv-dim"), "change");
  assert.equal(doc.getElementById("pv-engine-yaml").disabled, false,
    "engine slot stays live for dim 3 (band evaluation + provenance are dimension-agnostic)");
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-engine-yaml", "phonopy.yaml", "YAML");
  state.fetchResult = { ok: true, url: "/pv_serve_out/F/index.html" };
  doc.getElementById("pv-go").click();
  await tick();
  const fd = state.captured.body;
  assert.equal(fd.has("engine_yaml"), true, "engine_yaml posted under dim 3");
  assert.equal(fd.has("mesh"), false, "mesh still dim-gated as before");
});

test("launcher: FC-without-yaml server 400 renders in the pv-error region", async () => {
  const { window, doc, state } = loadLauncher();
  stageFile(window, doc, "pv-band", "band.hdf5", "BAND");
  stageFile(window, doc, "pv-force-constants", "FORCE_CONSTANTS", "FC");
  state.fetchResult = { ok: false,
                        error: "FORCE_CONSTANTS requires phonopy.yaml (engine mode)" };
  doc.getElementById("pv-go").click();
  await tick();
  const st = doc.getElementById("pv-status");
  assert.equal(st.textContent, "FORCE_CONSTANTS requires phonopy.yaml (engine mode)",
    "the server's named error surfaced verbatim");
  assert.ok(st.classList.contains("pv-error"), "pv-error class applied");
  assert.equal(doc.getElementById("pv-results").children.length, 0, "no result entry");
});
