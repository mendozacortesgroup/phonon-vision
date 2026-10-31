/* JS-side tests for adp.js and polyhedra.js.
 *
 * The ADP half is a PARITY suite: python/phonon_adp.py is the oracle, and the
 * expected values in tests-js/_oracle.json are generated from it by
 * tests/make_oracle.py. If the two implementations ever drift, these fail.
 *
 * The polyhedra half checks the hull against solids whose face counts and
 * volumes are known in closed form — so the geometry is verified against
 * mathematics, not against itself.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, "..", "web");

function loadModule(file) {
  const ctx = { globalThis: {}, Math, console };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  vm.runInContext(readFileSync(join(WEB, file), "utf8"), ctx);
  return ctx;
}

const { PVAdp } = loadModule("adp.js");
const { PVPolyhedra } = loadModule("polyhedra.js");

const ORACLE_PATH = join(HERE, "_oracle.json");
const oracle = existsSync(ORACLE_PATH)
  ? JSON.parse(readFileSync(ORACLE_PATH, "utf8")) : null;

/* ------------------------------------------------------------------ ADP */

test("ORTEP scale factors are the documented constants", () => {
  assert.equal(PVAdp.ellipsoidScale(50).toFixed(6), "1.538172");
  assert.equal(PVAdp.ellipsoidScale(90).toFixed(6), "2.500278");
  assert.equal(PVAdp.ellipsoidScale(99).toFixed(6), "3.368228");
  assert.throws(() => PVAdp.ellipsoidScale(42), /50, 90 or 99/);
});

test("single oscillator matches the closed form", () => {
  const nu = 4.0, m = 50.0, T = 300.0;
  const mesh = {
    frequency: [[nu]],
    eigenvectors_real: [[[[1, 0, 0]]]],
    eigenvectors_imag: [[[[0, 0, 0]]]]
  };
  const { U } = PVAdp.adpTensors(mesh, [m], T, { freqFloorTHz: 0 });
  const { HBAR, KB, AMU, THZ_TO_RAD_S } = PVAdp.constants;
  const w = THZ_TO_RAD_S * nu;
  const expect = HBAR / (2 * m * AMU * w) / Math.tanh(HBAR * w / (2 * KB * T)) * 1e20;
  assert.ok(Math.abs(U[0][0][0] / expect - 1) < 1e-12, `${U[0][0][0]} vs ${expect}`);
  assert.equal(U[0][1][1], 0);
});

test("zero-point motion survives at low temperature", () => {
  const mesh = {
    frequency: [[6.0]],
    eigenvectors_real: [[[[1, 0, 0]]]],
    eigenvectors_imag: [[[[0, 0, 0]]]]
  };
  const cold = PVAdp.adpTensors(mesh, [30], 1e-3, { freqFloorTHz: 0 }).U[0][0][0];
  const colder = PVAdp.adpTensors(mesh, [30], 1e-4, { freqFloorTHz: 0 }).U[0][0][0];
  assert.ok(cold > 0);
  assert.ok(Math.abs(cold / colder - 1) < 1e-9, "quantum ADP must saturate as T->0");
});

test("high temperature converges on equipartition", () => {
  const mesh = {
    frequency: [[5.0, 9.0]],
    eigenvectors_real: [[[[0.6, 0.8, 0]], [[0, 0, 1]]]],
    eigenvectors_imag: [[[[0, 0, 0]], [[0, 0, 0]]]]
  };
  const q = PVAdp.adpTensors(mesh, [40], 20000, { freqFloorTHz: 0 }).U[0];
  const c = PVAdp.adpTensors(mesh, [40], 20000, { freqFloorTHz: 0, classical: true }).U[0];
  assert.ok(Math.abs(PVAdp.uIsotropic(q) / PVAdp.uIsotropic(c) - 1) < 1e-4);
});

test("the frequency floor is reported, not silent", () => {
  const mesh = {
    frequency: [[0.02, 5.0]],
    eigenvectors_real: [[[[1, 0, 0]], [[0, 1, 0]]]],
    eigenvectors_imag: [[[[0, 0, 0]], [[0, 0, 0]]]]
  };
  const { info } = PVAdp.adpTensors(mesh, [20], 300, { freqFloorTHz: 0.1 });
  assert.equal(info.nModesDropped, 1);
  assert.equal(info.nModesKept, 1);
  assert.equal(info.fMinKeptTHz, 5.0);
});

test("rejects bad input loudly", () => {
  const mesh = {
    frequency: [[5.0]],
    eigenvectors_real: [[[[1, 0, 0]]]],
    eigenvectors_imag: [[[[0, 0, 0]]]]
  };
  assert.throws(() => PVAdp.adpTensors(mesh, [20], 0), /temperature/);
  assert.throws(() => PVAdp.adpTensors(mesh, [20, 30], 300), /shape mismatch/);
});

test("principalAxes reconstructs a known tensor", () => {
  // diag(4,1,9) rotated 30 deg about z — eigenvalues must come back sorted
  const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
  const R = [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  const D = [[4, 0, 0], [0, 1, 0], [0, 0, 9]];
  const U = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let v = 0;
    for (let k = 0; k < 3; k++) v += R[i][k] * D[k][k] * R[j][k];
    U[i][j] = v;
  }
  const { values, vectors } = PVAdp.principalAxes(U);
  // cross-realm: arrays from the vm context have that realm's prototype, so
  // deepStrictEqual would fail on identity alone — compare values, not objects
  assert.deepEqual([...values].map(v => Math.round(v * 1e9) / 1e9), [1, 4, 9]);
  for (let i = 0; i < 3; i++) {
    const n = Math.hypot(...vectors[i]);
    assert.ok(Math.abs(n - 1) < 1e-12, "eigenvectors must be unit length");
  }
});

test("ellipsoidFor gives a right-handed frame and ORTEP semi-axes", () => {
  const U = [[0.01, 0, 0], [0, 0.02, 0], [0, 0, 0.05]];
  const e = PVAdp.ellipsoidFor(U, 50);
  const c = PVAdp.ellipsoidScale(50);
  assert.ok(Math.abs(e.semiAxes[2] - c * Math.sqrt(0.05)) < 1e-12);
  assert.ok(Math.abs(e.semiAxes[0] - c * Math.sqrt(0.01)) < 1e-12);
  const [a, b, d] = e.axes;
  const det = a[0] * (b[1] * d[2] - b[2] * d[1])
            - a[1] * (b[0] * d[2] - b[2] * d[0])
            + a[2] * (b[0] * d[1] - b[1] * d[0]);
  assert.ok(det > 0, "frame must be a rotation, not a mirror");
});

test("PARITY with the python oracle", { skip: !oracle && "run tests/make_oracle.py first" },
  () => {
    for (const casename of Object.keys(oracle.cases)) {
      const c = oracle.cases[casename];
      const { U, info } = PVAdp.adpTensors(c.mesh, c.masses, c.temperature,
        { freqFloorTHz: c.freqFloor });
      assert.equal(info.nModesDropped, c.expect.nModesDropped, `${casename}: dropped`);
      for (let s = 0; s < U.length; s++) {
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            const got = U[s][a][b], want = c.expect.U[s][a][b];
            const scale = Math.max(Math.abs(want), 1e-12);
            assert.ok(Math.abs(got - want) / scale < 1e-10,
              `${casename} U[${s}][${a}][${b}]: js ${got} vs py ${want}`);
          }
        }
      }
    }
  });

/* ----------------------------------------------------------- polyhedra */

const OCTAHEDRON = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const CUBE = [
  [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
  [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
];

const poly = (pts) => ({
  centre: -1,
  vertexAtoms: pts.map((_, i) => i),
  faces: PVPolyhedra.convexHullFaces(pts)
});
const at = (pts) => (i) => pts[i];

test("octahedron hull has 8 triangular faces and the exact volume", () => {
  const p = poly(OCTAHEDRON);
  assert.equal(p.faces.length, 8);
  // regular octahedron with unit vertices: V = 4/3
  assert.ok(Math.abs(PVPolyhedra.volume(p, at(OCTAHEDRON)) - 4 / 3) < 1e-12);
});

test("cube hull triangulates to 12 faces and volume 8", () => {
  const p = poly(CUBE);
  assert.ok(Math.abs(PVPolyhedra.volume(p, at(CUBE)) - 8) < 1e-9,
    `volume ${PVPolyhedra.volume(p, at(CUBE))}`);
  assert.ok(p.faces.length >= 12, `expected >=12 triangles, got ${p.faces.length}`);
});

test("faces are wound outward (positive volume)", () => {
  for (const pts of [OCTAHEDRON, CUBE]) {
    assert.ok(PVPolyhedra.volume(poly(pts), at(pts)) > 0);
  }
});

test("a breathing distortion changes volume; a rotation does not", () => {
  const p = poly(OCTAHEDRON);
  const v0 = PVPolyhedra.volume(p, at(OCTAHEDRON));

  const breathed = OCTAHEDRON.map(q => q.map(x => x * 1.1));
  const vB = PVPolyhedra.volume({ ...p }, at(breathed));
  assert.ok(Math.abs(vB / v0 - 1.331) < 1e-9, "breathing must scale volume as s^3");

  const th = 0.3, c = Math.cos(th), s = Math.sin(th);
  const rotated = OCTAHEDRON.map(([x, y, z]) => [c * x - s * y, s * x + c * y, z]);
  const vR = PVPolyhedra.volume({ ...p }, at(rotated));
  assert.ok(Math.abs(vR / v0 - 1) < 1e-12, "rigid rotation must preserve volume");
});

test("degenerate shells are skipped, not crashed on", () => {
  assert.equal(PVPolyhedra.convexHullFaces([[0, 0, 0], [1, 0, 0]]).length, 0);
  const collinear = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]];
  assert.equal(PVPolyhedra.convexHullFaces(collinear).length, 0);
});

test("buildPolyhedra honours the minimum-vertex rule", () => {
  const pos = [[0, 0, 0], ...OCTAHEDRON];
  const built = PVPolyhedra.buildPolyhedra(
    [0], () => [1, 2, 3], (i) => pos[i], { minVertices: 4 });
  assert.equal(built.length, 0, "3 neighbours cannot enclose a volume");

  const ok = PVPolyhedra.buildPolyhedra(
    [0], () => [1, 2, 3, 4, 5, 6], (i) => pos[i], { minVertices: 4 });
  assert.equal(ok.length, 1);
  assert.equal(ok[0].faces.length, 8);
});

test("triangleVertices packs 9 floats per face and reuses its buffer", () => {
  const p = poly(OCTAHEDRON);
  const buf = PVPolyhedra.triangleVertices(p, at(OCTAHEDRON));
  assert.equal(buf.length, p.faces.length * 9);
  const again = PVPolyhedra.triangleVertices(p, at(OCTAHEDRON), buf);
  assert.equal(again, buf, "must write in place when a buffer is supplied");
});

/* -------------------------------------------------- radius tables (Tier 2) */

const { PVRadii } = loadModule("radii_data.js");

test("radius tables load with the documented coverage", () => {
  assert.ok(PVRadii.coverage.vdw >= 90, `vdW coverage ${PVRadii.coverage.vdw}`);
  assert.ok(PVRadii.coverage.atomic >= 85, `atomic coverage ${PVRadii.coverage.atomic}`);
  assert.deepEqual([...PVRadii.styles], ["covalent", "atomic", "vdw"]);
  assert.match(PVRadii.source, /pymatgen/);
});

test("known van der Waals radii survive regeneration", () => {
  // independently known values (Bondi 1964 / Alvarez 2013). These guard the
  // SHIPPED file, so a future regeneration that reads the wrong pymatgen
  // field fails here rather than silently resizing every atom on screen.
  const expect = { C: 1.70, N: 1.55, O: 1.52, S: 1.80, Se: 1.90, Sn: 2.17 };
  for (const [sym, want] of Object.entries(expect)) {
    const got = PVRadii.get("vdw", sym);
    assert.ok(Math.abs(got - want) < 0.06, `vdW ${sym}: ${got} vs ~${want}`);
  }
});

test("van der Waals radius exceeds atomic radius for every shared element", () => {
  let checked = 0;
  for (const sym of Object.keys(PVRadii.atomic)) {
    const v = PVRadii.vdw[sym];
    if (typeof v !== "number") continue;
    assert.ok(v > PVRadii.atomic[sym], `${sym}: vdW ${v} <= atomic ${PVRadii.atomic[sym]}`);
    checked++;
  }
  assert.ok(checked > 80, `only ${checked} elements compared`);
});

test("missing entries return null so the caller chooses the fallback", () => {
  // the contract: never invent a radius. index.html falls back to covalent
  // AND reports which elements it did that for.
  assert.equal(PVRadii.get("vdw", "Zz"), null);
  assert.equal(PVRadii.get("atomic", "He"), null);   // no Slater radius for He
  assert.equal(PVRadii.get("covalent", "C"), null);  // covalent lives in radii.py
  assert.equal(PVRadii.get("nonsense", "C"), null);
});

/* ------------------------------------------------ trajectory export (Tier 3) */

const { PVTrajectory } = loadModule("trajectory.js");

const TRAJ_SPEC = {
  symbols: ["Sn", "Se"],
  lattice: [[3.2, 0, 0], [0, 3.2, 0], [0, 0, 40]],
  frames: [
    { phase: 0, positions: [[0, 0, 0], [1, 1, 1]], displacements: [[0.1, 0, 0], [0, -0.1, 0]] },
    { phase: 1, positions: [[0.1, 0, 0], [1, 0.9, 1]], displacements: [[0, 0, 0], [0, 0, 0]] }
  ]
};

test("extxyz has the frame structure readers expect", () => {
  const lines = PVTrajectory.buildExtxyz(TRAJ_SPEC).trimEnd().split("\n");
  // 2 frames x (1 count line + 1 comment line + 2 atoms)
  assert.equal(lines.length, 2 * (2 + 2));
  assert.equal(lines[0], "2");
  assert.equal(lines[4], "2", "each frame restates the atom count");
  assert.match(lines[1], /^Lattice="/);
  assert.match(lines[1], /Properties=species:S:1:pos:R:3:disp:R:3/);
  assert.match(lines[2], /^Sn (-?\d+\.\d+ ){5}-?\d+\.\d+$/, `atom row: ${lines[2]}`);
});

test("the lattice is written as nine row-major numbers", () => {
  const head = PVTrajectory.buildExtxyz(TRAJ_SPEC).split("\n")[1];
  const nums = /Lattice="([^"]+)"/.exec(head)[1].trim().split(/\s+/);
  assert.equal(nums.length, 9);
  assert.equal(parseFloat(nums[0]), 3.2);
  assert.equal(parseFloat(nums[8]), 40);
});

test("displacement column is omitted when not supplied", () => {
  const spec = { ...TRAJ_SPEC, frames: [{ positions: [[0, 0, 0], [1, 1, 1]] }] };
  const lines = PVTrajectory.buildExtxyz(spec).trimEnd().split("\n");
  assert.match(lines[1], /Properties=species:S:1:pos:R:3\b/);
  assert.ok(!/:disp:/.test(lines[1]), "no displacement column should be declared");
  assert.equal(lines[2].trim().split(/\s+/).length, 4, "symbol + 3 coordinates");
});

test("metadata is quoted only when it needs to be", () => {
  const spec = { ...TRAJ_SPEC, meta: { branch: 3, note: "two words" } };
  const head = PVTrajectory.buildExtxyz(spec).split("\n")[1];
  assert.match(head, /branch=3(?!")/);
  assert.match(head, /note="two words"/);
});

test("loop phases exclude the endpoint so the animation does not stutter", () => {
  const p = PVTrajectory.loopPhases(4);
  assert.equal(p.length, 4);
  assert.equal(p[0], 0);
  assert.ok(Math.abs(p[3] - 1.5 * Math.PI) < 1e-12);
  assert.ok(p[p.length - 1] < 2 * Math.PI, "2*pi would duplicate frame 0");
});

test("tiled lattice scales each row by its tile count", () => {
  const L = PVTrajectory.tiledLattice([[2, 0, 0], [0, 3, 0], [0, 0, 5]], 3, 2, 1);
  assert.deepEqual([...L[0]], [6, 0, 0]);
  assert.deepEqual([...L[1]], [0, 6, 0]);
  assert.deepEqual([...L[2]], [0, 0, 5]);
});

test("malformed input is rejected rather than written out", () => {
  assert.throws(() => PVTrajectory.buildExtxyz({ symbols: [], frames: [], lattice: [] }),
    /no symbols/);
  assert.throws(() => PVTrajectory.buildExtxyz({ ...TRAJ_SPEC, frames: [] }), /no frames/);
  assert.throws(() => PVTrajectory.buildExtxyz({
    ...TRAJ_SPEC, frames: [{ positions: [[0, 0, 0]] }]     // 1 position, 2 symbols
  }), /1 positions for 2 symbols/);
});

test("negative zero is normalized away", () => {
  const spec = {
    symbols: ["C"], lattice: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    frames: [{ positions: [[-1e-12, 0, 0]] }]
  };
  const row = PVTrajectory.buildExtxyz(spec).split("\n")[2];
  assert.ok(!/-0\.000000/.test(row), `row still carries a signed zero: ${row}`);
});
