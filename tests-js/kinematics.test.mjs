/*
 * Node golden gate for web/kinematics.js — the cross-language proof that the JS
 * kinematics engine reproduces phonon_core.kinematics on the FROZEN golden to
 * 1e-10 (matches Python's _REGRESSION_ATOL). Zero npm dependencies; uses Node's
 * built-in test runner (node >= 18). No DOM, no browser.
 *
 *   Run:  node --test tests-js/kinematics.test.mjs
 *         (or, from the repo root, simply: node --test)
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// require the classic-script module directly (it sets module.exports in Node)
const K = require(join(__dirname, "..", "web", "kinematics.js"));
const GOLD = JSON.parse(
  readFileSync(join(__dirname, "..", "tests", "golden_kinematics_web.json"), "utf-8"));

const ATOL = 1e-10;                 // == Python _REGRESSION_ATOL
const MODES = ["gamma_acoustic", "gamma_optical", "m_point", "k_point", "generic"];

test("finite-q modes are present (the contract is meaningless without them)", () => {
  const nFinite = MODES.filter(m => GOLD.modes[m].is_finite_q).length;
  assert.ok(nFinite >= 3, `expected >= 3 finite-q modes, got ${nFinite}`);
});

test("blochPhase: 2*pi * n.q_frac, no metric (n=[1,0,0], q=[1/2,0,0] -> pi)", () => {
  assert.ok(Math.abs(K.blochPhase([1, 0, 0], [0.5, 0, 0]) - Math.PI) < 1e-12);
});

test("masses are present and match the eigenvector atom count", () => {
  const na = GOLD.modes.generic.eigenvector_real.length;
  assert.equal(GOLD.masses.length, na);
});

for (const m of MODES) {
  test(`displaceMode reproduces disp_phi to ${ATOL} for mode '${m}'`, () => {
    const md = GOLD.modes[m];
    const u = K.displaceMode(
      md.eigenvector_real, md.eigenvector_imag, GOLD.masses, md.slots,
      md.q_frac, md.inputs.amplitude, md.inputs.phi, md.inputs.mass_weighted);
    const ref = md.disp_phi;
    assert.equal(u.length, ref.length, `slot count for ${m}`);
    let worst = 0;
    for (let k = 0; k < ref.length; k++) {
      for (let d = 0; d < 3; d++) {
        const e = Math.abs(u[k][d] - ref[k][d]);
        if (e > worst) worst = e;
        assert.ok(e <= ATOL,
          `mode ${m} slot ${k} dim ${d}: |${u[k][d]} - ${ref[k][d]}| = ${e} > ${ATOL}`);
      }
    }
    // surface the worst error for the record (well under ATOL)
    assert.ok(worst <= ATOL);
  });
}
