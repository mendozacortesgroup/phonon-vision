/*
 * web/kinematics.js — phonon kinematics in JS (the SECOND kinematics engine).
 *
 * Pure, dependency-free, side-effect-free. Turns a selected band-path mode (the
 * canonicalized eigenvector + fractional q already in the navigator payload) into
 * real Cartesian displacements for the 1b atom-motion animator. It mirrors
 * phonon_core.kinematics exactly and is gated against the frozen golden to 1e-10
 * (see tests-js/kinematics.test.mjs); the eigenvectors are already canonical, so
 * NO canonicalization is done here.
 *
 * Conventions (must match phonon_core.kinematics / the golden's meta.conventions):
 *   - bloch_phase  : 2*pi * (n_int . q_frac); q FRACTIONAL, n INTEGER, no metric.
 *   - displacement : u = A * Re[ e * exp(i*(bloch + phi)) ] / sqrt(m); Dove 4.16,
 *                    mass-weighted by default (phonopy eigenvectors are
 *                    mass-weighted-normalized).
 *
 * Loads as a classic script (file:// friendly — NOT an ES module): attaches
 * window.PhononKinematics in the browser and is require()-able in Node.
 */
(function (root, factory) {
  const api = factory();
  root.PhononKinematics = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Bloch phase 2*pi * (n . q_frac). q in fractional reciprocal coords, n integer
  // cell index in the lattice basis, so q.R = 2*pi * q_frac . n with no metric.
  function blochPhase(nInt, qFrac) {
    return 2 * Math.PI *
      (nInt[0] * qFrac[0] + nInt[1] * qFrac[1] + nInt[2] * qFrac[2]);
  }

  // One slot's Cartesian displacement at global phase phi:
  //   u = A * Re[ e * exp(i*theta) ] / sqrt(m),  theta = blochPhase + phi
  // For complex e = a + i b:  Re[(a+ib)(cos t + i sin t)] = a*cos t - b*sin t.
  // (The sign of the b-term is the classic mistake; the finite-q golden catches it.)
  function displaceSlot(eigReAtom, eigImAtom, qFrac, nInt, mass, A, phi, massWeighted) {
    if (massWeighted === undefined) massWeighted = true;
    const theta = blochPhase(nInt, qFrac) + phi;
    const c = Math.cos(theta), s = Math.sin(theta);
    const u = [0, 1, 2].map(d => A * (eigReAtom[d] * c - eigImAtom[d] * s));
    return massWeighted ? u.map(x => x / Math.sqrt(mass)) : u;
  }

  // Displacement of every supercell slot for one mode.
  //   eigReal, eigImag : (Na,3) real/imag parts of the canonical eigenvector
  //   masses           : (Na,) primitive-atom masses (same order as the atom axis)
  //   slots            : { j: [Ns], n_int: [Ns][3] } atom index + integer cell
  // Returns (Ns,3): slot k uses primitive atom j = slots.j[k].
  function displaceMode(eigReal, eigImag, masses, slots, qFrac, A, phi, massWeighted) {
    if (massWeighted === undefined) massWeighted = true;
    const j = slots.j, nInt = slots.n_int, out = new Array(j.length);
    for (let k = 0; k < j.length; k++) {
      const a = j[k];
      out[k] = displaceSlot(eigReal[a], eigImag[a], qFrac, nInt[k],
                            masses[a], A, phi, massWeighted);
    }
    return out;
  }

  // Phases for a seamless animation loop over [0, 2*pi): endpoint EXCLUDED so
  // frame nFrames coincides with frame 0 (no doubled frame). For 1b-3.
  function phaseGrid(nFrames) {
    const out = new Array(nFrames);
    for (let i = 0; i < nFrames; i++) out[i] = 2 * Math.PI * i / nFrames;
    return out;
  }

  return { blochPhase, displaceSlot, displaceMode, phaseGrid };
});
