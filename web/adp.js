/* adp.js — atomic displacement parameters (thermal ellipsoids) in the browser.
 *
 * The numerical twin of python/phonon_adp.py, which is the oracle: the JS and
 * Python results are asserted equal in tests-js/adp.test.mjs. Keep the two in
 * step — if you change a constant or a convention here, change it there.
 *
 * This file exists so that index.html stays free of Math.* calls: the house
 * rule in this project is that all arithmetic lives in a dedicated module
 * (kinematics.js does displacements, this does thermal tensors).
 *
 * Physics, in one line:
 *
 *   U_s^ab(T) = (1/Nq) SUM_{q,nu} hbar/(2 m_s w) coth(hbar w / 2 kB T)
 *                                 Re[e_{s,a} conj(e_{s,b})]
 *
 * i.e. the harmonic mean-square displacement tensor a crystallographer
 * refines and draws as an ORTEP ellipsoid (Willis & Pryor 1975; Johnson,
 * ORTEP, ORNL-3794, 1965). Its square root, scaled by the ORTEP probability
 * factor, gives the ellipsoid semi-axes.
 *
 * Two things a reader should know before trusting a number:
 *   - the 1/w weight makes soft modes dominate, so FREQ_FLOOR drops the
 *     near-Gamma acoustic modes a discrete mesh cannot resolve; the dropped
 *     count is returned, never hidden;
 *   - on a 2D material the flexural branch (w ~ q^2) makes U_zz grow as the
 *     mesh is refined. That is real physics, not a defect, but it means a
 *     coarse-mesh U is qualitative.
 */
(function (global) {
  "use strict";

  var HBAR = 1.054571817e-34;          // J s      (CODATA 2018)
  var KB = 1.380649e-23;               // J / K
  var AMU = 1.66053906660e-27;         // kg
  var TWO_PI = 6.283185307179586;
  var THZ_TO_RAD_S = TWO_PI * 1.0e12;
  var M2_TO_A2 = 1.0e20;

  var DEFAULT_FREQ_FLOOR_THZ = 0.1;

  /* ORTEP probability scale: semi-axis = c * sqrt(eigenvalue), where c solves
   * P(chi^2 with 3 dof <= c^2) = p. */
  var ORTEP_C = { 50: 1.5381722, 90: 2.5002777, 99: 3.3682279 };

  function ellipsoidScale(percent) {
    var c = ORTEP_C[percent];
    if (!c) throw new Error("adp: probability must be 50, 90 or 99, got " + percent);
    return c;
  }

  /* coth(x) for x > 0. Math.tanh saturates to 1 well before overflow, so the
   * reciprocal is safe; the guard is only for the x -> 0 pole. */
  function coth(x) {
    return 1.0 / Math.tanh(x < 1e-12 ? 1e-12 : x);
  }

  /* Mean-square displacement tensors.
   *
   * mesh: { frequency: [Nq][Nb], eigenvectors_real: [Nq][Nb][Na][3],
   *         eigenvectors_imag: same }
   * masses: [Na] in amu.  temperature: K (> 0).
   *
   * Returns { U: [Na][3][3] in A^2, info: {...} }.
   */
  function adpTensors(mesh, masses, temperature, options) {
    options = options || {};
    var floor = options.freqFloorTHz === undefined
      ? DEFAULT_FREQ_FLOOR_THZ : options.freqFloorTHz;
    var classical = !!options.classical;

    var f = mesh.frequency, er = mesh.eigenvectors_real, ei = mesh.eigenvectors_imag;
    if (!f || !er || !ei) throw new Error("adp: mesh is missing frequency/eigenvectors");
    if (!(temperature > 0)) throw new Error("adp: temperature must be > 0 K");

    var nq = f.length, nb = f[0].length, na = masses.length;
    if (er.length !== nq || er[0].length !== nb || er[0][0].length !== na) {
      throw new Error("adp: mesh/masses shape mismatch");
    }

    var U = new Array(na), s, a, b;
    for (s = 0; s < na; s++) U[s] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

    var dropped = 0, kept = 0, fMinKept = Infinity;

    for (var qi = 0; qi < nq; qi++) {
      var fq = f[qi], erq = er[qi], eiq = ei[qi];
      for (var bi = 0; bi < nb; bi++) {
        var nu = fq[bi];
        var anu = nu < 0 ? -nu : nu;
        if (anu < floor) { dropped++; continue; }
        kept++;
        if (anu < fMinKept) fMinKept = anu;

        var w = THZ_TO_RAD_S * anu;
        var amp = classical
          ? KB * temperature / (w * w)
          : HBAR / (2.0 * w) * coth(HBAR * w / (2.0 * KB * temperature));

        for (s = 0; s < na; s++) {
          var re = erq[bi][s], im = eiq[bi][s];
          var scale = amp / (masses[s] * AMU);
          var Us = U[s];
          for (a = 0; a < 3; a++) {
            var ra = re[a], ia = im[a];
            for (b = a; b < 3; b++) {
              /* Re[e_a conj(e_b)] — the antisymmetric imaginary part cancels
               * in the thermal average, so only this symmetric combination
               * contributes. */
              Us[a][b] += scale * (ra * re[b] + ia * im[b]);
            }
          }
        }
      }
    }

    for (s = 0; s < na; s++) {
      for (a = 0; a < 3; a++) {
        for (b = a; b < 3; b++) {
          var v = U[s][a][b] * M2_TO_A2 / nq;
          U[s][a][b] = v;
          U[s][b][a] = v;                       // symmetric by construction
        }
      }
    }

    return {
      U: U,
      info: {
        nQ: nq, nModesTotal: nq * nb, nModesDropped: dropped, nModesKept: kept,
        freqFloorTHz: floor, temperatureK: temperature,
        fMinKeptTHz: kept ? fMinKept : null
      }
    };
  }

  function uIsotropic(Us) { return (Us[0][0] + Us[1][1] + Us[2][2]) / 3.0; }

  /* Symmetric 3x3 eigen-decomposition by cyclic Jacobi rotations.
   *
   * Chosen over a closed-form (Cardano) solver because Jacobi stays accurate
   * for the near-degenerate tensors an isotropic site produces, and because
   * it returns an exactly orthonormal frame — which is what feeds a rotation
   * matrix. Converges in a handful of sweeps for 3x3.
   *
   * Returns { values: [3] ascending, vectors: [3][3] with vectors[i] the unit
   * axis belonging to values[i] }.
   */
  function principalAxes(Us) {
    var a = [[Us[0][0], Us[0][1], Us[0][2]],
             [Us[1][0], Us[1][1], Us[1][2]],
             [Us[2][0], Us[2][1], Us[2][2]]];
    var v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    var p, q, i, sweep;

    for (sweep = 0; sweep < 24; sweep++) {
      var off = 0;
      for (p = 0; p < 2; p++) for (q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
      if (off < 1e-30) break;

      for (p = 0; p < 2; p++) {
        for (q = p + 1; q < 3; q++) {
          if (a[p][q] === 0) continue;
          var theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;

          for (i = 0; i < 3; i++) {
            var aip = a[i][p], aiq = a[i][q];
            a[i][p] = c * aip - s * aiq;
            a[i][q] = s * aip + c * aiq;
          }
          for (i = 0; i < 3; i++) {
            var api = a[p][i], aqi = a[q][i];
            a[p][i] = c * api - s * aqi;
            a[q][i] = s * api + c * aqi;
            var vip = v[i][p], viq = v[i][q];
            v[i][p] = c * vip - s * viq;
            v[i][q] = s * vip + c * viq;
          }
        }
      }
    }

    var order = [0, 1, 2].sort(function (x, y) { return a[x][x] - a[y][y]; });
    return {
      values: order.map(function (k) { return a[k][k]; }),
      vectors: order.map(function (k) { return [v[0][k], v[1][k], v[2][k]]; })
    };
  }

  /* Everything a renderer needs for one atom's ellipsoid: the three semi-axis
   * lengths in Angstrom and an orthonormal frame (column-major 3x3 as a flat
   * 9-array, ready for a rotation matrix). Degenerate/tiny axes are floored
   * so a nearly-isotropic site still renders as a visible sphere. */
  function ellipsoidFor(Us, probabilityPercent, minSemiAxisA) {
    var c = ellipsoidScale(probabilityPercent);
    var pa = principalAxes(Us);
    var floorA = minSemiAxisA === undefined ? 1e-3 : minSemiAxisA;
    var semi = pa.values.map(function (lam) {
      var r = c * Math.sqrt(lam > 0 ? lam : 0);
      return r < floorA ? floorA : r;
    });
    var m = pa.vectors;
    /* right-handed frame: flip the last axis if the triple product is negative,
     * so the matrix is a pure rotation and never mirrors the geometry */
    var det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    var axes = det < 0 ? [m[0], m[1], [-m[2][0], -m[2][1], -m[2][2]]] : m;
    return {
      semiAxes: semi,
      axes: axes,
      basis: [axes[0][0], axes[0][1], axes[0][2],
              axes[1][0], axes[1][1], axes[1][2],
              axes[2][0], axes[2][1], axes[2][2]],
      uIso: uIsotropic(Us),
      anisotropy: semi[2] / (semi[0] > 0 ? semi[0] : 1e-12)
    };
  }

  global.PVAdp = {
    adpTensors: adpTensors,
    principalAxes: principalAxes,
    ellipsoidFor: ellipsoidFor,
    ellipsoidScale: ellipsoidScale,
    uIsotropic: uIsotropic,
    DEFAULT_FREQ_FLOOR_THZ: DEFAULT_FREQ_FLOOR_THZ,
    constants: { HBAR: HBAR, KB: KB, AMU: AMU, THZ_TO_RAD_S: THZ_TO_RAD_S }
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
