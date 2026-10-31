/* trajectory.js — export the animated mode as a standard trajectory file.
 *
 * The viewer can show a phonon mode; this lets someone take it away. The
 * output is extended XYZ (extxyz), the format ASE writes and OVITO, VMD and
 * most analysis tools read, so a mode exported here opens in whatever the
 * recipient already uses without a converter.
 *
 * Format reference: the extxyz convention as implemented by ASE
 * (Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017)).
 * Per frame:
 *
 *     <natoms>
 *     Lattice="ax ay az bx by bz cx cy cz" Properties=... key=value ...
 *     <symbol> <x> <y> <z> [<dx> <dy> <dz>]
 *     ...
 *
 * Two deliberate choices:
 *
 *   - The LATTICE written is the tiled block's, not the primitive cell's.
 *     The frames contain the tiled atoms, so the primitive lattice would be
 *     inconsistent with them and any tool computing a density or applying
 *     periodic boundary conditions would get it wrong.
 *
 *   - Displacements are written as a per-atom vector column alongside the
 *     positions. OVITO renders those directly as arrows, and it makes the
 *     file self-describing: the motion is recoverable even from a single
 *     frame, without differencing.
 *
 * Pure string building — no DOM, no engine calls. The caller supplies frames
 * that it generated through kinematics.js, so this module cannot introduce a
 * second displacement convention.
 */
(function (global) {
  "use strict";

  /* Fixed-decimal formatter. Deliberately not toPrecision: extxyz readers are
   * whitespace-tokenized and exponent forms like 1e-7 are legal but noisy,
   * while a fixed 6 decimals in Angstrom is ~1e-6 A resolution — far below
   * anything a phonon amplitude means. Also normalizes "-0.000000" to
   * "0.000000", which some stricter parsers dislike. */
  function fx(v, nd) {
    var s = (typeof v === "number" && isFinite(v) ? v : 0).toFixed(nd === undefined ? 6 : nd);
    return (/^-0\.0*$/.test(s)) ? s.slice(1) : s;
  }

  /* extxyz comment-line values: quote anything with whitespace, and escape
   * embedded quotes, so a material label with a space cannot break the line. */
  function kv(key, value) {
    var s = String(value);
    if (/[\s"]/.test(s)) return key + '="' + s.replace(/"/g, '\\"') + '"';
    return key + "=" + s;
  }

  /* Build a multi-frame extxyz document.
   *
   * spec = {
   *   symbols  : [Natoms]            chemical symbols, one per atom
   *   lattice  : [[3],[3],[3]]       ROW vectors of the exported cell (A)
   *   frames   : [ { positions: [Natoms][3], displacements?: [Natoms][3],
   *                  phase?: number } ]
   *   meta     : { ... }             extra key=value pairs for every frame
   * }
   */
  function buildExtxyz(spec) {
    var symbols = spec.symbols, lattice = spec.lattice, frames = spec.frames;
    if (!symbols || !symbols.length) throw new Error("trajectory: no symbols");
    if (!frames || !frames.length) throw new Error("trajectory: no frames");
    if (!lattice || lattice.length !== 3) throw new Error("trajectory: lattice must be 3x3");

    var n = symbols.length;
    var latStr = [];
    for (var i = 0; i < 3; i++) {
      for (var j = 0; j < 3; j++) latStr.push(fx(lattice[i][j]));
    }

    var meta = spec.meta || {};
    var metaKeys = Object.keys(meta);
    var out = [];

    for (var f = 0; f < frames.length; f++) {
      var fr = frames[f];
      var pos = fr.positions, disp = fr.displacements || null;
      if (!pos || pos.length !== n) {
        throw new Error("trajectory: frame " + f + " has " + (pos ? pos.length : 0)
                        + " positions for " + n + " symbols");
      }

      var props = disp ? "species:S:1:pos:R:3:disp:R:3" : "species:S:1:pos:R:3";
      var head = ['Lattice="' + latStr.join(" ") + '"', "Properties=" + props,
                  kv("frame", f), kv("nframes", frames.length)];
      if (typeof fr.phase === "number") head.push(kv("phase_rad", fx(fr.phase, 6)));
      for (var k = 0; k < metaKeys.length; k++) {
        head.push(kv(metaKeys[k], meta[metaKeys[k]]));
      }

      out.push(String(n));
      out.push(head.join(" "));
      for (var a = 0; a < n; a++) {
        var row = symbols[a] + " " + fx(pos[a][0]) + " " + fx(pos[a][1]) + " " + fx(pos[a][2]);
        if (disp) {
          var d = disp[a] || [0, 0, 0];
          row += " " + fx(d[0]) + " " + fx(d[1]) + " " + fx(d[2]);
        }
        out.push(row);
      }
    }
    return out.join("\n") + "\n";
  }

  /* The tiled block's lattice: the primitive row vectors scaled by the tile
   * counts. Exported alongside the tiled atoms so the two agree. */
  function tiledLattice(primitive, n1, n2, n3) {
    var t = [n1 || 1, n2 || 1, n3 || 1], out = [];
    for (var i = 0; i < 3; i++) {
      out.push([primitive[i][0] * t[i], primitive[i][1] * t[i], primitive[i][2] * t[i]]);
    }
    return out;
  }

  /* Phases for a closed loop of nFrames samples over one period.
   * The endpoint is EXCLUDED: including both 0 and 2*pi would duplicate a
   * frame and make a looping animation stutter for one tick. */
  function loopPhases(nFrames) {
    var n = Math.max(1, nFrames | 0), out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = 2 * Math.PI * i / n;
    return out;
  }

  global.PVTrajectory = {
    buildExtxyz: buildExtxyz,
    tiledLattice: tiledLattice,
    loopPhases: loopPhases
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
