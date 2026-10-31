/* polyhedra.js — coordination polyhedra for the atom panel.
 *
 * A coordination polyhedron is the convex hull of the neighbours bonded to a
 * centre atom: SnSe6 is an octahedron, a tetrahedral site is a tetrahedron.
 * Drawing them is standard structural chemistry; doing it DURING a phonon
 * animation is the point of this module, because the way a polyhedron
 * deforms names the mode:
 *
 *   - all vertices out together        -> a breathing mode
 *   - rigid rotation about the centre  -> a tilt/libration mode
 *   - opposite faces sliding           -> a shear mode
 *
 * Design note that matters for animation quality: the hull TOPOLOGY is
 * computed once from the equilibrium geometry and then held fixed while the
 * vertices follow the displacements. Recomputing the hull every frame would
 * let faces flip whenever a vibration makes four vertices momentarily
 * coplanar, which reads as flicker. Phonon amplitudes are small compared with
 * bond lengths, so the equilibrium topology stays valid throughout.
 *
 * Pure geometry: no rendering, no DOM. Feeds plain triangle indices to
 * whatever draws them.
 */
(function (global) {
  "use strict";

  /* Faces of the convex hull of a small 3D point set.
   *
   * Brute force over candidate triples: a triple spans a supporting plane of
   * the hull exactly when every other point lies on one side of it. For
   * coordination shells (n typically 3-12) that is a few hundred cheap tests
   * — far simpler to get right than an incremental hull, and with no
   * degenerate-input failure modes to nurse.
   *
   * COPLANAR FACETS are the subtle part, and the reason this is not just a
   * loop over triples. A square facet (square-planar coordination, a cube,
   * a cuboctahedron) has four points on one plane and therefore FOUR
   * supporting triples, each covering half the square. Emitting all of them
   * double-covers the facet — which silently doubles any surface or volume
   * computed from the mesh. So supporting triples are grouped by the PLANE
   * they span, and each distinct plane is triangulated once as a fan around
   * its own centroid.
   *
   * points: [[x,y,z], ...] in any consistent length unit.
   * Returns [[i,j,k], ...] wound counter-clockwise seen from outside (normal
   * pointing away from the interior), each facet covered exactly once.
   */
  function convexHullFaces(points, tolerance) {
    var n = points.length;
    var tol = tolerance === undefined ? 1e-6 : tolerance;
    if (n < 3) return [];
    if (n === 3) return [[0, 1, 2], [0, 2, 1]];       // flat: both orientations

    // interior reference point: the centroid is inside any convex hull
    var cx = 0, cy = 0, cz = 0, i, j, k, p;
    for (i = 0; i < n; i++) { cx += points[i][0]; cy += points[i][1]; cz += points[i][2]; }
    cx /= n; cy /= n; cz /= n;

    var planes = [];          // {nx,ny,nz,d, members:[...]}

    for (i = 0; i < n; i++) {
      for (j = i + 1; j < n; j++) {
        for (k = j + 1; k < n; k++) {
          var a = points[i], b = points[j], c = points[k];
          var ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
          var vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
          var nx = uy * vz - uz * vy;
          var ny = uz * vx - ux * vz;
          var nz = ux * vy - uy * vx;
          var len = Math.sqrt(nx * nx + ny * ny + nz * nz);
          if (len < tol) continue;                    // collinear triple
          nx /= len; ny /= len; nz /= len;
          var d = nx * a[0] + ny * a[1] + nz * a[2];

          var pos = 0, neg = 0;
          for (p = 0; p < n; p++) {
            var sd = nx * points[p][0] + ny * points[p][1] + nz * points[p][2] - d;
            if (sd > tol) pos++;
            else if (sd < -tol) neg++;
            if (pos && neg) break;
          }
          if (pos && neg) continue;                   // straddles: interior plane

          // canonical outward orientation, so the same facet found via
          // different triples collapses to ONE plane record
          var cd = nx * cx + ny * cy + nz * cz - d;
          if (cd > 0) { nx = -nx; ny = -ny; nz = -nz; d = -d; }

          var found = null;
          for (var pi = 0; pi < planes.length; pi++) {
            var pl = planes[pi];
            if (Math.abs(pl.nx - nx) < 1e-6 && Math.abs(pl.ny - ny) < 1e-6 &&
                Math.abs(pl.nz - nz) < 1e-6 && Math.abs(pl.d - d) < 1e-6) {
              found = pl; break;
            }
          }
          if (!found) {
            found = { nx: nx, ny: ny, nz: nz, d: d, members: [] };
            planes.push(found);
          }
          if (found.members.indexOf(i) < 0) found.members.push(i);
          if (found.members.indexOf(j) < 0) found.members.push(j);
          if (found.members.indexOf(k) < 0) found.members.push(k);
        }
      }
    }

    var faces = [];
    for (var f = 0; f < planes.length; f++) {
      var pl2 = planes[f], mem = pl2.members;
      if (mem.length < 3) continue;

      // in-plane orthonormal basis, then order the facet's points by angle
      // about their centroid so the fan is a simple (non self-crossing) polygon
      var ax = Math.abs(pl2.nx) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      var e1x = ax[1] * pl2.nz - ax[2] * pl2.ny;
      var e1y = ax[2] * pl2.nx - ax[0] * pl2.nz;
      var e1z = ax[0] * pl2.ny - ax[1] * pl2.nx;
      var e1n = Math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
      e1x /= e1n; e1y /= e1n; e1z /= e1n;
      var e2x = pl2.ny * e1z - pl2.nz * e1y;
      var e2y = pl2.nz * e1x - pl2.nx * e1z;
      var e2z = pl2.nx * e1y - pl2.ny * e1x;

      var gx = 0, gy = 0, gz = 0, mi;
      for (mi = 0; mi < mem.length; mi++) {
        gx += points[mem[mi]][0]; gy += points[mem[mi]][1]; gz += points[mem[mi]][2];
      }
      gx /= mem.length; gy /= mem.length; gz /= mem.length;

      var ordered = mem.slice().sort(function (u, w) {
        var pu = points[u], pw = points[w];
        var au = Math.atan2((pu[0] - gx) * e2x + (pu[1] - gy) * e2y + (pu[2] - gz) * e2z,
                            (pu[0] - gx) * e1x + (pu[1] - gy) * e1y + (pu[2] - gz) * e1z);
        var aw = Math.atan2((pw[0] - gx) * e2x + (pw[1] - gy) * e2y + (pw[2] - gz) * e2z,
                            (pw[0] - gx) * e1x + (pw[1] - gy) * e1y + (pw[2] - gz) * e1z);
        return au - aw;
      });

      // fan from the first vertex; check one triangle's winding against the
      // facet normal and flip the whole fan if needed
      var v0 = points[ordered[0]], v1 = points[ordered[1]], v2 = points[ordered[2]];
      var wx = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0]);
      var wy = (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1]);
      var wz = (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2]);
      var flip = (wy * pl2.nx + wz * pl2.ny + wx * pl2.nz) < 0;

      for (var t = 1; t + 1 < ordered.length; t++) {
        faces.push(flip ? [ordered[0], ordered[t + 1], ordered[t]]
                        : [ordered[0], ordered[t], ordered[t + 1]]);
      }
    }
    return faces;
  }

  /* Build the polyhedron descriptors for a structure.
   *
   * centres  : [atomIndex, ...]        atoms that get a polyhedron
   * neighbours(centreIndex) -> [atomIndex, ...]   its bonded partners
   * positionOf(atomIndex) -> [x,y,z]   EQUILIBRIUM position
   * minVertices: skip centres with fewer neighbours than this (default 4 —
   *   three or fewer cannot enclose a volume, so there is nothing to draw)
   *
   * Returns [{ centre, vertexAtoms, faces }] where `faces` indexes into
   * `vertexAtoms`. Topology only — no coordinates are retained, so the caller
   * is free to re-read displaced positions each frame.
   */
  function buildPolyhedra(centres, neighbours, positionOf, options) {
    options = options || {};
    var minVertices = options.minVertices === undefined ? 4 : options.minVertices;
    var tol = options.tolerance;
    var out = [];

    for (var ci = 0; ci < centres.length; ci++) {
      var centre = centres[ci];
      var verts = neighbours(centre) || [];
      if (verts.length < minVertices) continue;

      var pts = new Array(verts.length);
      for (var vi = 0; vi < verts.length; vi++) pts[vi] = positionOf(verts[vi]);

      var faces = convexHullFaces(pts, tol);
      if (!faces.length) continue;                    // fully degenerate shell
      out.push({ centre: centre, vertexAtoms: verts.slice(), faces: faces });
    }
    return out;
  }

  /* Flat Float32Array of triangle vertices for a polyhedron at the CURRENT
   * (displaced) positions — call once per frame per polyhedron.
   *
   * Writes into `target` when supplied so the animation loop can reuse one
   * buffer instead of allocating; the array is 9 floats per face. */
  function triangleVertices(poly, positionOf, target) {
    var faces = poly.faces, va = poly.vertexAtoms;
    var need = faces.length * 9;
    var arr = (target && target.length === need) ? target : new Float32Array(need);
    for (var f = 0, o = 0; f < faces.length; f++) {
      for (var c = 0; c < 3; c++) {
        var p = positionOf(va[faces[f][c]]);
        arr[o++] = p[0]; arr[o++] = p[1]; arr[o++] = p[2];
      }
    }
    return arr;
  }

  /* Volume of the polyhedron at the given positions (Angstrom^3), via the
   * divergence theorem over its oriented faces.
   *
   * Exposed because it is the quantitative read-out of the qualitative
   * picture: watch it oscillate and you are watching a breathing mode; watch
   * it stay flat while the shape changes and the mode is a tilt or a shear.
   * Sign is positive for outward-wound faces, so a negative result means the
   * winding broke. */
  function volume(poly, positionOf) {
    var faces = poly.faces, va = poly.vertexAtoms, vol = 0;
    for (var f = 0; f < faces.length; f++) {
      var a = positionOf(va[faces[f][0]]);
      var b = positionOf(va[faces[f][1]]);
      var c = positionOf(va[faces[f][2]]);
      vol += (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
    }
    return vol;
  }

  global.PVPolyhedra = {
    convexHullFaces: convexHullFaces,
    buildPolyhedra: buildPolyhedra,
    triangleVertices: triangleVertices,
    volume: volume
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
