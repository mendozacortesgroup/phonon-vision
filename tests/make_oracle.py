"""Generate tests-js/_oracle.json — the Python-computed expected values that
the JavaScript parity test asserts against.

Run this whenever phonon_adp.py changes:

    python tests/make_oracle.py

The cases are deliberately small (a handful of q-points) so the fixture stays
readable and the parity failure, when it comes, points at a specific mode
rather than at a 340-node blur. One case IS taken from the real SnSe2 payload,
subsampled, so the parity check also covers realistic magnitudes and a genuine
mass ratio.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))

from phonon_adp import adp_tensors  # noqa: E402


def case(mesh, masses, temperature, freq_floor):
    U, info = adp_tensors(
        np.array(mesh["frequency"]),
        np.array(mesh["eigenvectors_real"]),
        np.array(mesh["eigenvectors_imag"]),
        np.array(masses), temperature, freq_floor_thz=freq_floor)
    return {
        "mesh": mesh, "masses": masses, "temperature": temperature,
        "freqFloor": freq_floor,
        "expect": {"U": U.tolist(), "nModesDropped": info["n_modes_dropped"]},
    }


def synthetic():
    """Two atoms, two q-points, complex eigenvectors — exercises the
    Re[e conj(e)] path including the imaginary contribution."""
    rng = np.random.default_rng(20260723)
    nq, nb, na = 2, 4, 2
    er = rng.normal(size=(nq, nb, na, 3))
    ei = rng.normal(size=(nq, nb, na, 3))
    norm = np.sqrt((er ** 2 + ei ** 2).sum(axis=(2, 3), keepdims=True))
    er, ei = er / norm, ei / norm
    freq = np.array([[0.05, 2.5, 6.0, 11.0], [1.2, 3.3, 7.7, 12.5]])
    return {
        "frequency": freq.tolist(),
        "eigenvectors_real": er.tolist(),
        "eigenvectors_imag": ei.tolist(),
    }, [12.0107, 195.084]


def from_payload(slug, n_nodes=6):
    path = os.path.join(HERE, "..", "web", f"navigator_data_{slug}.js")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.loads(fh.read().split("=", 1)[1].strip().rstrip(";"))
    mm = d["mesh_modes"]
    step = max(1, len(mm["frequency"]) // n_nodes)
    sel = list(range(0, len(mm["frequency"]), step))[:n_nodes]
    mesh = {
        "frequency": [mm["frequency"][i] for i in sel],
        "eigenvectors_real": [mm["eigenvectors_real"][i] for i in sel],
        "eigenvectors_imag": [mm["eigenvectors_imag"][i] for i in sel],
    }
    return mesh, d["meta"]["material"]["masses"]


def main():
    cases = {}

    mesh, masses = synthetic()
    cases["synthetic_300K"] = case(mesh, masses, 300.0, 0.1)
    cases["synthetic_10K_nofloor"] = case(mesh, masses, 10.0, 0.0)
    cases["synthetic_1200K"] = case(mesh, masses, 1200.0, 0.1)

    real = from_payload("snse2")
    if real:
        cases["snse2_subsampled_300K"] = case(real[0], real[1], 300.0, 0.1)
        cases["snse2_subsampled_80K"] = case(real[0], real[1], 80.0, 0.1)

    out = os.path.join(HERE, "..", "tests-js", "_oracle.json")
    with open(out, "w") as fh:
        json.dump({"generated_by": "tests/make_oracle.py", "cases": cases}, fh, indent=1)
    print(f"wrote {out} with {len(cases)} cases: {', '.join(cases)}")


if __name__ == "__main__":
    main()
