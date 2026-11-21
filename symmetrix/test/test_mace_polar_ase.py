"""ASE interface tests; real-model evaluations require explicit local fixtures."""

import json
import os
import sys
import types
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.calculators.singlepoint import SinglePointCalculator

from symmetrix.experimental import MACEPolar


def water():
    return Atoms(
        "OH2OH2",
        positions=[
            [12.2, 0.9, 1.1], [13.16, 0.9, 1.1], [11.96, 1.83, 1.1],
            [6.0, 6.0, 6.0], [6.96, 6.0, 6.0], [5.76, 6.93, 6.0],
        ],
        cell=[12.42, 12.42, 12.42], pbc=True,
    )


@pytest.fixture
def fake_backend(monkeypatch):
    class Fake:
        def __init__(self, *args):
            self.args = args
            self.calls = []

        def compute(self, x, numbers, cell, pbc, charge, spin, energies):
            self.calls.append((x.copy(), cell.copy(), charge, spin, energies))
            energy = 12.0 + charge + spin
            return (
                energy, np.full_like(x, 2.0), np.arange(1.0, 7.0),
                np.full(len(numbers), energy / len(numbers)) if energies else None,
            )

    monkeypatch.setitem(sys.modules, "symmetrix._mace_polar", types.SimpleNamespace(MACEPolar=Fake))
    return Fake


def make(model="unused.json", **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MACEPolar is experimental")
        return MACEPolar(model, **kwargs)


@pytest.mark.parametrize("device,dtype", [("cpu", "float64"), ("cuda", "float32")])
def test_defaults(fake_backend, device, dtype):
    calc = make(device=device)
    assert calc.dtype == dtype
    assert calc.evaluator.args == ("unused.json", device, dtype, 1.0)


@pytest.mark.parametrize("kwargs", [
    {"device": "bad"}, {"device": "cuda", "dtype": "float64"},
    {"device": "cpu", "dtype": "float32"}, {"device": "cpu", "dtype": "float32-mixed"},
    {"skin": -1}, {"skin": float("nan")}, {"charge": float("inf")}, {"spin": float("nan")},
])
def test_bad_options(fake_backend, kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_results_and_parameters(fake_backend):
    a = water()
    a.info["charge"] = 99.0  # Explicit calculator parameters control this interface.
    a.calc = make(charge=0.0, spin=1.0)
    assert a.get_potential_energy() == 13.0
    assert a.get_potential_energy(force_consistent=True) == 13.0
    np.testing.assert_array_equal(a.get_forces(), np.full((6, 3), 2.0))
    np.testing.assert_allclose(a.get_stress(), -np.array([1, 2, 3, 6, 5, 4]) / a.get_volume())
    assert len(a.calc.evaluator.calls) == 1
    assert a.calc.evaluator.calls[-1][2:4] == (0.0, 1.0)
    assert sum(a.get_potential_energies()) == pytest.approx(a.get_potential_energy())
    a.calc.set(charge=1.0, spin=2.0)
    assert a.get_potential_energy() == 15.0
    assert a.calc.evaluator.calls[-1][2:4] == (1.0, 2.0)
    assert a.calc.get_spin_polarized()
    with pytest.raises(TypeError, match="create a new"):
        a.calc.set(device="cpu")


@pytest.mark.parametrize("kind", ["nonperiodic", "slab", "singular", "nan", "empty"])
def test_bad_geometry(fake_backend, kind):
    a = water()
    if kind == "nonperiodic":
        a.pbc = False
    elif kind == "slab":
        a.pbc = [True, True, False]
    elif kind == "singular":
        a.cell[2] = 0.0
    elif kind == "nan":
        a.positions[0, 0] = np.nan
    else:
        a = Atoms(cell=[12, 12, 12], pbc=True)
    a.calc = make()
    with pytest.raises(ValueError):
        a.get_potential_energy()
    assert not a.calc.evaluator.calls


def test_sum_calculator(fake_backend):
    a = water()
    correction = SinglePointCalculator(a, energy=2.0, forces=np.ones((6, 3)), stress=np.zeros(6))
    # SinglePointCalculator retrieves stored results but does not declare them
    # in implemented_properties, which ASE's Mixer uses for its intersection.
    correction.implemented_properties = ["energy", "forces", "stress"]
    a.calc = SumCalculator([make(), correction])
    assert a.get_potential_energy() == 15.0
    np.testing.assert_allclose(a.get_forces(), 3.0)


def test_wrong_model_type(tmp_path):
    model = tmp_path / "wrong.json"
    model.write_text(json.dumps({"format_version": 3, "model_type": "NequIP"}))
    with pytest.raises((RuntimeError, ValueError), match="PolarMACE"):
        make(model, device="cpu")


@pytest.fixture(scope="module")
def polar_paths():
    names = ("SYMMETRIX_POLAR_MODEL_JSON", "SYMMETRIX_POLAR_TORCH_MODEL")
    values = [os.environ.get(name) for name in names]
    if not all(values):
        pytest.skip("set both local PolarMACE model paths to opt in to reference checks")
    for path in values:
        if not os.path.isfile(path):
            pytest.fail(f"PolarMACE fixture does not exist: {path}")
    return values


@pytest.fixture(scope="module")
def reference(polar_paths):
    from mace.calculators import mace_polar

    return mace_polar(model=polar_paths[1], device="cpu", default_dtype="float64")


def reference_values(a, reference, charge=0.0, spin=1.0):
    a = a.copy()
    a.info.update(charge=charge, spin=spin)
    reference.reset()
    a.calc = reference
    energy, forces = a.get_potential_energy(), a.get_forces()
    # In MACE 0.3.16 the analytic PolarMACE stress omits a contribution:
    # for a charged periodic fixture it disagrees with its own energy derivative.
    # Differentiate the original model energy instead of inheriting that error.
    stress = finite_difference_stress(a, reference)
    return energy, forces, stress


def finite_difference_stress(a, calculator, h=1e-4):
    stress = []
    for i, j in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        values = []
        for sign in (1, -1):
            strain = np.eye(3)
            strain[i, j] += sign*h if i == j else sign*h/2
            if i != j:
                strain[j, i] += sign*h/2
            b = a.copy()
            b.set_cell(a.cell.array @ strain, scale_atoms=True)
            b.calc = calculator
            calculator.reset()
            values.append(b.get_potential_energy())
        stress.append((values[0] - values[1]) / (2*h*a.get_volume()))
    return np.array(stress)


@pytest.fixture(scope="module")
def reference_cache():
    return {}


@pytest.mark.polar_reference
@pytest.mark.parametrize("device,dtype", [
    ("cpu", "float64"), ("cuda", "float32"), ("cuda", "float32-mixed"),
])
@pytest.mark.parametrize("triclinic", [False, True])
@pytest.mark.parametrize("charge,spin", [(0.0, 1.0), (1.0, 2.0)])
def test_polar_reference(polar_paths, reference, reference_cache, device, dtype, triclinic, charge, spin):
    a = water()
    if triclinic:
        a.set_cell([[12.42, 0, 0], [0.7, 12.3, 0], [-0.2, 0.3, 12.5]])
    key = triclinic, charge, spin
    if key not in reference_cache:
        reference_cache[key] = reference_values(a, reference, charge, spin)
    e, f, s = reference_cache[key]
    a.calc = make(polar_paths[0], device=device, dtype=dtype, charge=charge, spin=spin)
    assert abs(a.get_potential_energy() - e) / len(a) < 1e-5
    assert np.max(np.abs(a.get_forces() - f)) < 1e-3
    assert np.max(np.abs(a.get_stress() - s)) < 1e-4
    assert abs(sum(a.get_potential_energies()) - a.get_potential_energy()) < 1e-9


@pytest.mark.polar_reference
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_polar_list_reuse(polar_paths, device):
    a = water()
    a.calc = make(polar_paths[0], device=device)
    a.get_potential_energy()
    assert a.calc.evaluator.neighbor_list_builds == 1
    for change in ("small", "skin", "wrap", "cell", "species", "count"):
        if change == "small":
            a.positions[0, 0] += 0.02
        elif change == "skin":
            a.positions[:3, 0] += 0.55
        elif change == "wrap":
            a.wrap()
        elif change == "cell":
            a.set_cell(a.cell.array * 1.01, scale_atoms=True)
        elif change == "species":
            a = a[[1, 0, 2, 3, 4, 5]]
            # Preserve the original calculator to test reorder invalidation.
            a.calc = cached
        else:
            a = a[:3]
            a.calc = cached
        cached = a.calc
        f = a.get_forces()
        b = a.copy()
        b.calc = make(polar_paths[0], device=device)
        assert np.max(np.abs(f - b.get_forces())) < 1e-3
        if change == "small":
            assert cached.evaluator.neighbor_list_builds == 1
        else:
            assert cached.evaluator.neighbor_list_builds > 1


@pytest.mark.polar_reference
def test_polar_finite_differences(polar_paths):
    a = water()
    a.calc = make(polar_paths[0], device="cpu")
    f = a.get_forces()
    s = a.get_stress()
    h = 1e-4
    for i, axis in [(0, 0), (1, 2), (3, 1)]:
        values = []
        for sign in (1, -1):
            b = a.copy()
            b.positions[i, axis] += sign*h
            b.calc = a.calc
            values.append(b.get_potential_energy())
        assert abs(-(values[0] - values[1]) / (2*h) - f[i, axis]) < 1e-4
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
        values = []
        for sign in (1, -1):
            strain = np.eye(3)
            strain[i, j] += sign*h if i == j else sign*h/2
            if i != j:
                strain[j, i] += sign*h/2
            b = a.copy()
            b.set_cell(a.cell.array @ strain, scale_atoms=True)
            b.calc = a.calc
            values.append(b.get_potential_energy())
        assert abs((values[0] - values[1]) / (2*h*a.get_volume()) - s[k]) < 1e-5


@pytest.mark.polar_reference
def test_polar_errors(polar_paths):
    a = water()
    a.calc = make(polar_paths[0], device="cpu")
    a.numbers[0] = 2
    with pytest.raises(ValueError, match="not an element"):
        a.get_potential_energy()
    a = water()
    a.positions[1] = a.positions[0] + [0.1, 0, 0]
    a.calc = make(polar_paths[0], device="cpu")
    with pytest.raises(ValueError, match="r_min"):
        a.get_potential_energy()
