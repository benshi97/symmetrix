import warnings

import numpy as np
from ase.calculators.calculator import BaseCalculator, all_changes
from ase.calculators.calculator import Calculator


class MACEFoundation(BaseCalculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model_file, device="cuda", dtype="float64", skin=1.0):
        from ._mace_foundation import MACEFoundation as Evaluator

        warnings.warn(
            "MACEFoundation is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        BaseCalculator.__init__(self)
        self.evaluator = Evaluator(str(model_file), device, dtype, skin)

    def get_spin_polarized(self):
        return False

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        a = self.atoms
        energy, forces, virial, energies = self.evaluator.compute(
            a.positions, a.numbers, a.cell.array, a.pbc, "energies" in properties
        )
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if energies is not None:
            self.results["energies"] = energies
        if a.pbc.all():
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / a.get_volume()


class NequIPFoundation(BaseCalculator):
    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]

    def __init__(self, model_file, device="cuda", dtype="float64", skin=1.0):
        from ._nequip_foundation import NequIPFoundation as Evaluator

        warnings.warn(
            "NequIPFoundation is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )
        BaseCalculator.__init__(self)
        self.evaluator = Evaluator(str(model_file), device, dtype, skin)

    def get_spin_polarized(self):
        return False

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()
        a = self.atoms
        energy, forces, virial, energies = self.evaluator.compute(
            a.positions, a.numbers, a.cell.array, a.pbc, "energies" in properties
        )
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}
        if energies is not None:
            self.results["energies"] = energies
        if a.pbc.all():
            self.results["stress"] = -virial[[0, 1, 2, 5, 4, 3]] / a.get_volume()


class MACEPolar(Calculator):
    """Experimental ASE interface to Symmetrix's PolarMACE engines.

    ``dtype=None`` selects float64 on CPU and float32 on CUDA. CUDA also
    supports ``float32-mixed``; float64 CUDA and float32 CPU are unsupported.
    Fully open molecules use real-space electrostatics. Fully periodic systems
    use reciprocal-space sums and support stress.

    ``charge`` is the total charge in electron units. ``spin`` follows the
    Symmetrix convention: spin-channel charges are (charge +/- (spin - 1))/2.
    These are explicit parameters, not inferred from atoms.info or initial
    charges/magnetic moments. Update them with ``calc.set(charge=..., spin=...)``.
    Model, device, dtype and skin are fixed for the calculator instance.
    """

    implemented_properties = ["energy", "free_energy", "energies", "forces", "stress"]
    default_parameters = {"charge": 0.0, "spin": 1.0}

    def __init__(
        self, model_file, device="cuda", dtype=None, skin=1.0, charge=0.0, spin=1.0
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if dtype is None:
            dtype = "float64" if device == "cpu" else "float32"
        supported = ("float64",) if device == "cpu" else ("float32", "float32-mixed")
        if dtype not in supported:
            raise ValueError(
                f"PolarMACE on {device} supports {supported}, not {dtype!r}. "
                "Use device='cpu' for float64 calculations."
            )
        skin = float(skin)
        if not np.isfinite(skin) or skin < 0:
            raise ValueError("skin must be finite and nonnegative")

        Calculator.__init__(self, charge=charge, spin=spin)
        from ._mace_polar import MACEPolar as Evaluator

        self.model_file = str(model_file)
        self.device = device
        self.dtype = dtype
        self.skin = skin
        self.evaluator = Evaluator(self.model_file, device, dtype, skin)
        warnings.warn(
            "MACEPolar is experimental: its interface, model format and results may "
            "change between releases",
            stacklevel=2,
        )

    def set(self, **kwargs):
        unknown = set(kwargs) - {"charge", "spin"}
        if unknown:
            raise TypeError(
                "Only charge and spin can be updated with set(); create a new "
                f"MACEPolar calculator to change {', '.join(sorted(unknown))}"
            )
        for key, value in kwargs.items():
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            kwargs[key] = value
        changes = Calculator.set(self, **kwargs)
        if changes:
            self.reset()
        return changes

    def get_spin_polarized(self):
        return self.parameters.spin != 1.0

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        a = self.atoms
        periodic = a.pbc.all()
        if a.pbc.any() and not periodic:
            raise ValueError("MACEPolar requires fully open or fully periodic boundaries")
        if len(a) == 0:
            raise ValueError("MACEPolar requires at least one atom")
        if not np.isfinite(a.positions).all() or not np.isfinite(a.cell.array).all():
            raise ValueError("MACEPolar positions and cell must be finite")
        if periodic:
            volume = abs(np.linalg.det(a.cell.array))
            if not np.isfinite(volume) or volume == 0:
                raise ValueError("MACEPolar requires a nonsingular periodic cell")
        energy, forces, virial, energies = self.evaluator.compute(
            a.positions, a.numbers, a.cell.array, a.pbc,
            self.parameters.charge, self.parameters.spin, "energies" in properties,
        )
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.asarray(forces),
        }
        if periodic:
            # Backend: xx yy zz xy xz yz. ASE: xx yy zz yz xz xy.
            self.results["stress"] = -np.asarray(virial)[[0, 1, 2, 5, 4, 3]] / volume
        if energies is not None:
            self.results["energies"] = np.asarray(energies)
