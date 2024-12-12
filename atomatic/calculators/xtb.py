import os
import re

import numpy as np
from ase import units
from ase.calculators.calculator import FileIOCalculator


def is_debug():
    return os.environ.get("DEBUG", "").lower() in ["1", "true"]


def build_cmd(xtb_cmd, parameters):
    args = [xtb_cmd, "input.xyz"]

    # Charge and spin
    args.append(f"--chrg {parameters['charge']}")
    if (spin := parameters["spin"]) != 0:
        args.append(f"--uhf {spin}")
        args.append("--spinpol --tblite")

    # Solvent
    if (solvent := parameters["solvent"]) is not None:
        args.append(f"--gbsa {solvent.lower()}")

    args.append("--grad")

    return " ".join(args) + " > xtb.out 2>&1"


def extract_energy_and_gradients(file_content):
    # Use regex to find the energy line and extract the energy value from
    # `gradient` file, which is output of xtb program
    energy_match = re.search(r"SCF energy =\s+([-\d.]+)", file_content)
    energy = float(energy_match.group(1)) if energy_match else None

    # Split the file content into lines
    lines = file_content.strip().split("\n")

    # Extract the gradient lines
    num_gradient_lines = (len(lines) - 2) // 2
    gradient_lines = lines[2 + num_gradient_lines : 2 + 2 * num_gradient_lines]  # noqa

    # Convert the gradient lines into a numpy array
    gradients = np.array([list(map(float, line.split()[:3])) for line in gradient_lines])

    return energy, gradients


class XTB(FileIOCalculator):
    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        charge=0,
        spin=0,
        solvent=None,
    )

    def __init__(
        self,
        restart=None,
        ignore_bad_restart_file=FileIOCalculator._deprecated,
        label=None,
        atoms=None,
        **kwargs,
    ):
        xtb_cmd = os.environ.get("ASE_XTB_COMMAND", "xtb -T 32")
        super().__init__(
            restart,
            ignore_bad_restart_file,
            label,
            atoms,
            command=build_cmd(xtb_cmd, self.default_parameters),
            **kwargs,
        )
        self.command = build_cmd(xtb_cmd, self.parameters)
        if is_debug():
            print(self.command)

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)
        filepath = os.path.join(self.directory, "input.xyz")
        atoms.write(filepath)

    def read_results(self):
        output_file = os.path.join(self.directory, "gradient")
        with open(output_file, "r") as f:
            file_content = f.read()

        energy, gradients = extract_energy_and_gradients(file_content)
        energy = energy * units.Hartree
        gradients = gradients * units.Hartree / units.Bohr
        forces = -gradients

        self.results["energy"] = energy
        self.results["forces"] = forces


if __name__ == "__main__":
    from time import perf_counter

    import ase.build
    from ase.data.pubchem import pubchem_atoms_search
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import QuasiNewton

    atoms = ase.build.molecule("H2O")
    t1 = perf_counter()
    atoms.calc = XTB(directory="xtbcalc", charge=0, spin=0)
    t2 = perf_counter()
    print(atoms.get_potential_energy())
    print("Time to run:", t2 - t1)

    t3 = perf_counter()
    atoms.calc = XTB(directory="xtbcalc", charge=0, spin=0, solvent="water")
    t4 = perf_counter()
    print(atoms.get_potential_energy())
    print("Time to run:", t4 - t3)

    atoms = pubchem_atoms_search(name="Caffeine")
    print(atoms)
    t1 = perf_counter()
    atoms.calc = XTB(directory="xtbcalc", charge=0, spin=0)
    opt = QuasiNewton(atoms)
    opt.run(fmax=0.02)
    t2 = perf_counter()
    print(atoms.get_potential_energy())
    print("Time to run:", t2 - t1)

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    md = NVTBerendsen(
        atoms,
        1 * units.fs,
        taut=10 * units.fs,
        temperature_K=300,
        logfile="-",
        trajectory="md.traj",
        loginterval=10,
    )
    md.run(1000)
