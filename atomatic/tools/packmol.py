import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Sequence, Union

import ase.io
import numpy as np
from ase import Atoms
from ase.units import mol


class PackMol:
    """Main class for interfacing with Packmol."""

    def __init__(
        self,
        tolerance: float,
        seed: int,
        box: tuple[float, float, float] | None = None,
        pbc: bool = False,
    ):
        """
        Initialize PackMol.

        Args:
            tolerance: Minimum distance between molecules
            seed: Random seed for reproducibility
        """
        self.tolerance = tolerance
        self.seed = seed
        self.box = box
        self.pbc = pbc

    def _write_xyz(self, atoms: Atoms, filepath: Path) -> None:
        """Write atoms object to xyz file."""
        atoms.write(filepath, format="xyz")

    def _read_xyz(self, filepath: Path) -> Atoms:
        """Read xyz file into atoms object."""
        return ase.io.read(filepath, format="xyz")

    def _create_input_file(self, structures: List["Structure"], temp_dir: Path, output_file: Path) -> Path:
        """Create Packmol input file."""
        input_lines = [
            f"tolerance {self.tolerance}",
            f"seed {self.seed}",
            "filetype xyz",
            f"output {output_file}",
        ]
        if self.pbc:
            if self.box is None:
                raise ValueError("PBC requires a box size")
            input_lines.append(f"pbc {self.box[0]:.6f} {self.box[1]:.6f} {self.box[2]:.6f}")

        # Write molecule files and create structure blocks
        for i, structure in enumerate(structures):
            mol_file = temp_dir / f"molecule_{i}.xyz"
            self._write_xyz(structure.molecule, mol_file)
            input_lines.extend(structure.get_structure_block(str(mol_file)))

        # Write input file
        input_file = temp_dir / "input.inp"
        with open(input_file, "w") as f:
            f.write("\n".join(input_lines))

        return input_file

    def run(self, structures: List["Structure"], logfile: Optional[Union[str, Path]] = None) -> Atoms:
        """
        Run Packmol with given structures.

        Args:
            structures: List of Structure objects to pack
            logfile: Optional path to write program output

        Returns:
            ASE Atoms object of the packed system

        Raises:
            RuntimeError: If Packmol fails to run or pack the system
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            output_file = temp_dir / "output.xyz"
            input_file = self._create_input_file(structures, temp_dir, output_file)

            # Run Packmol
            with open(input_file, "r") as f:
                process = subprocess.run(
                    ["packmol"],
                    stdin=f,
                    capture_output=True,
                )

            # Handle output
            if logfile is not None:
                with open(logfile, "w") as f:
                    f.write(process.stdout.decode("utf-8"))

            # Check for errors
            if process.returncode != 0:
                raise RuntimeError("Packmol failed to run successfully")

            # Check if output file exists and is non-empty
            if not output_file.exists() or output_file.stat().st_size == 0:
                raise RuntimeError("Packmol failed to generate output")

            result = self._read_xyz(output_file)
        # Postprocess the output
        if self.box is not None:
            result.cell = self.box
        if self.pbc:
            result.pbc = True

        return result


class Structure(ABC):
    """Base class for Packmol structures."""

    def __init__(self, molecule: Atoms, number: int):
        """
        Initialize Structure.

        Args:
            molecule: ASE Atoms object representing the molecule
            number: Number of molecules to place
        """
        self.molecule = molecule
        self.number = number

    @abstractmethod
    def get_constraint_commands(self) -> List[str]:
        """Return the constraint commands for this structure."""
        pass

    def get_structure_block(self, filename: str) -> List[str]:
        """
        Generate the complete structure block for Packmol input.

        Args:
            filename: The xyz file path for this molecule

        Returns:
            List of command lines for this structure
        """
        commands = ["structure " + filename]
        commands.append(f"  number {self.number}")
        if contraint_commands := self.get_constraint_commands():
            commands.extend("  " + cmd for cmd in contraint_commands)
        commands.append("end structure")
        return commands


class Box(Structure):
    """Structure confined to a box region."""

    def __init__(
        self,
        molecule: Atoms,
        number: int,
        origin: tuple[float, float, float],
        sides: tuple[float, float, float],
        outside: bool = False,
    ):
        """
        Initialize Box structure.

        Args:
            molecule: ASE Atoms object
            number: Number of molecules
            origin: (x, y, z) coordinates of box origin
            sides: (dx, dy, dz) side lengths of box
            outside: If True, molecules are placed outside the box
        """
        super().__init__(molecule, number)
        self.origin = origin
        self.sides = sides
        self.outside = outside

    def get_constraint_commands(self) -> List[str]:
        command = f'{"outside" if self.outside else "inside"} box'
        coords = [*self.origin, *(o + s for o, s in zip(self.origin, self.sides))]
        return [f'{command} {" ".join(str(c) for c in coords)}']


class Free(Structure):
    def __init__(self, molecule, number):
        super().__init__(molecule, number)

    def get_constraint_commands(self) -> List[str]:
        return ""


class Sphere(Structure):
    """Structure confined to a spherical region."""

    def __init__(
        self,
        molecule: Atoms,
        number: int,
        center: tuple[float, float, float],
        radius: float,
        outside: bool = False,
    ):
        """
        Initialize Sphere structure.

        Args:
            molecule: ASE Atoms object
            number: Number of molecules
            center: (x, y, z) coordinates of sphere center
            radius: Radius of sphere
            outside: If True, molecules are placed outside the sphere
        """
        super().__init__(molecule, number)
        self.center = center
        self.radius = radius
        self.outside = outside

    def get_constraint_commands(self) -> List[str]:
        command = f'{"outside" if self.outside else "inside"} sphere'
        coords = [*self.center, self.radius]
        return [f'{command} {" ".join(str(c) for c in coords)}']


def calculate_required_volume(molecules: Sequence[Atoms], numbers: Sequence[int], densities: Sequence[float]) -> float:
    """
    Calculate required volume for packing molecules at given densities.

    Args:
        molecules: Sequence of ASE Atoms objects
        numbers: Number of each molecule to pack
        densities: Target density for each molecule type in g/cm³

    Returns:
        Required volume in Å³
    """
    if not (len(molecules) == len(numbers) == len(densities)):
        raise ValueError("Length of molecules, numbers, and densities must match")

    # Calculate molecular masses in g/mol
    masses = [sum(atom.mass for atom in mol) for mol in molecules]

    # Calculate total number of molecules
    total_molecules = sum(numbers)

    # Calculate mole fractions
    mole_fractions = np.array(numbers) / total_molecules

    # Calculate average density in g/cm³ based on mole fractions
    avg_density = sum(d * f for d, f in zip(densities, mole_fractions))

    # Calculate total mass in grams
    total_mass = sum(mass * n / mol for mass, n in zip(masses, numbers))

    # Calculate volume in cm³
    volume_cm3 = total_mass / avg_density

    # Convert to Å³ (1 cm³ = 10^24 Å³)
    volume_ang3 = volume_cm3 * 1e24

    return volume_ang3


def estimate_molecule_numbers(
    molecules: Sequence[Atoms], volume: float, fractions: Sequence[float], densities: Sequence[float]
) -> List[int]:
    """
    Estimate number of molecules needed to fill a volume at given fractions and densities.

    Args:
        molecules: Sequence of ASE Atoms objects
        volume: Volume in Å³
        fractions: Mole fractions for each molecule type (must sum to 1)
        densities: Target density for each molecule type in g/cm³

    Returns:
        List of estimated numbers for each molecule type
    """
    if not (len(molecules) == len(fractions) == len(densities)):
        raise ValueError("Length of molecules, fractions, and densities must match")

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Fractions must sum to 1")

    # Calculate molecular masses in g/mol
    masses = [sum(atom.mass for atom in mol) for mol in molecules]

    # Calculate average molecular mass (g/mol)
    avg_mass = sum(mass * frac for mass, frac in zip(masses, fractions))

    # Calculate average density (g/cm³)
    avg_density = sum(d * f for d, f in zip(densities, fractions))

    # Convert volume from Å³ to cm³
    volume_cm3 = volume * 1e-24

    # Calculate total moles (n = ρV/M)
    total_molecules = int((avg_density * volume_cm3) / (avg_mass / mol))

    # Calculate numbers based on fractions
    raw_numbers = [fraction * total_molecules for fraction in fractions]

    # Round to nearest integers while preserving total
    numbers = np.round(raw_numbers).astype(int)

    # Adjust for rounding errors to maintain exact total
    diff = sum(numbers) - total_molecules
    if diff != 0:
        # Add/subtract from largest number to minimize relative error
        idx = np.argmax(numbers)
        numbers[idx] -= diff

    return list(numbers)
