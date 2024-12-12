from typing import Tuple

import numpy as np
from ase import Atoms

from atomatic import _ext


def compute_rmsd(atoms_1: Atoms, atoms_2: Atoms, compute_grad: bool = False) -> float | Tuple[float, np.ndarray]:
    if atoms_1.pbc.any() or atoms_2.pbc.any():
        raise NotImplementedError("PBC not supported yet.")
    # Check system equivalency
    cond_1 = len(atoms_1) != len(atoms_2)
    cond_2 = (atoms_1.numbers != atoms_2.numbers).any()
    if cond_1 or cond_2:
        raise ValueError("Two Atoms objects represent different systems.")
    x1 = atoms_1.get_positions()
    x2 = atoms_2.get_positions()
    rmsd, rmsd_grad = _ext.compute_rmsd(x1, x2, compute_grad)
    if compute_grad:
        return rmsd, rmsd_grad
    return rmsd
