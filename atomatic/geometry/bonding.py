from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list
from scipy.sparse.csgraph import connected_components
from typing_extensions import Self

from .molecule import CutoffDict, MoleculeGraph, NeighborList, _default_covalent_cutoffs


class CovalentBondingNetwork:
    def __init__(self, atoms: Atoms, bond_tol: float = 0.2, extra_cutoffs: CutoffDict = None):
        self._bond_cutoffs = _default_covalent_cutoffs(bond_tol)
        self._extra_cutoffs = extra_cutoffs
        self.bond_tol = bond_tol
        if extra_cutoffs is not None:
            self._bond_cutoffs.update(extra_cutoffs)
        self._atoms = atoms.copy()

        # Graph information
        self.nodes = np.arange(len(self._atoms))
        self.Z = self._atoms.get_atomic_numbers()
        self.nbrs = NeighborList.from_atoms(self._atoms, self._bond_cutoffs)
        self.adj = self.nbrs.to_coo()

        # Find all molecules in the structure
        n_components, self._molecule_idx = connected_components(self.nbrs.to_coo(), directed=False)
        self.molecules = []
        for i in range(n_components):
            mask = self._molecule_idx == i
            indices = self.nodes[mask]
            Z_sub = self.Z[mask]
            adj_sub = self.adj.tocsc()[:, mask][mask].tocoo()
            self.molecules.append(MoleculeGraph(indices, adj_sub, Z_sub))

        self._unwrap()  # Unwrap molecules that might be split across periodic boundaries

    def __repr__(self) -> str:
        comp = self.composition
        comp_str = ", ".join([f"{f}: {comp[f]}" for f in comp])
        return (
            f"CovalentBondingNetwork([{comp_str}], n_atoms={self.n_atoms}, "
            f"n_bonds={self.n_bonds}, n_molecules={self.n_molecules})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def n_molecules(self) -> int:
        return len(self.molecules)

    @property
    def n_atoms(self) -> int:
        return len(self.nodes)

    @property
    def n_bonds(self) -> int:
        return sum(mol.n_bonds for mol in self.molecules)

    @property
    def composition(self):
        formulas = [mol.formula for mol in self.molecules]
        return Counter(formulas)

    def is_equivalent(self, other: Self) -> bool:
        g1 = MoleculeGraph.from_atoms(self._atoms, extra_cutoffs=self._extra_cutoffs, bond_tol=self.bond_tol)
        g2 = MoleculeGraph.from_atoms(other._atoms, extra_cutoffs=other._extra_cutoffs, bond_tol=other.bond_tol)
        return g1.is_equivalent(g2)

    def get_species_bonds(self, s1: str, s2: str) -> List[Tuple[int, int]]:
        bonds = []
        for mol in self.molecules:
            bonds.extend(mol.get_species_bonds(s1, s2))
        return bonds

    def is_bonded(self, i: int, j: int) -> bool:
        if i == j:
            return False
        return self.adj.tocsr()[i, j] > 0

    def get_bond_length(self, i: int, j: int) -> float:
        if not self.is_bonded(i, j):
            return np.nan
        idx = self.adj.tocsr()[i, j] - 1
        return self.nbrs.d[idx]

    def find_molecule_idx(self, idx: int) -> Optional[MoleculeGraph]:
        if idx not in self.nodes:
            raise ValueError(f"Index {idx} out of range.")
        return self._molecule_idx[idx]

    def _unwrap(self) -> None:
        """
        Unwrap molecules that might be split across periodic boundaries.
        This modifies self.atoms in-place.
        """
        positions = self._atoms.get_positions()
        cell = self._atoms.get_cell()
        if np.linalg.det(cell) < 1e-8:
            return

        # Alias
        i = self.nbrs.i
        j = self.nbrs.j
        S = self.nbrs.S

        # Process each molecule
        for mol_idx in range(self.n_molecules):
            mol_mask = self._molecule_idx == mol_idx
            mol_atoms = np.where(mol_mask)[0]

            # Start from first atom in molecule
            start_atom = mol_atoms[0]
            processed = {start_atom}
            to_process = [(start_atom, np.zeros(3))]  # (atom_idx, accumulated_shift)

            # Store accumulated shifts for each atom
            shifts = np.zeros((len(self._atoms), 3))

            # Breadth-first search through bonds
            while to_process:
                current, current_shift = to_process.pop(0)

                # Find all bonds involving current atom
                bonds_i = i == current
                bonds_j = j == current

                # Process neighbors through both i->j and j->i bonds
                for idx, shift in zip(j[bonds_i], S[bonds_i], strict=True):
                    if idx not in processed and self._molecule_idx[idx] == mol_idx:
                        new_shift = current_shift + shift
                        shifts[idx] = new_shift
                        processed.add(idx)
                        to_process.append((idx, new_shift))

                for idx, shift in zip(i[bonds_j], -S[bonds_j], strict=True):
                    if idx not in processed and self._molecule_idx[idx] == mol_idx:
                        new_shift = current_shift + shift
                        shifts[idx] = new_shift
                        processed.add(idx)
                        to_process.append((idx, new_shift))

            # Apply accumulated shifts to molecule atoms
            positions[mol_mask] += np.dot(shifts[mol_mask], cell)

        self._atoms.set_positions(positions)


def get_local_cluster(structure: CovalentBondingNetwork, idx: int, cutoff: float = 5.0) -> Atoms:
    """Get a local cluster around an atom.
    Args:
        structure (CovalentBondingNetwork): Molecular structure object.
        idx (int): Index of the central atom.
        cutoff (float): Cutoff radius for the cluster. Default is 5.0 Å.

    Returns:
        Atoms: Local cluster around the central atom.
    """
    atoms = structure._atoms.copy()
    nbrs_i, nbrs_j, nbrs_S = neighbour_list("ijS", atoms, cutoff)
    if atoms.pbc.any():
        cell = atoms.get_cell().array
        offset = np.dot(nbrs_S, cell)
    else:
        offset = np.zeros((len(nbrs_i), 3), dtype=np.float64)

    visited_molecules = {(structure._molecule_idx[idx].item(), (0, 0, 0))}
    mask = nbrs_i == idx
    for j, s in zip(nbrs_j[mask], offset[mask], strict=True):
        mol_with_j = structure.find_molecule_idx(j)
        if mol_with_j not in visited_molecules:
            visited_molecules.add((mol_with_j, (s[0].item(), s[1].item(), s[2].item())))

    cluster = Atoms()
    for mol_idx, shift in visited_molecules:
        indices = structure.molecules[mol_idx].indices
        mol_atoms = atoms[indices].copy()
        mol_atoms.positions += np.asarray(shift)
        cluster.extend(mol_atoms)

    return cluster
