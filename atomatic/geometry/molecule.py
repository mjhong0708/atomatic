import functools
import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols, covalent_radii
from ase.formula import Formula
from matscipy.neighbours import neighbour_list
from scipy.sparse import coo_matrix
from typing_extensions import Self

CutoffDict = Dict[Tuple[str, str], float]


@functools.cache
def _default_covalent_cutoffs(tol: float = 0.2) -> CutoffDict:
    """Default covalent bond cutoffs for all pairs of elements.
    Args:
        tol (float): Tolerance factor to increase the cutoff radius.
            Default is 0.2.

    Returns:
        Dict[Tuple[str, str], float]: Dictionary of cutoff radii for all pairs of elements.
    """
    cutoffs = {}
    for s1, s2 in itertools.combinations_with_replacement(chemical_symbols, 2):
        z1, z2 = atomic_numbers[s1], atomic_numbers[s2]
        if z1 > z2:
            s2, s1 = s1, s2
        d_12 = (1 + tol) * (covalent_radii[z1] + covalent_radii[z2])
        cutoffs[(s1, s2)] = d_12
    return cutoffs


@dataclass(frozen=True)
class NeighborList:
    """Dataclass to store neighbor list information.
    Args:
        N (int): Number of atoms.
        i (np.ndarray): Indices of atom i in the pair.
        j (np.ndarray): Indices of atom j in the pair.
        d (np.ndarray): Distance between atoms i and j.
        D (np.ndarray): Displacement vector from atom i to atom j.
        S (np.ndarray): Displacement vector from atom i to atom j in the supercell.
    """

    N: int  # Number of atoms
    i: np.ndarray  # (n_pairs,)
    j: np.ndarray  # (n_pairs,)
    d: np.ndarray  # (n_pairs,)
    D: np.ndarray  # (n_pairs, 3)
    S: np.ndarray  # (n_pairs, 3)

    @classmethod
    def from_atoms(cls, atoms: Atoms, cutoff: float | CutoffDict) -> Self:
        """Create a neighbor list from an Atoms object.
        Args:
            atoms (Atoms): Atoms object.
            cutoff (float | Dict[Tuple[str, str], float]): Cutoff radius for neighbors.
                If a float, it is used for all pairs of elements.
                If a dictionary, it is used for specific pairs of elements.

        Returns:
            NeighborList: Neighbor list object.
        """
        if np.linalg.det(atoms.get_cell().array) < 1e-8:
            centroid = atoms.get_positions().mean(axis=0)
            radius = np.linalg.norm(atoms.get_positions() - centroid, axis=1).max()
            max_cutoff = max(cutoff.values()) if isinstance(cutoff, dict) else cutoff
            celldim = max_cutoff * 2 + 10 * radius
            cell = np.eye(3) * celldim
            atoms.cell = cell
        i, j, d, D, S = neighbour_list("ijdDS", atoms, cutoff)
        return cls(len(atoms), i, j, d, D, S)

    def apply_mask(self, mask: np.ndarray) -> Self:
        return NeighborList(self.i[mask], self.j[mask], self.d[mask], self.D[mask], self.S[mask])

    def to_directed(self) -> Self:
        mask = self.i < self.j
        return self.apply_mask(mask)

    def to_coo(self) -> coo_matrix:
        # data is (index of pair in the list) + 1
        data = np.arange(len(self.i)) + 1
        return coo_matrix((data, (self.i, self.j)), shape=(self.N, self.N))


@dataclass(frozen=True)
class MoleculeGraph:
    """Dataclass to store molecular graph information.
    Args:
        indices (np.ndarray): Indices of atoms in the molecule. Default is [0, 1, ..., n_atoms-1].
            The atoms are always represented by indices.
        adj (coo_matrix): Adjacency matrix of the molecule.
        Z (np.ndarray): Atomic numbers of atoms in the molecule.
    """

    indices: np.ndarray  # (n_atoms,)
    adj: coo_matrix  # (n_atoms, n_atoms) in coo format
    Z: np.ndarray  # (n_atoms,)

    def __repr__(self) -> str:
        return f"MoleculeGraph({self.formula}, n_atoms={self.n_atoms}, n_bonds={self.n_bonds})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def formula(self) -> str:
        symbols = [chemical_symbols[z] for z in self.Z]
        return Formula.from_list(symbols).format("hill")

    @property
    def n_atoms(self) -> int:
        return len(self.Z)

    @property
    def n_bonds(self) -> int:
        return self.adj.nnz // 2

    def get_species_bonds(self, s1: str, s2: str) -> List[Tuple[int, int]]:
        """Get bonds between two species.
        Args:
            s1 (str): Symbol of the first species.
            s2 (str): Symbol of the second species.

        Returns:
            List[Tuple[int, int]]: List of tuples (i, j) where i and j are indices of atoms
            of species s1 and s2 respectively.
        """
        z1 = atomic_numbers[s1]
        z2 = atomic_numbers[s2]
        i, j = self.adj.row, self.adj.col
        mask = (self.Z[i] == z1) & (self.Z[j] == z2)
        idx_i, idx_j = self.indices[i[mask]], self.indices[j[mask]]
        return list(zip(idx_i, idx_j, strict=True))

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        indices: Optional[np.ndarray] = None,
        adj: Optional[coo_matrix] = None,
        bond_tol: float = 0.2,
        extra_cutoffs: Optional[CutoffDict] = None,
    ) -> Self:
        if indices is not None and len(indices) != len(atoms):
            raise ValueError("Indices must have the same length as the number of atoms.")

        cutoffs = _default_covalent_cutoffs(bond_tol)
        if extra_cutoffs is not None:
            cutoffs.update(extra_cutoffs)
        Z = atoms.get_atomic_numbers()
        indices = np.asarray(indices) if indices is not None else np.arange(len(atoms))
        adj = adj if adj is not None else NeighborList.from_atoms(atoms, cutoffs).to_coo()
        return cls(indices, adj, Z)

    def is_equivalent(self, other: Self) -> bool:
        """Check if two molecule graphs are equivalent"""
        # Quick checks first
        if len(self.Z) != len(other.Z):
            return False

        # Check if atomic composition is the same
        if not np.array_equal(Counter(self.Z), Counter(other.Z)):
            return False

        # Group atoms by atomic number for efficient matching
        self_atoms_by_Z = defaultdict(list)
        other_atoms_by_Z = defaultdict(list)

        for i, z in enumerate(self.Z):
            self_atoms_by_Z[z].append(i)
        for i, z in enumerate(other.Z):
            other_atoms_by_Z[z].append(i)

        def is_compatible(idx1: int, idx2: int, mapping: dict[int, int]) -> bool:
            """Check if two atoms can be mapped to each other"""
            # Get neighbors of both atoms
            neighbors1 = self.get_neighbors(idx1)
            neighbors2 = other.get_neighbors(idx2)

            # Must have same number of neighbors
            if len(neighbors1) != len(neighbors2):
                return False

            # Check if mapped neighbors match
            mapped_neighbors1 = set()
            for n1 in neighbors1:
                if n1 in mapping:
                    mapped_neighbors1.add(mapping[n1])

            mapped_neighbors2 = set(neighbors2)
            if not mapped_neighbors1.issubset(mapped_neighbors2):
                return False

            return True

        def find_isomorphism(mapping: dict[int, int]) -> bool:
            """Recursive backtracking to find isomorphic mapping"""
            if len(mapping) == len(self.Z):
                return True

            # Find next unmapped atom
            for z in self_atoms_by_Z:
                for idx1 in self_atoms_by_Z[z]:
                    if idx1 not in mapping:
                        break
                else:
                    continue
                break

            # Try mapping to each compatible atom of same element
            for idx2 in other_atoms_by_Z[z]:
                if idx2 not in mapping.values():
                    if is_compatible(idx1, idx2, mapping):
                        # Try this mapping
                        mapping[idx1] = idx2
                        if find_isomorphism(mapping):
                            return True
                        del mapping[idx1]

            return False

        # Start with empty mapping and try to find isomorphism
        return find_isomorphism({})

    def get_bonded_atoms(self, idx):
        """Get indices of atoms connected to atom idx"""
        # Get both rows and cols where idx appears
        row_neighbors = self.adj.col[self.adj.row == idx]
        col_neighbors = self.adj.row[self.adj.col == idx]
        return np.unique(np.concatenate([row_neighbors, col_neighbors]))


