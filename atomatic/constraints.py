import numpy as np
from ase.constraints import FixConstraint
from atomatic._ext import shake_utils
import warnings

class SHAKE(FixConstraint):
    """
    Implementation of SHAKE algorithm as a constraint in ASE.
    Maintains fixed distances between specified pairs of atoms.
    """

    maxiter = 500  # Maximum number of iterations for SHAKE

    def __init__(self, pairs, tolerance=1e-12, bondlengths=None, debug=False):
        """
        Initialize SHAKE constraint.

        Args:
            pairs: List of tuples containing indices of atom pairs to constrain
            tolerance: Convergence criterion for constraint satisfaction
            bondlengths: Optional list of target distances for each pair.
                        If None, current distances will be used.
        """
        self.pairs = np.asarray(pairs)
        self.tolerance = tolerance
        self.bondlengths = bondlengths
        self.constraint_forces = None
        self.debug = debug

    def get_removed_dof(self, atoms):
        """Return number of removed degrees of freedom."""
        return len(self.pairs)

    def adjust_positions(self, atoms, new):
        """
        Adjust positions to satisfy distance constraints using SHAKE algorithm.

        Args:
            atoms: ASE Atoms object with current positions
            new: Array of new positions to be adjusted
        """
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)
        cell = atoms.cell.array
        pbc = atoms.pbc
        n_iter, new_adjusted = shake_utils.adjust_positions(
            old, new, cell, pbc, masses, self.pairs, self.bondlengths, self.maxiter, self.tolerance, False
        )
        new[:] = new_adjusted
        del new_adjusted
        if n_iter == self.maxiter:
            warnings.warn("SHAKE did not converge after {} iterations".format(self.maxiter))

    def adjust_momenta(self, atoms, momenta):
        """
        Adjust momenta to maintain constraints.

        Args:
            atoms: ASE Atoms object
            momenta: Momenta array to be modified
        """
        old = atoms.positions
        masses = atoms.get_masses()

        if self.bondlengths is None:
            self.bondlengths = self.initialize_bond_lengths(atoms)

        cell = atoms.cell.array
        pbc = atoms.pbc
        n_iter, momenta_adjusted = shake_utils._adjust_momenta(
            old, momenta, cell, pbc, masses, self.pairs, self.bondlengths, self.maxiter, self.tolerance, False
        )
        momenta[:] = momenta_adjusted
        del momenta_adjusted
        if n_iter == self.maxiter:
            warnings.warn("SHAKE did not converge after {} iterations".format(self.maxiter))

    def adjust_forces(self, atoms, forces):
        """
        Adjust forces to maintain constraints.

        Args:
            atoms: ASE Atoms object
            forces: Forces array to be modified
        """
        # Store the constraint forces
        self.constraint_forces = -forces.copy()
        # Use the same algorithm as for momenta
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces

    def initialize_bond_lengths(self, atoms):
        """
        Initialize bond lengths from current atomic positions if not provided.

        Args:
            atoms: ASE Atoms object

        Returns:
            Array of bond lengths
        """
        bondlengths = np.zeros(len(self.pairs))

        for i, (a, b) in enumerate(self.pairs):
            bondlengths[i] = atoms.get_distance(a, b, mic=True)

        return bondlengths

    def get_indices(self):
        """Return indices of all constrained atoms."""
        return np.unique(self.pairs.ravel())

    def index_shuffle(self, atoms, ind):
        """
        Shuffle the indices of atoms in this constraint.

        Args:
            atoms: ASE Atoms object
            ind: New indices
        """
        mapping = np.zeros(len(atoms), int)
        mapping[ind] = 1
        n = mapping.sum()
        mapping[:] = -1
        mapping[ind] = range(n)
        pairs = mapping[self.pairs]
        self.pairs = pairs[(pairs != -1).all(1)]
        if len(self.pairs) == 0:
            raise IndexError("Constraint not part of slice")

    def todict(self):
        """Convert constraint to dictionary for serialization."""
        return {
            "name": "SHAKEConstraint",
            "kwargs": {
                "pairs": self.pairs.tolist(),
                "tolerance": self.tolerance,
                "bondlengths": self.bondlengths.tolist() if self.bondlengths is not None else None,
            },
        }
