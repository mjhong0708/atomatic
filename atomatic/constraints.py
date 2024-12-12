import numpy as np
from ase.constraints import FixConstraint
from numba import njit


@njit(fastmath=True)
def find_mic(dr: np.ndarray, cell: np.ndarray, pbc: np.ndarray):
    # Check where distance larger than 1/2 cell. Particles have crossed
    # periodic boundaries then and need to be unwrapped.
    rec = np.linalg.inv(cell)

    pbc = pbc.astype(np.int64)
    rec *= pbc.reshape(3, 1)
    dri = np.round(np.dot(dr, rec))
    # Unwrap
    return dr - np.dot(dri, cell)


@njit(fastmath=True)
def _adjust_positions(pairs, bondlengths, masses, old, new, maxiter, tolerance, cell, pbc):
    for _ in range(maxiter):
        converged = True
        for i, (a, b) in enumerate(pairs):
            # Get target bond length
            target_length = bondlengths[i]

            # Calculate vectors considering MIC
            r0 = old[a] - old[b]
            d0 = find_mic(r0, cell, pbc)

            # New displacement including MIC correction
            d1 = new[a] - new[b] - r0 + d0

            # Calculate mass factor
            m = 1 / (1 / masses[a] + 1 / masses[b])

            # Calculate correction factor (lambda)
            x = 0.5 * (target_length**2 - np.dot(d1, d1)) / np.dot(d0, d1)

            if abs(x) > tolerance:
                # Apply position corrections
                new[a] += x * m / masses[a] * d0
                new[b] -= x * m / masses[b] * d0
                converged = False

        if converged:
            break
    else:
        raise RuntimeError("SHAKE did not converge within maximum iterations")


@njit(fastmath=True)
def _adjust_momenta(pairs, bondlengths, masses, old, momenta, maxiter, tolerance, cell, pbc):
    for _ in range(maxiter):
        converged = True
        for i, (a, b) in enumerate(pairs):
            # Get current bond vector with MIC
            d = old[a] - old[b]
            d = find_mic(d, cell, pbc)

            # Calculate relative velocity
            dv = momenta[a] / masses[a] - momenta[b] / masses[b]

            # Calculate mass factor
            m = 1 / (1 / masses[a] + 1 / masses[b])

            # Calculate correction factor
            x = -np.dot(dv, d) / bondlengths[i] ** 2

            if abs(x) > tolerance:
                # Apply momentum corrections
                momenta[a] += x * m * d
                momenta[b] -= x * m * d
                converged = False

        if converged:
            break
    else:
        raise RuntimeError("Momentum adjustment did not converge")


class SHAKE(FixConstraint):
    """
    Implementation of SHAKE algorithm as a constraint in ASE.
    Maintains fixed distances between specified pairs of atoms.
    """

    maxiter = 500  # Maximum number of iterations for SHAKE

    def __init__(self, pairs, tolerance=1e-13, bondlengths=None, debug=False):
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
        _adjust_positions(self.pairs, self.bondlengths, masses, old, new, self.maxiter, self.tolerance, cell, pbc)

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
        _adjust_momenta(self.pairs, self.bondlengths, masses, old, momenta, self.maxiter, self.tolerance, cell, pbc)

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
        map = np.zeros(len(atoms), int)
        map[ind] = 1
        n = map.sum()
        map[:] = -1
        map[ind] = range(n)
        pairs = map[self.pairs]
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
