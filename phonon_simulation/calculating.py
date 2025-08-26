from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from phonopy.api_phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms


@dataclass(kw_only=True, frozen=True)
class Lattice2DSystem:
    """
    Represents a 2D  lattice system for phonon calculations.

    Attributes
    ----------
    element : str
        The chemical symbol of the element (e.g., "C" for carbon).
    lattice_vector_a : tuple[float, float, float]
        Lattice vector along the a direction, in Angstroms, (1 Å = 1e-10 m).
    lattice_vector_b : tuple[float, float, float]
        Lattice vector along the b direction, in Angstroms, (1 Å = 1e-10 m).
    n_repeatsa : int
        Number of unit cells repeated along the x direction.
    n_repeatsb : int
        Number of unit cells repeated along the y direction.
    k_nn : float
        Nearest neighbor spring constant, in electronvolts per Angstrom d (1 eV/Å² = 16.02 N/m).
    k_nnn : float
        Next nearest neighbor spring constant, in electronvolts per Angstrom d (1 eV/Å² = 16.02 N/m).
    """

    element: str
    lattice_vector_a: tuple[float, float, float]
    lattice_vector_b: tuple[float, float, float]
    n_repeatsa: int
    n_repeatsb: int
    k_nn: float
    k_nnn: float

    @property
    def mass(self) -> float:
        """Return the mass of the element in atomic mass units(1 AMU = 1.6605402e-27 kg)."""
        cell = PhonopyAtoms(
            symbols=[self.element],
            cell=[
                list(self.lattice_vector_a),
                list(self.lattice_vector_b),
                [0, 0, 1],  # For 2D, z is just a dummy direction
            ],
            scaled_positions=[[0, 0, 0]],
        )
        return cell.masses[0]


@dataclass(frozen=True)
class PhononSystem2DResult:
    """
    Stores the results of a 2D phonon system calculation.

    Attributes
    ----------
    cell : PhonopyAtoms
        The atomic structure of the system.
    phonon : Phonopy
        The Phonopy object containing phonon calculation results.
    positions : np.ndarray
        The atomic positions in the supercell.
    """

    cell: PhonopyAtoms
    phonon: Phonopy
    positions: np.ndarray

    def get_cell(self) -> PhonopyAtoms:
        """
        Return the atomic structure (PhonopyAtoms) of the system.

        Returns
        -------
        PhonopyAtoms
            The atomic structure of the system.
        """
        return self.cell

    def get_phonon(self) -> Phonopy:
        """
        Return the Phonopy object containing phonon calculation results.

        Returns
        -------
        Phonopy
            The Phonopy object containing phonon calculation results.
        """
        return self.phonon

    def get_positions(self) -> np.ndarray:
        """
        Return the atomic positions in the supercell.

        Returns
        -------
        np.ndarray
            The atomic positions in the supercell.
        """
        return self.positions


@dataclass(frozen=True, kw_only=True)
class DispersionPath:
    """
    Represents a path in reciprocal space for 2D phonon calculations.

    Attributes
    ----------
    path_points : np.ndarray
        Array of k-point coordinates along the path (shape: (N, 3)).
    labels : list[str]
        List of labels for each k-point in the path.

    Methods
    -------
    __post_init__() -> None
        Validates that the number of labels matches the number of path points.
    """

    points: np.ndarray[tuple[int, int], np.dtype[np.floating]]
    labels: list[str]

    def __post_init__(self) -> None:
        assert self.points.shape == (len(self.points), 3), (
            f"Path points must be a 2D array with shape (N, 3), got {self.points.shape}"
        )


def build_force_constants_2d(
    system: Lattice2DSystem,
    result: PhononSystem2DResult,
    *,
    vacancy: bool = False,
) -> np.ndarray:
    """
    Build the force constant matrix for a 2D lattice system including nearest and next nearest neighbor interactions using the atomic positions from a PhononSystem2DResult object and bond pairs found in find_lattice_bond_pairs.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which to build the force constant matrix.
    result : PhononSystem2DResult
        The result object containing cell, phonon, and positions.
    vacancy : bool
        If True, exclude the central atom from the force constants.

    Returns
    -------
    np.ndarray
        The force constant matrix of shape (num_atoms, num_atoms, 3, 3).
    """
    positions = result.get_positions()
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    num_atoms = positions.shape[0]

    # Find central atom index if vacancy is True
    central_atom_index = None
    if vacancy:
        central_atom_index = find_central_atom_index(system, positions, a_vec, b_vec)

    nn_pairs, nnn_pairs = find_lattice_bond_pairs(
        positions, system, vacancy=vacancy, central_atom_index=central_atom_index
    )

    fc = np.zeros((num_atoms, num_atoms, 3, 3), dtype=float)

    for i, j in nn_pairs:  # Nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]

    for i, j in nnn_pairs:  # Next-nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                fc[i, j, d1, d2] += -system.k_nnn * direction[d1] * direction[d2]
                fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    print(f"vacancy={vacancy}: {len(nn_pairs)} NN bonds, {len(nnn_pairs)} NNN bonds")
    print(f"vacancy={vacancy}: force constant sum = {np.sum(np.abs(fc))}")
    return fc


def calculate_2d_modes(
    system: Lattice2DSystem,
    *,
    vacancy: bool = False,
) -> tuple[dict[str, np.ndarray], PhononSystem2DResult, NormalMode2DResult]:
    """
    Calculate the phonon modes for a 2D lattice system.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which to calculate phonon modes.
    vacancy : bool, optional
        If True, exclude the central atom from the calculation.

    Returns
    -------
    tuple
        A tuple containing:
            - mesh_dict: dict[str, np.ndarray]
                Dictionary with mesh information including frequencies, eigenvectors, and q-points.
            - result: PhononSystem2DResult
                The result object containing cell, phonon, and positions.
            - normal_modes: NormalMode2DResult
                The normal mode results including frequencies, eigenvectors, and q-points.
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            list(system.lattice_vector_a),
            list(system.lattice_vector_b),
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )
    supercell_matrix = [[system.n_repeatsa, 0, 0], [0, system.n_repeatsb, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    positions = phonon.supercell.get_positions()

    temp_result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)
    fc = build_force_constants_2d(system, temp_result, vacancy=vacancy)
    phonon.force_constants = fc

    mesh = [system.n_repeatsa, system.n_repeatsb, 1]
    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    positions = phonon.supercell.get_positions()
    result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)

    frequencies = mesh_dict["frequencies"]  # (Nq, Natoms*3)
    eigenvectors = mesh_dict["eigenvectors"]  # (Nq, Natoms*3, Natoms*3)
    qpoints = mesh_dict["qpoints"]  # (Nq, 3)
    normal_modes = NormalMode2DResult(
        system=system,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        qpoints=qpoints,
    )
    return mesh_dict, result, normal_modes


def calculate_2d_modes2(
    system: Lattice2DSystem,
    *,
    vacancy: bool = False,
) -> tuple[dict[str, np.ndarray], PhononSystem2DResult, NormalMode2DResult]:
    """
    Calculate the phonon modes for a 2D lattice system using a large unit cell approach.

    This function creates a large unit cell of size n_repeatsa * n_repeatsb atoms,
    and uses an identity supercell matrix. The mesh size is adjusted to produce
    a similar q-point density as calculate_2d_modes.
    """
    # Build positions for the large unit cell
    positions = []
    for i in range(system.n_repeatsa):
        for j in range(system.n_repeatsb):
            pos = (
                np.array(system.lattice_vector_a) * i
                + np.array(system.lattice_vector_b) * j
            )
            positions.append(pos)
    positions = np.array(positions)

    # Create the large unit cell
    symbols = [system.element] * len(positions)
    cell = PhonopyAtoms(
        symbols=symbols,
        cell=[
            (np.array(system.lattice_vector_a) * system.n_repeatsa).tolist(),
            (np.array(system.lattice_vector_b) * system.n_repeatsb).tolist(),
            [0, 0, 1],
        ],
        positions=positions,
    )

    # Use identity supercell matrix (no repetition)
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)
    positions = phonon.supercell.get_positions()

    temp_result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)
    fc = build_force_constants_2d(system, temp_result, vacancy=vacancy)
    phonon.force_constants = fc

    # Use a much smaller mesh size to get equivalent density
    # For a large unit cell, mesh [1,1,1] gives roughly the same q-point density
    # as mesh [n_repeatsa, n_repeatsb, 1] in the primitive cell approach
    mesh = [8, 8, 1]

    phonon.run_mesh(
        mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )
    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()
    positions = phonon.supercell.get_positions()
    result = PhononSystem2DResult(cell=cell, phonon=phonon, positions=positions)

    frequencies = mesh_dict["frequencies"]
    eigenvectors = mesh_dict["eigenvectors"]
    qpoints = mesh_dict["qpoints"]
    normal_modes = NormalMode2DResult(
        system=system,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        qpoints=qpoints,
    )
    return mesh_dict, result, normal_modes


@dataclass(kw_only=True, frozen=True)
class NormalMode2DResult:
    """
    Stores the normal mode results for a 2D phonon system.

    Attributes
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which the normal modes are calculated.
    frequencies : np.ndarray
        Array of phonon frequencies (shape: (Nq, Natoms*3)).
    eigenvectors : np.ndarray
        Array of phonon eigenvectors (shape: (Nq, Natoms*3, Natoms*3)).
    qpoints : np.ndarray
        Array of q-points in reciprocal space (shape: (Nq, 3)).

    Methods
    -------
    to_human_readable() -> str
        Returns a human-readable string representation of the normal modes, including frequencies, q-points, and eigenvectors.
    """

    system: Lattice2DSystem
    frequencies: np.ndarray
    eigenvectors: np.ndarray
    qpoints: np.ndarray

    def to_human_readable(self) -> str:
        """
        Return a human-readable string representation of the normal modes, including frequencies, q-points, and eigenvectors.

        Returns
        -------
        str
            A formatted string containing the normal mode information for the system.
        """
        np.set_printoptions(
            threshold=10000000000
        )  # Large to ensure all are printed out with no truncation
        print(self.eigenvectors.shape)
        return (
            f"Normal modes for system: {self.system}\n"
            f"Frequencies (THz) shape:\n{self.frequencies.shape}\n"
            f"q-points shape:\n{self.qpoints.shape}\n"
            f"Eigenvectors shape:\n{self.eigenvectors.shape}\n"
        )


def find_central_atom_index(
    system: Lattice2DSystem,
    positions: np.ndarray,
) -> int:
    """
    Find the index of the central atom in a 2D lattice.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which the central atom is to be found.
    positions : np.ndarray
        Array of atomic positions in the supercell.

    Returns
    -------
    int
        The index of the central atom.
    """
    # Calculate center of supercell
    center = (
        np.array(system.lattice_vector_a) * system.n_repeatsa / 2
        + np.array(system.lattice_vector_b) * system.n_repeatsb / 2
    )
    # Find atom closest to center
    distances = np.linalg.norm(positions - center, axis=1)
    central_atom_index = int(np.argmin(distances))

    print(
        f"Central atom identified at index {central_atom_index}, position {positions[central_atom_index]}"
    )
    return central_atom_index


def find_lattice_bond_pairs(
    positions: np.ndarray,
    system: Lattice2DSystem,
    *,
    vacancy: bool = False,
    central_atom_index: int | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Fast, vectorized search for nearest neighbor (nn) and next-nearest neighbor (nnn) atom pairs in a 2D lattice."""
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    nnn_diag1 = a_vec + b_vec
    nnn_diag2 = a_vec - b_vec
    rtol = 1e-3

    num_atoms = positions.shape[0]
    # Mask for vacancy
    mask = np.ones(num_atoms, dtype=bool)
    if vacancy and central_atom_index is not None:
        mask[central_atom_index] = False

    # Compute all displacements (i, j, 3)
    disp = positions[None, :, :] - positions[:, None, :]

    # NN mask
    nn_mask = np.all(np.isclose(disp, a_vec, rtol=rtol), axis=-1) | np.all(
        np.isclose(disp, b_vec, rtol=rtol), axis=-1
    )
    # NNN mask
    nnn_mask = np.all(np.isclose(disp, nnn_diag1, rtol=rtol), axis=-1) | np.all(
        np.isclose(disp, nnn_diag2, rtol=rtol), axis=-1
    )

    # Apply vacancy mask to both axes
    mask2d = mask[:, None] & mask[None, :]

    nn_indices = np.argwhere(nn_mask & mask2d)
    nnn_indices = np.argwhere(nnn_mask & mask2d)

    nn_pairs = [tuple(idx) for idx in nn_indices]
    nnn_pairs = [tuple(idx) for idx in nnn_indices]
    return nn_pairs, nnn_pairs
