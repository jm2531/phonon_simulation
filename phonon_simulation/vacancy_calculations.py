from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

        return (
            f"Normal modes for system: {self.system}\n"
            f"Frequencies (THz) shape:\n{self.frequencies.shape}\n"
            f"q-points shape:\n{self.qpoints.shape}\n"
            f"Eigenvectors shape:\n{self.eigenvectors.shape}\n"
        )


def find_lattice_bond_pairs(
    positions: np.ndarray,
    system: Lattice2DSystem,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Find nearest neighbor (NN) and next-nearest neighbor (NNN) bond pairs in a lattice system with periodic boundary conditions.

    Parameters
    ----------
    positions : np.ndarray
        Array of atomic positions in the supercell.
    system : Lattice2DSystem
        The 2D lattice system parameters.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        Lists of NN and NNN bond pairs as tuples of atom indices. This is then used to build the force constant matrix.
    """
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    nnn_diag1 = a_vec + b_vec
    nnn_diag2 = a_vec - b_vec
    rtol = 1e-2  # Relative tolerance for displacement comparison
    supercell_a = a_vec * system.n_repeatsa
    supercell_b = b_vec * system.n_repeatsb

    nn_pairs: list[tuple[int, int]] = []
    nnn_pairs: list[tuple[int, int]] = []

    # These include shifts across periodic boundaries
    pbc_shifts = [
        np.array([0, 0, 0]),
        supercell_a,
        -supercell_a,
        supercell_b,
        -supercell_b,
        supercell_a + supercell_b,
        supercell_a - supercell_b,
        -supercell_a + supercell_b,
        -supercell_a - supercell_b,
    ]

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i == j:
                continue
            disp_direct = positions[j] - positions[i]  # Calculate direct displacement

            for shift in pbc_shifts:  # Check with all possible periodic image shifts
                disp = disp_direct + shift

                # Check if this is a NN bond
                nn_condition = (
                    np.allclose(disp, a_vec, rtol=rtol)
                    or np.allclose(disp, -a_vec, rtol=rtol)
                    or np.allclose(disp, b_vec, rtol=rtol)
                    or np.allclose(disp, -b_vec, rtol=rtol)
                )
                if nn_condition and (i, j) not in nn_pairs and (j, i) not in nn_pairs:
                    nn_pairs.append((i, j))
                    break

                # Check if this is a NNN bond
                nnn_condition = (
                    np.allclose(disp, nnn_diag1, rtol=rtol)
                    or np.allclose(disp, -nnn_diag1, rtol=rtol)
                    or np.allclose(disp, nnn_diag2, rtol=rtol)
                    or np.allclose(disp, -nnn_diag2, rtol=rtol)
                )
                if (
                    nnn_condition
                    and (i, j) not in nnn_pairs
                    and (j, i) not in nnn_pairs
                ):
                    nnn_pairs.append((i, j))
                    break

    return nn_pairs, nnn_pairs


def calculate_normal_modes(
    system: Lattice2DSystem, *, vacancy: bool
) -> tuple[dict[str, np.ndarray], PhononSystem2DResult, NormalMode2DResult]:
    """
    Calculate phonon modes for a lattice with an optional vacancy.

    Instead of physically removing the vacancy atom, this function keeps all atoms
    but zeros out the force constants for bonds connected to the vacancy atom when
    vacancy=True.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system parameters
    vacancy : bool
        If True, apply vacancy effects

    Returns
    -------
    tuple
        mesh_dict, result, normal_modes
    """
    cell = PhonopyAtoms(
        symbols=[system.element],
        cell=[
            list(system.lattice_vector_a),
            list(system.lattice_vector_b),
            [0, 0, 1],
        ],
        scaled_positions=[[0, 0, 0]],
    )  # Create initial unit cell

    supercell_matrix = [
        [system.n_repeatsa, 0, 0],
        [0, system.n_repeatsb, 0],
        [0, 0, 1],
    ]  # Create n_repeatsa x n_repeatsb supercell
    phonon = Phonopy(unitcell=cell, supercell_matrix=supercell_matrix)

    large_cell = PhonopyAtoms(
        symbols=phonon.supercell.symbols,
        cell=phonon.supercell.cell,
        scaled_positions=phonon.supercell.scaled_positions,
    )  # Input large supercell as new unit cell

    phonon = Phonopy(
        unitcell=large_cell, supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )  # Create new phonon object with identity supercell matrix
    positions = phonon.supercell.positions

    result = PhononSystem2DResult(cell=large_cell, phonon=phonon, positions=positions)

    fc = build_force_constants(system, result, vacancy=vacancy)
    phonon.force_constants = fc
    #  Use print statements if code failing to run to see where code crashes
    mesh_dict: dict[str, np.ndarray] = {}
    mesh = np.array(
        [system.n_repeatsa, system.n_repeatsb, 1]
    )  # Change mesh density as required
    phonon.run_mesh(
        mesh=mesh, with_eigenvectors=True, is_mesh_symmetry=False, is_gamma_center=True
    )

    mesh_dict: dict[str, np.ndarray] = phonon.get_mesh_dict()

    normal_modes = NormalMode2DResult(
        system=system,
        frequencies=mesh_dict["frequencies"],
        eigenvectors=mesh_dict["eigenvectors"],
        qpoints=mesh_dict["qpoints"],
    )

    return mesh_dict, result, normal_modes


def build_force_constants(
    system: Lattice2DSystem, result: PhononSystem2DResult, *, vacancy: bool
) -> np.ndarray:
    """
    Build force constants of the lattice for a structure with an optional vacancy.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system parameters.
    result : PhononSystem2DResult
        Result object containing positions and structure information.
    vacancy : bool
        If True, apply vacancy effects.

    Returns
    -------
    np.ndarray
        Force constants matrix of shape (num_atoms, num_atoms, 3, 3).
    """
    positions = result.get_positions()
    num_atoms = len(positions)
    fc = np.zeros(
        (num_atoms, num_atoms, 3, 3), dtype=float
    )  # Initialize force constants matrix
    # Can induce any defect you want in this area of code
    vacancy_atom_index = find_vacancy_index(system, positions)
    nn_pairs, nnn_pairs = find_lattice_bond_pairs(positions, system)
    vacancy_bond_strength = 0.01  # Change this to adjust vacancy bond strength
    for i, j in nn_pairs:  # Assign force constants for nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                if vacancy and vacancy_atom_index in {i, j}:
                    fc[i, j, d1, d2] += (
                        -vacancy_bond_strength
                        * system.k_nn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[i, i, d1, d2] += (
                        vacancy_bond_strength
                        * system.k_nn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[j, i, d1, d2] += (
                        -vacancy_bond_strength
                        * system.k_nn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[j, j, d1, d2] += (
                        vacancy_bond_strength
                        * system.k_nn
                        * direction[d1]
                        * direction[d2]
                    )
                else:
                    fc[i, j, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                    fc[i, i, d1, d2] += system.k_nn * direction[d1] * direction[d2]
                    fc[j, i, d1, d2] += -system.k_nn * direction[d1] * direction[d2]
                    fc[j, j, d1, d2] += system.k_nn * direction[d1] * direction[d2]

    for i, j in nnn_pairs:  # Assign force constants for next-nearest neighbor bonds
        displacement_vector = positions[j] - positions[i]
        direction = displacement_vector / np.linalg.norm(displacement_vector)
        for d1 in range(3):
            for d2 in range(3):
                if vacancy and vacancy_atom_index in {i, j}:
                    fc[i, j, d1, d2] += (
                        -vacancy_bond_strength
                        * system.k_nnn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[i, i, d1, d2] += (
                        vacancy_bond_strength
                        * system.k_nnn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[j, i, d1, d2] += (
                        -vacancy_bond_strength
                        * system.k_nnn
                        * direction[d1]
                        * direction[d2]
                    )
                    fc[j, j, d1, d2] += (
                        vacancy_bond_strength
                        * system.k_nnn
                        * direction[d1]
                        * direction[d2]
                    )
                else:
                    fc[i, j, d1, d2] += -system.k_nnn * direction[d1] * direction[d2]
                    fc[i, i, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
                    fc[j, i, d1, d2] += -system.k_nnn * direction[d1] * direction[d2]
                    fc[j, j, d1, d2] += system.k_nnn * direction[d1] * direction[d2]
    return fc


def find_vacancy_index(
    system: Lattice2DSystem,
    positions: np.ndarray,
) -> int:
    """
    Find the index of the vacancy atom in a 2D lattice.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system for which the vacancy atom is to be found.
    positions : np.ndarray
        Array of atomic positions in the supercell.

    Returns
    -------
    int
        The index of the vacancy atom.
    """
    desired_vacancy_location = (
        np.array(system.lattice_vector_a) * system.n_repeatsa / 2
        + np.array(system.lattice_vector_b) * system.n_repeatsb / 2
    )  # Change as required, currently set up for the vaacancy to be the central atom
    distances = np.linalg.norm(
        positions - desired_vacancy_location, axis=1
    )  # Find atom closest to desired_vacancy_location
    return int(np.argmin(distances))


def get_eigenvectors(
    system: Lattice2DSystem,
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    desired_q_point: np.ndarray,
    band_index: int,
) -> dict[str, Any]:
    """
    Get eigenvector and frequency measurements at the desired q-point for both vacancy and pristine cases used for later calculations.

    Parameters
    ----------
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the pristine case.
    desired_q_point : np.ndarray
        The desired q-point at which to obtain eigenvector measurements.
    band_index : int
        The index of the band for which to obtain eigenvector measurements.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing eigenvectors and frequencies of the correct shape to be used in further calculations.
    """
    num_atoms = system.n_repeatsa * system.n_repeatsb
    qpoints = mesh_dict_vac["qpoints"]
    q_index = np.argmin(
        np.linalg.norm(qpoints - desired_q_point, axis=1)
    )  # Find the q-point index for desired q_point

    direction = (  # This works for 1D as there is only displacement in the x direction so this is required to specify the eigenvectors corresponding to y and z are not extracted
        0  # 0 corresponds to x-direction, can be changed to 1 for y-direction or 2 for z-direction
    )
    # This method will need to be changed for a 2D or 3D system
    indices = (
        np.arange(num_atoms, dtype=int) * 3 + direction
    )  # Skips the y and z components, as they will be zero for the current example of 1D chain along x

    # Calls eigenvectors for vacancy and pristine cases
    eigenvector_vac = mesh_dict_vac["eigenvectors"][q_index, :, band_index][indices]
    eigenvectors_pristine = mesh_dict_pristine["eigenvectors"][q_index, :, band_index][
        indices
    ]

    # Turn those eigenvectors into normalised column vectors
    normalised_vac_eigenvectors_column = (
        eigenvector_vac / np.linalg.norm(eigenvector_vac)
    ).reshape(-1, 1)
    normalised_pristine_eigenvectors_column = (
        eigenvectors_pristine / np.linalg.norm(eigenvectors_pristine)
    ).reshape(-1, 1)

    # Create num_atoms**2 long column vectors with normalised column eigenvector vectors repeated with a phase
    # difference in order to be a useful size for later calculations. This is done for both vacancy and pristine cases.
    phases = np.exp(
        1j
        * np.arange(num_atoms)
        * qpoints[q_index][0]
        * 2
        * np.pi
        * system.lattice_vector_a[0]
    )
    large_vac_eigenvectors = np.zeros(num_atoms**2, dtype=complex)
    for i in range(num_atoms):
        for j in range(num_atoms):
            large_vac_eigenvectors[i * num_atoms + j] = (
                normalised_vac_eigenvectors_column[j] * phases[i]
            )

    large_pristine_frequencies = np.zeros(
        num_atoms**2
    )  # Column vector created that will be filled with pristine frequencies for those specific bands and q_points.
    large_pristine_eigenvectors = np.zeros(num_atoms**2, dtype=complex)
    for i in range(num_atoms):
        for j in range(num_atoms):
            large_pristine_eigenvectors[i * num_atoms + j] = (
                normalised_pristine_eigenvectors_column[j] * phases[i]
            )
            large_pristine_frequencies[i * num_atoms + j] = mesh_dict_pristine[
                "frequencies"
            ][q_index, band_index]

    return {
        "eigenvectors_vac": eigenvector_vac,
        "eigenvectors_pristine": eigenvectors_pristine,
        "normalised_vac_eigenvectors": normalised_vac_eigenvectors_column,
        "normalised_pristine_eigenvectors": normalised_pristine_eigenvectors_column,
        "large_vac_eigenvectors": large_vac_eigenvectors,
        "large_pristine_eigenvectors": large_pristine_eigenvectors,
        "large_pristine_frequencies": large_pristine_frequencies,
    }


def create_eigenvectors_matrix_from_eigenvectors(
    system: Lattice2DSystem,
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    desired_q_point: np.ndarray,
    band_index: int,
) -> dict[str, Any]:
    """
    Create a matrix where each column is the large_vac_eigenvectors for a q-point and band. Then uses get_eigenvectors() to calculate each column vector. Then use this matrix to calculate coefficients.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system.
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data for vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data for pristine case.
    desired_q_point : np.ndarray
        The q-point at which to extract eigenvectors.
    band_index : int
        The band index to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing vacancy eigenvector matrix, pristine eigenvectors and coefficients.
    """
    num_atoms = system.n_repeatsa * system.n_repeatsb

    qpoints = mesh_dict_vac["qpoints"]
    num_qpoints = len(qpoints)

    start_band = 2 * num_atoms
    num_bands_to_use = num_atoms

    matrix = np.zeros(
        (num_atoms**2, num_qpoints * num_bands_to_use), dtype=complex
    )  # Initialize the matrix to correct shape

    for q_idx, q_point in enumerate(qpoints):  # Loop through all q-points and bands
        for band_offset in range(num_bands_to_use):
            band_idx = start_band + band_offset
            column_idx = q_idx * num_bands_to_use + band_offset

            temp_dict_vac = mesh_dict_vac.copy()
            temp_dict_pristine = mesh_dict_pristine.copy()

            result = get_eigenvectors(  # Call get_eigenvectors with the desired q-point and band
                system=system,
                mesh_dict_vac=temp_dict_vac,
                mesh_dict_pristine=temp_dict_pristine,
                desired_q_point=q_point,
                band_index=band_idx,
            )

            matrix[:, column_idx] = result[
                "large_vac_eigenvectors"
            ]  # Creates matrix where each column is the large eigenvector for a specific q-point and band

    eigenvectors = get_eigenvectors(
        system=system,
        mesh_dict_vac=mesh_dict_vac,
        mesh_dict_pristine=mesh_dict_pristine,
        desired_q_point=desired_q_point,
        band_index=band_index,
    )

    u_n = eigenvectors[
        "large_pristine_eigenvectors"
    ]  # Gets eigenvectors for specific q-point and band for the pristine case.
    # I have used notation of coefficients = matrix multiplied by u_n
    matrix_inverse = np.linalg.inv(matrix)  # Inverts matrix
    coefficients = np.matmul(matrix_inverse, u_n)  # Calculates coefficients

    return {
        "matrix": matrix,
        "u_n": u_n,
        "coefficients": coefficients,
    }
