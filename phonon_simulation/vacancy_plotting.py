from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# Find bonds (excluding vacancy atom)
from phonon_simulation.vacancy_calculations import (
    build_force_constants,
    create_eigenvectors_matrix_from_eigenvectors,
    find_lattice_bond_pairs,
    find_vacancy_index,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from phonon_simulation.vacancy_calculations import DispersionPath, Lattice2DSystem


def plot_frequencies(
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    desired_q_point: np.ndarray,
) -> tuple[Figure, Axes]:
    """
    Plot the frequencies at desired_q_point for both vacancy and pristine cases.

    Parameters
    ----------
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the pristine case.
    desired_q_point : np.ndarray
        The q-point at which to plot the frequencies.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    qpoints = mesh_dict_pristine["qpoints"]
    frequencies_vac = mesh_dict_vac["frequencies"]
    frequencies_pristine = mesh_dict_pristine["frequencies"]
    q_index = np.argmin(np.linalg.norm(qpoints - desired_q_point, axis=1))
    freqs_at_gamma_vac = frequencies_vac[
        q_index, :
    ]  # All bands at desired_q_point for vacancy case
    freqs_at_gamma_pristine = frequencies_pristine[
        q_index, :
    ]  # All bands at desired_q_point for pristine case
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Frequencies for q={qpoints[q_index]} for all bands")
    ax.plot(freqs_at_gamma_vac, marker="o", label="Vacancy")
    ax.plot(freqs_at_gamma_pristine, marker="o", label="Pristine")
    ax.set_xlabel("Band index")
    ax.set_ylabel("Frequency (THz)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_eigenvectors(  # noqa: PLR0913, PLR0917
    system: Lattice2DSystem,
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    eigenvector_component_type: str,
    desired_q_point: np.ndarray,
    band_index: int,
) -> tuple[Figure, Axes]:
    """
    Plot the components of the eigenvectors at the desired q-point for both vacancy and pristine cases.

    Parameters
    ----------
    system: Lattice2DSystem
        The 2D lattice system being analysed.
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the pristine case.
    eigenvector_component_type : str
        The type of eigenvector component to plot, only choose real, imaginary and absolute.
    desired_q_point : np.ndarray
        The q-point at which to plot the eigenvectors.
    band_index : int
        The index of the band to plot.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    num_atoms = system.n_repeatsa * system.n_repeatsb
    eigenvector_info = {
        "eigenvector_component_type": eigenvector_component_type,
        "band_index": band_index,
        "num_atoms": num_atoms,
        "desired qpoint": desired_q_point,
    }
    eigenvectors = get_eigenvector_components(
        mesh_dict_vac, mesh_dict_pristine, eigenvector_info
    )

    qpoints = mesh_dict_pristine["qpoints"]
    q_index = np.argmin(np.linalg.norm(qpoints - desired_q_point, axis=1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Eigenvector for q={qpoints[q_index]}), band={band_index}")
    ax.plot(np.real(eigenvectors["eigenvectors_vac"]), label="Vacancy")
    ax.plot(np.real(eigenvectors["eigenvectors_pristine"]), label="Pristine")
    ax.set_xlabel("Atom coordinate index")
    ax.set_ylabel(f"Eigenvector component ({eigenvector_component_type})")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def get_eigenvector_components(
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    eigenvector_info: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Get eigenvector measurements at the desired q-point for both vacancy and pristine cases.

    Parameters
    ----------
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies) for the pristine case.
    eigenvector_info : dict[str, Any]
        Dictionary containing measurement information (e.g., band index, desired q-point).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing frequency and eigenvector measurements at the Γ-point for both cases.

    Raises
    ------
    ValueError
        If the eigenvector_component_type argument is not 'real', 'imaginary', or 'absolute'.
    """
    band_index = eigenvector_info["band_index"]
    desired_q_point = eigenvector_info["desired qpoint"]
    eigenvector_component_type = eigenvector_info["eigenvector_component_type"]

    direction = (
        0  # x-direction, can be changed to 1 for y-direction or 2 for z-direction
    )

    qpoints = mesh_dict_vac[
        "qpoints"
    ]  # Doesn't matter if you use vac pr pristine as q_points will be same for both

    q_index = np.argmin(
        np.linalg.norm(qpoints - desired_q_point, axis=1)
    )  # Find the q-point index for the desired q-point

    indices = np.arange(eigenvector_info["num_atoms"], dtype=int) * 3 + direction

    if eigenvector_component_type == "real":
        eigenvector_vac = np.real(
            mesh_dict_vac["eigenvectors"][q_index, :, band_index][indices]
        )
        eigenvectors_pristine = np.real(
            mesh_dict_pristine["eigenvectors"][q_index, :, band_index][indices]
        )
    elif eigenvector_component_type == "imaginary":
        eigenvector_vac = np.imag(
            mesh_dict_vac["eigenvectors"][q_index, :, band_index][indices]
        )
        eigenvectors_pristine = np.imag(
            mesh_dict_pristine["eigenvectors"][q_index, :, band_index][indices]
        )
    elif eigenvector_component_type == "absolute":
        eigenvector_vac = np.abs(
            mesh_dict_vac["eigenvectors"][q_index, :, band_index][indices]
        )
        eigenvectors_pristine = np.abs(
            mesh_dict_pristine["eigenvectors"][q_index, :, band_index][indices]
        )
    else:
        error_message = f"Invalid eigenvector_component_type: {eigenvector_component_type}. Must be 'real', 'imaginary', or 'absolute'."
        raise ValueError(error_message)
    return {
        "eigenvectors_vac": eigenvector_vac,
        "eigenvectors_pristine": eigenvectors_pristine,
    }


def plot_dispersion_along_path(
    system: Lattice2DSystem,
    path: DispersionPath,
    mesh_dict: dict[str, np.ndarray],
) -> tuple[Figure, Axes]:
    """Plot the phonon dispersion relation along specified path in reciprocal space in reduced coordinates.

    Parameters
    ----------
    system: Lattice2DSystem
        The 2D lattice system being analysed.
    path: DispersionPath
        The path along which to plot the dispersion, in reduced reciprocal coordinates.
    mesh_dict: dict[str, np.ndarray]
        The mesh dictionary containing q-points and frequencies. Choose either pristine or vacancy case.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    qpoints = mesh_dict["qpoints"]
    frequencies = mesh_dict["frequencies"]

    bands_to_plot = list(
        range(3 * system.n_repeatsa * system.n_repeatsb)
    )  # Number of bands to plot, should be 3N in order to plot all bands

    path_indices = []
    path_distances: np.ndarray = np.array([])
    total_distance = 0.0

    # For each path segment
    for i in range(len(path.points) - 1):
        start_point = path.points[i]
        end_point = path.points[i + 1]
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        unit_vector = segment_vector / segment_length

        # Find all q-points near this segment
        for q_idx, q in enumerate(qpoints):
            # Project q-point onto segment line
            q_relative = q - start_point
            projection = np.dot(q_relative, unit_vector)

            # If projection is within segment length and q-point is close to segment
            if 0 <= projection <= segment_length:
                # Calculate perpendicular distance from q-point to line
                perp_distance = np.linalg.norm(q_relative - projection * unit_vector)

                # If q-point is close enough to path segment
                if perp_distance < 0.005:  # Tolerance value
                    path_indices.append(q_idx)
                    path_distances = np.append(
                        path_distances, total_distance + projection
                    )

        total_distance += segment_length

    # Sort by distance along path
    sorted_indices: np.ndarray = np.argsort(path_distances)
    sorted_path_indices: list[int] = [path_indices[i] for i in sorted_indices]
    sorted_path_distances = [path_distances[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 8))

    rng = np.random.default_rng(
        42
    )  # Each band set to random colour so they can be distinguished, could be changed to a set colour scheme
    colors = rng.uniform(0, 1, size=(len(bands_to_plot), 3))

    for idx, band_idx in enumerate(bands_to_plot):
        band_freqs = frequencies[sorted_path_indices, band_idx]
        color = colors[idx]
        ax.scatter(
            sorted_path_distances,
            band_freqs,
            s=20,
            color=color,
        )  # Plot dots at each frequency point
        ax.plot(
            sorted_path_distances,
            band_freqs,
            linewidth=1,
            color=color,
            alpha=0.7,
        )  # Plot line connecting frequencies within same band with same colour

    # Used to set x-ticks and labels at high symmetry points
    path_distances: np.ndarray = np.array([0])
    for i in range(1, len(path.points)):
        path_distances = np.append(
            path_distances,
            path_distances[-1] + np.linalg.norm(path.points[i] - path.points[i - 1]),
        )

    ax.set_xticks(path_distances)
    ax.set_xticklabels(path.labels)

    for distance in path_distances:
        ax.axvline(
            x=distance, color="k", linestyle="--", alpha=0.3
        )  # Add vertical lines at high symmetry points

    ax.set_xlabel("Wave Vector")
    ax.set_ylabel("Frequency (THz)")
    ax.set_title("Phonon Dispersion Relation")

    fig.tight_layout()
    return fig, ax


def animate_eigenvectors_time(
    system: Lattice2DSystem,
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    eigenvector_component_type: str,
    desired_q_point: np.ndarray,
    band_index: int | None = None,
) -> animation.FuncAnimation:
    """
    Animate the time evolution of eigenvector displacements for a given band at a desired q-point, using cos(omega * t) for both vacancy and pristine cases. Shows the eigenvectors oscillating at different frequencies.

    Parameters
    ----------
    system: Lattice2DSystem
        The 2D lattice system being analysed.
    mesh_dict_vac : dict[str, np.ndarray]
        Contains the mesh information for the vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Contains the mesh information for the pristine case.
    eigenvector_component_type : str
        Component of complex eigenvectors: either "real", "imaginary", or "absolute"
    desired_q_point : np.ndarray
        The desired q-point for the animation.
    band_index : int, optional
        The band index to animate. If None, uses highest band.

    Returns
    -------
    animation.FuncAnimation
        The matplotlib animation object.
    """
    duration = 40  # Duration set at 20 seconds
    num_atoms = system.n_repeatsa * system.n_repeatsb
    if band_index is None:
        band_index = 3 * num_atoms - 1

    qpoints = mesh_dict_vac["qpoints"]
    q_index = np.argmin(np.linalg.norm(qpoints - desired_q_point, axis=1))

    direction = 0  # x-direction at the moment as using 1D
    indices = np.arange(num_atoms) * 3 + direction

    eig_vac = mesh_dict_vac["eigenvectors"][q_index, :, band_index][indices]
    eig_pristine = mesh_dict_pristine["eigenvectors"][q_index, :, band_index][indices]
    mesh_dict_vac["frequencies"][q_index, band_index]
    mesh_dict_pristine["frequencies"][q_index, band_index]

    nframes = int(duration * 100)
    t_vals = np.linspace(0, duration, nframes)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(f"Eigenvector Animation (q={qpoints[q_index]}, band={band_index})")
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Displacement")
    (line_vac,) = ax.plot([], [], "r-", label="Vacancy")
    (line_pristine,) = ax.plot([], [], "b-", label="Pristine")
    ax.legend()
    ax.set_xlim(0, num_atoms - 1)

    max_amp = max(np.max(np.abs(eig_vac)), np.max(np.abs(eig_pristine)))
    ax.set_ylim(-1.2 * max_amp, 1.2 * max_amp)

    def get_disp(eig: np.ndarray, omega: float, t: float) -> np.ndarray:
        """
        Find the displacement at time t.

        Parameters
        ----------
        eig : np.ndarray
            The eigenvector displacements.
        omega : float
            The angular frequency.
        t : float
            The time.

        Returns
        -------
        np.ndarray
            The displacement at time t.

        Raises
        ------
        ValueError
            Invalid eigenvector_component_type.
        """
        if eigenvector_component_type == "real":
            return np.real(eig * np.cos(omega * t))
        if eigenvector_component_type == "imaginary":
            return np.imag(eig * np.cos(omega * t))
        if eigenvector_component_type == "absolute":
            return np.abs(eig * np.cos(omega * t))
        msg = "Invalid eigenvector_component_type"
        raise ValueError(msg)

    def animate(frame: int) -> tuple[Line2D, Line2D]:
        """
        Find the displacement at time t.

        Parameters
        ----------
        frame : int
            The frame index.

        Returns
        -------
        line_vac
            The line object for the vacancy.
        line_pristine
            The line object for the pristine.
        """
        t = (
            5 * t_vals[frame]
        )  # This makes animation 10 times slower at the moment so difference in frequencies can be observed

        omega_vac = (
            2 * np.pi * mesh_dict_vac["frequencies"][q_index, band_index]
        )  # Convert to omega (radians) as frequency was in THz beforehand
        omega_pristine = (
            2 * np.pi * mesh_dict_pristine["frequencies"][q_index, band_index]
        )

        y_vac = get_disp(eig_vac, omega_vac, t)
        y_pristine = get_disp(eig_pristine, omega_pristine, t)
        line_vac.set_data(np.arange(num_atoms), y_vac)
        line_pristine.set_data(np.arange(num_atoms), y_pristine)
        return line_vac, line_pristine

    return animation.FuncAnimation(
        fig, animate, frames=nframes, interval=1000 / 20, blit=True
    )


def plot_eigenvector_decomposition_coefficients(
    system: Lattice2DSystem,
    mesh_dict_vac: dict[str, np.ndarray],
    mesh_dict_pristine: dict[str, np.ndarray],
    desired_q_point: np.ndarray,
    band_index: int,
) -> tuple[Figure, Axes]:
    """
    Plot the coefficients of the eigenvector decomposition against frequencies.

    Only shows coefficients for vacancy eigenvectors with the same q-point
    as the pristine lattice eigenvector.

    Parameters
    ----------
    system: Lattice2DSystem
        The 2D lattice system being analysed.
    mesh_dict_vac : dict[str, np.ndarray]
        Dictionary containing mesh data for vacancy case.
    mesh_dict_pristine : dict[str, np.ndarray]
        Dictionary containing mesh data for pristine case.
    desired_q_point : np.ndarray
        The q-point at which to plot the eigenvectors.
    band_index : int
        The band index for which to plot the eigenvectors.


    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    decomposition = create_eigenvectors_matrix_from_eigenvectors(
        system=system,
        mesh_dict_vac=mesh_dict_vac,
        mesh_dict_pristine=mesh_dict_pristine,
        desired_q_point=desired_q_point,
        band_index=band_index,
    )

    coefficients = decomposition["coefficients"]  # Get coefficients

    # Find the q-point index that matches desired_q_point
    qpoints = mesh_dict_vac["qpoints"]
    q_index = np.argmin(np.linalg.norm(qpoints - desired_q_point, axis=1))

    num_atoms = system.n_repeatsa * system.n_repeatsb

    # Calculate the range of indices in the coefficient array for just the desired q-point as only those will contribute due to Blochs Theorem
    start_idx = q_index * num_atoms
    end_idx = start_idx + num_atoms

    # Filter coefficients to only include those for the desired q-point
    filtered_coefficients = coefficients[start_idx:end_idx]
    coeff_magnitudes = np.abs(filtered_coefficients)

    # Get frequencies for the vacancy modes at the desired q-point
    # Use 2 * num_atoms + band_offset for 1D and num_atoms + band_offset for 2D as two thirds of the bands are zero for 1D and one third for 2D
    frequencies: np.ndarray = np.array([])
    for band_offset in range(num_atoms):
        band_idx = 2 * num_atoms + band_offset
        frequencies = np.append(
            frequencies, mesh_dict_vac["frequencies"][q_index, band_idx]
        )

    # Get the pristine frequency for the specific q-point and band
    pristine_frequency = mesh_dict_pristine["frequencies"][q_index, band_index]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the coefficients
    ax.scatter(frequencies, coeff_magnitudes**2, s=30, alpha=0.7)

    # Add labels and title
    ax.set_xlabel("Frequency (THz)", fontsize=12)
    ax.set_ylabel("|Coefficient|", fontsize=12)
    ax.set_title(
        f"Eigenvector Decomposition Coefficients (Same q-point only)\n"
        f"q-point: {qpoints[q_index]}, band: {band_index}",
        fontsize=14,
    )
    ax.grid(visible=True, alpha=0.3)

    # Add a horizontal line at zero
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    # Add vertical line for pristine frequency
    ax.axvline(
        x=pristine_frequency,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Pristine frequency: {pristine_frequency:.4f} THz",
    )

    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_vacancy_lattice(  # Plot vacancy lattice is currently just ai geneerated in order to check the structure is as expected.
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot a 2D lattice structure with a vacancy, highlighting the vacancy site and showing the distortion of bonds around it.

    Parameters
    ----------
    system : Lattice2DSystem
        The 2D lattice system being plotted.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    # Generate all atom positions in the supercell
    positions = []
    for i in range(system.n_repeatsa):
        for j in range(system.n_repeatsb):
            pos = (
                np.array(system.lattice_vector_a) * i
                + np.array(system.lattice_vector_b) * j
            )
            positions.append(pos)
    positions = np.array(positions, dtype=float)

    vacancy_index = find_vacancy_index(system, positions)
    vacancy_pos = positions[vacancy_index]

    # Remove the vacancy atom from the positions for plotting
    mask = np.arange(len(positions)) != vacancy_index
    plot_positions = positions[mask]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot atoms (excluding vacancy)
    ax.scatter(
        plot_positions[:, 0], plot_positions[:, 1], s=60, c="black", label="Atoms"
    )

    # Highlight the vacancy position
    ax.scatter(
        vacancy_pos[0], vacancy_pos[1], s=120, c="red", marker="x", label="Vacancy"
    )
    nn_pairs, nnn_pairs = find_lattice_bond_pairs(plot_positions, system)

    # Get force constants for all bonds (with vacancy)
    # Build a dummy result object just for force constants
    class DummyResult:
        def get_positions(self):
            return plot_positions

    dummy_result = DummyResult()
    fc = build_force_constants(system, dummy_result, vacancy=True)

    # Plot nearest neighbor bonds (orange, fixed alpha)
    for i, j in nn_pairs:
        ax.plot(
            [plot_positions[i, 0], plot_positions[j, 0]],
            [plot_positions[i, 1], plot_positions[j, 1]],
            color="orange",
            linewidth=1.5,
            alpha=0.8,
            solid_capstyle="round",
            label="NN bond" if (i, j) == nn_pairs[0] else None,
        )

    # Calculate relative size of NNN force constants for transparency
    nnn_strengths = []
    for i, j in nnn_pairs:
        fc_val = np.linalg.norm(fc[i, j])
        nnn_strengths.append(fc_val)
        if np.max(nnn_strengths) > 0:
            nnn_alphas = nnn_strengths / np.max(nnn_strengths)
        else:
            nnn_alphas = np.ones_like(nnn_strengths)

    # Plot next-nearest neighbor bonds with alpha proportional to force constant
    for idx, (i, j) in enumerate(nnn_pairs):
        ax.plot(
            [plot_positions[i, 0], plot_positions[j, 0]],
            [plot_positions[i, 1], plot_positions[j, 1]],
            color="blue",
            linewidth=1.5,
            alpha=nnn_alphas[idx],
            linestyle="--",
            solid_capstyle="round",
            label="NNN bond" if idx == 0 else None,
        )

    # Set plot limits
    margin = 1.0
    x_min, x_max = (
        positions[:, 0].min() - margin,
        positions[:, 0].max() + margin,
    )
    y_min, y_max = (
        positions[:, 1].min() - margin,
        positions[:, 1].max() + margin,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add title and labels
    ax.set_title(
        f"{system.n_repeatsa}x{system.n_repeatsb} {system.element} supercell with vacancy",
        fontsize=14,
    )
    ax.set_xlabel("x (Å)", fontsize=12)
    ax.set_ylabel("y (Å)", fontsize=12)
    ax.set_aspect("equal")

    # Create a legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.8,
    )

    fig.tight_layout()
    return fig, ax
