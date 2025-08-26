from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from phonon_simulation.calculating import (
    find_central_atom_index,
    find_lattice_bond_pairs,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from phonon_simulation.calculating import (
        DispersionPath,
        Lattice2DSystem,
        PhononSystem2DResult,
    )


def _plot_dispersion_path_lines(
    path: DispersionPath,
    system: Lattice2DSystem,
    axis: tuple[int, int] = (0, 1),
    *,
    ax: Axes,
) -> None:
    cart_points = reduced_to_cartesian(path.points, system)
    (line,) = ax.plot(
        cart_points[:, axis[0]],
        cart_points[:, axis[1]],
        "k-",
        lw=2,
    )
    line.set_marker("o")
    line.set_markerfacecolor("r")
    line.set_markeredgecolor("r")
    """
    arrowprops = {
        "arrowstyle": "->",
        "color": "k",
        "lw": 1.5,
        "shrinkA": 0,
        "shrinkB": 0,
    }
    # Add arrows between points
    midpoints = (path.points[:-1] + path.points[1:]) / 2
    startpoints = (path.points[:-1] + midpoints) / 2
    for start, end in zip(startpoints, midpoints, strict=False):
        ax.annotate(
            "",
            xy=end[list(axis)],
            xytext=start[list(axis)],
            arrowprops=arrowprops,
        )"""


def _plot_dispersion_path_labels(
    path: DispersionPath,
    system: Lattice2DSystem,
    axis: tuple[int, int] = (0, 1),
    *,
    ax: Axes,
) -> None:
    cart_points = reduced_to_cartesian(path.points, system)
    x_center = 0.0
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    qx_lim = 0.5 * np.linalg.norm(b1)
    offset = 0.05 * qx_lim

    for label, pt in zip(path.labels, cart_points, strict=True):
        x, y = pt[list(axis)]
        if x > x_center:
            ax.text(x + offset, y, label, fontsize=10, va="center", ha="left")
        elif x < x_center:
            ax.text(x - offset, y, label, fontsize=10, va="center", ha="right")
        elif x == x_center:
            if y > 0:
                ax.text(x, y + offset, label, fontsize=10, va="bottom", ha="center")
            else:
                ax.text(x, y - offset, label, fontsize=10, va="top", ha="center")
        else:
            ax.text(x, y, label, fontsize=10, va="center", ha="center")


def plot_dispersion_path(
    path: DispersionPath, system: Lattice2DSystem, axis: tuple[int, int] = (0, 1)
) -> tuple[Figure, Axes]:
    """Plot the first Brillouin zone (BZ) of a 2D lattice and the path used for phonon dispersion calculations. The plot size is set to 2pi/a by 2pi/b where a and b are the magnitudes of the lattice vectors."""
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    b2 = 2 * np.pi * np.array([-a_vec[1], a_vec[0]]) / area
    qx_lim = 0.5 * np.linalg.norm(b1)
    qy_lim = 0.5 * np.linalg.norm(b2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        [-qx_lim, qx_lim, qx_lim, -qx_lim, -qx_lim],
        [-qy_lim, -qy_lim, qy_lim, qy_lim, -qy_lim],
        "k-",
        lw=1,
    )
    _plot_dispersion_path_lines(path, system, axis=axis, ax=ax)
    _plot_dispersion_path_labels(path, system, axis=axis, ax=ax)
    ax.set_xticks(np.array([-qx_lim, 0, qx_lim]))
    ax.set_xticklabels([r"$-\pi/a$", "0", r"$\pi/a$"])
    ax.set_yticks(np.array([-qy_lim, 0, qy_lim]))
    ax.set_yticklabels([r"$-\pi/b$", "0", r"$\pi/b$"])
    ax.set_xlim(float(-1.1 * qx_lim), float(1.1 * qx_lim))
    ax.set_ylim(float(-1.1 * qy_lim), float(1.1 * qy_lim))
    ax.set_title("FBZ & Path", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


def plot_2d_lattice(
    result: PhononSystem2DResult,
    system: Lattice2DSystem,
    *,
    vacancy: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot the 2D lattice structure inputted with nearest and next nearest neighbour bonds shown.

    Parameters
    ----------
    result : PhononSystem2DResult
        The result object containing atomic positions.
    system : Lattice2DSystem
        The 2D lattice system being plotted.
    vacancy : bool
        If True, exclude the central atom from the plot and bonds.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    positions = result.get_positions()

    # Find central atom index if vacancy is True
    central_atom_index = 0  # Setting a default value as central atom can never be zero
    if vacancy:
        central_atom_index: int = find_central_atom_index(
            system,
            positions,
            np.array(system.lattice_vector_a),
            np.array(system.lattice_vector_b),
        )

    # Exclude the central atom from plotting if vacancy is True
    if vacancy and central_atom_index != 0:
        mask = np.arange(len(positions)) != central_atom_index
        plot_positions = positions[mask]
    else:
        plot_positions = positions

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(plot_positions[:, 0], plot_positions[:, 1], s=50, c="black")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(
        f"{system.n_repeatsa}x{system.n_repeatsb} supercell of {system.element} atoms with nearest \n and next nearest neighbour interactions"
    )
    ax.set_aspect("equal")

    nn_pairs, nnn_pairs = find_lattice_bond_pairs(
        positions, system, vacancy=vacancy, central_atom_index=central_atom_index
    )
    label_added = [False, False]

    for i, j in nn_pairs:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="orange",
            linewidth=1.5,
            alpha=0.7,
            label="Nearest neighbour" if not label_added[0] else None,
        )
        label_added[0] = True
    for i, j in nnn_pairs:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="blue",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Next nearest neighbour" if not label_added[1] else None,
        )
        label_added[1] = True
    margin = 0.2 * max(np.ptp(plot_positions[:, 0]), np.ptp(plot_positions[:, 1]))
    x_min, x_max = (
        plot_positions[:, 0].min() - margin,
        plot_positions[:, 0].max() + margin,
    )
    y_min, y_max = (
        plot_positions[:, 1].min() - margin,
        plot_positions[:, 1].max() + margin,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1), ncol=2)
        plt.subplots_adjust(bottom=0.3)
    return fig, ax


def reduced_to_cartesian(path_points, system: Lattice2DSystem) -> np.ndarray:
    """Convert reduced coordinates to cartesian reciprocal coordinates for plotting."""  # This function may not be needed but works for now as the plot of the path is low priority to have working shape and labels for non square or rectangular lattices.
    path_points = np.array(path_points)
    a_vec = np.array(system.lattice_vector_a[:2])
    b_vec = np.array(system.lattice_vector_b[:2])
    area = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    b1 = 2 * np.pi * np.array([b_vec[1], -b_vec[0]]) / area
    b2 = 2 * np.pi * np.array([-a_vec[1], a_vec[0]]) / area
    return np.array([p[0] * b1 + p[1] * b2 for p in path_points])


def plot_2d_dispersion_band_and_mesh(  # noqa: PLR0914
    bands: dict[str, np.ndarray],
    mesh_dict: dict[str, np.ndarray],
    path: np.ndarray,
    labels: list[str],
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot the phonon dispersion relation for a 2D lattice system along a specified path in the First Brillouin zone, and overlay the mesh frequencies as scatter points.

    Parameters
    ----------
    bands : dict[str, np.ndarray]
        Dictionary containing band structure data (distances and frequencies).
    mesh_dict : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies).
    path : np.ndarray
        Array of q-points defining the path in reciprocal space.
    labels : list[str]
        List of labels for the high-symmetry points along the path.
    system : Lattice2DSystem
        The 2D lattice system being analysed.
    n_mesh_points : int
        Number of mesh points to interpolate along the path.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects for the plot.
    """
    # Band structure (smooth)
    distances = bands["distances"][0]
    band_frequencies = bands["frequencies"][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i in range(band_frequencies.shape[1]):
        color = cmap(i % cmap.N)
        ax.plot(distances, band_frequencies[:, i], color=color, linewidth=3)

    # Mesh points (sparse)
    mesh_qpoints = mesh_dict["qpoints"]
    mesh_freqs = mesh_dict["frequencies"]
    q_path_mesh = interpolate_path(
        path, n_points=int(np.floor(max(system.n_repeatsa, system.n_repeatsb) / 2))
    )
    # For each mesh q-point, find the closest band q-point and use its distance
    for idx, q in enumerate(q_path_mesh):
        # Find the closest q in the band path
        band_qs = interpolate_path(path, n_points=100)
        dists = np.linalg.norm(band_qs - q, axis=1)
        band_idx = np.argmin(dists)
        tol = 5e-4
        matches = np.where(np.all(np.abs(mesh_qpoints - q) < tol, axis=1))[0]
        for m in matches:
            for band in range(mesh_freqs.shape[1]):
                ax.scatter(
                    distances[band_idx],
                    mesh_freqs[m, band],
                    color="red",
                    s=8,
                    zorder=3,
                    label="Mesh frequencies"
                    if idx == 0 and m == matches[0] and band == 0
                    else None,
                )

    tick_locations = [distances[0]]
    tick_locations.extend(
        [
            distances[i * len(distances) // (len(path) - 1)]
            for i in range(1, len(path) - 1)
        ]
    )
    tick_locations.append(distances[-1])
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of  Lattice using up to next nearest neighbours: \n {system.n_repeatsa}x{system.n_repeatsb} supercell made of {system.element} atoms"
    )
    ax.grid(visible=True)
    _handles, legend_labels = ax.get_legend_handles_labels()
    if "Mesh frequencies" in legend_labels:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_2d_dispersion_band(
    bands: dict[str, np.ndarray],
    path: np.ndarray,
    labels: list[str],
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot the phonon dispersion relation for a 2D lattice system along a specified path in the First Brillouin zone.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object containing phonon calculation results.
    system : Lattice2DSystem
        The 2D lattice system being analysed.
    path : np.ndarray
        Array of q-points defining the path in reciprocal space.
    labels : list[str]
        List of labels for the high-symmetry points along the path.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects for the plot.
    """
    # Band structure (smooth)
    distances = bands["distances"][0]
    band_frequencies = bands["frequencies"][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i in range(band_frequencies.shape[1]):
        color = cmap(i % cmap.N)
        ax.plot(distances, band_frequencies[:, i], color=color, linewidth=3)

    tick_locations = [distances[0]]
    segment_length = len(distances) // (len(path) - 1)
    tick_locations.extend(
        [distances[i * segment_length] for i in range(1, len(path) - 1)]
    )
    tick_locations.append(distances[-1])
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of  Lattice using up to next nearest neighbours: \n {system.n_repeatsa}x{system.n_repeatsb} supercell made of {system.element} atoms"
    )
    ax.grid(visible=True)
    _handles, legend_labels = ax.get_legend_handles_labels()
    if "Mesh frequencies" in legend_labels:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def plot_2d_dispersion_mesh(
    bands: dict[str, np.ndarray],
    mesh_dict: dict[str, np.ndarray],
    path: np.ndarray,
    labels: list[str],
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot the mesh frequencies as scatter points for a 2D lattice system along a specified path in the First Brillouin zone.

    Parameters
    ----------
    phonon : Phonopy
        Phonopy object containing phonon calculation results.
    system : Lattice2DSystem
        The 2D lattice system being analysed.
    path : np.ndarray
        Array of q-points defining the path in reciprocal space.
    labels : list[str]
        List of labels for the high-symmetry points along the path.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects for the plot.
    """
    distances = bands["distances"][0]

    fig, ax = plt.subplots(figsize=(8, 6))

    mesh_qpoints = mesh_dict["qpoints"]
    mesh_freqs = mesh_dict["frequencies"]
    q_path_mesh = interpolate_path(
        path, n_points=int(np.floor(max(system.n_repeatsa, system.n_repeatsb) / 2))
    )
    # Overlay mesh frequencies only for mesh q-points that are exactly on the path (within a tolerance)
    for idx, q in enumerate(q_path_mesh):
        # Find the closest q in the band path
        band_qs = interpolate_path(path, n_points=100)
        dists = np.linalg.norm(band_qs - q, axis=1)
        band_idx = np.argmin(dists)
        # Find mesh q-point index
        tol = 5e-4
        matches = np.where(np.all(np.abs(mesh_qpoints - q) < tol, axis=1))[0]
        for m in matches:
            for band in range(mesh_freqs.shape[1]):
                ax.scatter(
                    distances[band_idx],
                    mesh_freqs[m, band],
                    color="red",
                    s=8,
                    zorder=3,
                    label="Mesh frequencies"
                    if idx == 0 and m == matches[0] and band == 0
                    else None,
                )

    tick_locations = [distances[0]]
    segment_length = len(distances) // (len(path) - 1)
    tick_locations.extend(
        [distances[i * segment_length] for i in range(1, len(path) - 1)]
    )
    tick_locations.append(distances[-1])
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(labels)
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_title(
        f"Phonon Dispersion of  Lattice using up to next nearest neighbours: \n {system.n_repeatsa}x{system.n_repeatsb} supercell made of {system.element} atoms"
    )
    ax.grid(visible=True)
    _handles, legend_labels = ax.get_legend_handles_labels()
    if "Mesh frequencies" in legend_labels:
        ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax


def interpolate_path(path: np.ndarray, n_points: int) -> np.ndarray:
    """
    Interpolate a path in reciprocal space by generating evenly spaced points between each pair of consecutive points.

    Parameters
    ----------
    path : np.ndarray
        Array of q-points defining the path in reciprocal space.
    n_points : int
        Number of points to interpolate between each pair of consecutive points.

    Returns
    -------
    np.ndarray
        Interpolated array of q-points along the path.
    """
    points = []
    for i in range(len(path) - 1):
        seg = np.linspace(path[i], path[i + 1], n_points, endpoint=False)
        points.append(seg)
    points.append(path[-1][None, :])
    return np.vstack(points)


def plot_2d_mesh_3d_scatter(
    mesh_dict: dict[str, np.ndarray],
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot the mesh frequencies as a 3D scatter plot for a 2D lattice system along a specified path in the First Brillouin zone.

    Parameters
    ----------
    mesh_dict : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies).
    system : Lattice2DSystem
        The 2D lattice system being analysed.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects for the plot.
    """
    qpoints = mesh_dict["qpoints"]
    freqs = mesh_dict["frequencies"]

    qx = qpoints[:, 0]
    qy = qpoints[:, 1]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.get_cmap("tab10").colors
    scatters = []
    handles = []
    for band in range(freqs.shape[1]):
        z = freqs[:, band]
        color = colors[band % len(colors)]
        scatter = ax.scatter(
            qx,
            qy,
            z,
            color=color,
            alpha=0.7,
            label=f"Band {band + 1}",
            picker=True,
        )
        scatters.append(scatter)
        handles.append(
            Patch(facecolor=color, edgecolor="k", label=f"Band {band + 1}", alpha=0.7)
        )

    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_zlabel("Frequency (THz)")
    ax.set_title(
        f"Phonon mesh scatter: {system.n_repeatsa}x{system.n_repeatsb} supercell of {system.element}"
    )

    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Bands",
        fancybox=True,
    )
    fig.tight_layout()

    # Interactivity: toggle band visibility on legend click
    def legend_pick(event) -> None:
        for i, legpatch in enumerate(leg.legend_handles):
            if event.artist == legpatch:
                scatter = scatters[i]
                visible = not scatter.get_visible()
                scatter.set_visible(visible)
                legpatch.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw_idle()
                break

    for legpatch in leg.legend_handles:
        legpatch.set_picker(True)
    fig.canvas.mpl_connect("pick_event", legend_pick)

    return fig, ax


def plot_2d_mesh_3d_surface(
    mesh_dict: dict[str, np.ndarray],
    system: Lattice2DSystem,
) -> tuple[
    Figure, Axes
]:  # Currently a copilot generated function just to check that mesh gives expected results, will rewrite properly when vacancies are working
    """
    Plot the mesh frequencies surfaces as a 3D surface plot for a 2D lattice system along a specified path in the First Brillouin zone.

    Parameters
    ----------
    mesh_dict : dict[str, np.ndarray]
        Dictionary containing mesh data (q-points and frequencies).
    system : Lattice2DSystem
        The 2D lattice system being analysed.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects for the plot.
    """
    qpoints = mesh_dict["qpoints"]
    freqs = mesh_dict["frequencies"]

    nqx = system.n_repeatsa
    nqy = system.n_repeatsb

    qx = qpoints[:, 0].reshape((nqx, nqy))
    qy = qpoints[:, 1].reshape((nqx, nqy))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.get_cmap("tab10").colors
    surfaces = []
    handles = []

    for band in range(freqs.shape[1]):
        z = freqs[:, band].reshape((nqx, nqy))
        color = colors[band % len(colors)]
        surf = ax.plot_surface(
            qx,
            qy,
            z,
            color=color,
            alpha=0.18,  # More transparent
            linewidth=0,
            antialiased=True,
            shade=True,
            label=f"Band {band + 1}",
            picker=True,
        )
        surfaces.append(surf)
        handles.append(
            Patch(facecolor=color, edgecolor="k", label=f"Band {band + 1}", alpha=0.5)
        )

    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_zlabel("Frequency (THz)")
    ax.set_title(
        f"Phonon mesh surface: {system.n_repeatsa}x{system.n_repeatsb} supercell of {system.element}"
    )

    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Bands",
        fancybox=True,
    )
    fig.tight_layout()

    # Interactivity: toggle band visibility on legend click
    def on_pick(event) -> None:
        legend = event.artist
        idx = handles.index(legend)
        surf = surfaces[idx]
        visible = not surf._visible
        surf.set_visible(visible)
        legend.set_alpha(1.0 if visible else 0.2)
        fig.canvas.draw_idle()

    # Connect legend patches to pick events
    for legpatch, surf in zip(leg.legend_handles, surfaces, strict=False):
        legpatch.set_picker(True)

    def legend_pick(event) -> None:
        for i, legpatch in enumerate(leg.legend_handles):
            if event.artist == legpatch:
                surf = surfaces[i]
                visible = not surf._visible
                surf.set_visible(visible)
                legpatch.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect("pick_event", legend_pick)

    return fig, ax


def plot_mesh_frequency_difference(
    mesh_dict_no_vac: dict[str, np.ndarray],
    mesh_dict_vac: dict[str, np.ndarray],
    system: Lattice2DSystem,
    diff_type: str = "absolute",  # or "signed"
) -> tuple[
    Figure, Axes
]:  # Currently a copilot generated function just to check that mesh gives expected results, will rewrite properly when vacancies are working
    """Plot the difference in mesh frequencies between no-vacancy and vacancy cases as a 3D surface, with interactive legend and colorbar."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    qpoints = mesh_dict_no_vac["qpoints"]
    freqs_no_vac = mesh_dict_no_vac["frequencies"]
    freqs_vac = mesh_dict_vac["frequencies"]

    nqx = system.n_repeatsa
    nqy = system.n_repeatsb

    qx = qpoints[:, 0].reshape((nqx, nqy))
    qy = qpoints[:, 1].reshape((nqx, nqy))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_diffs = []
    surfaces = []
    handles = []
    cmap = plt.get_cmap("viridis")

    # Compute global min/max for normalization across all bands
    for band in range(freqs_no_vac.shape[1]):
        if diff_type == "absolute":
            diff = np.abs(freqs_no_vac[:, band] - freqs_vac[:, band])
        else:
            diff = freqs_no_vac[:, band] - freqs_vac[:, band]
        all_diffs.append(diff)
    all_diffs_flat = np.concatenate(all_diffs)
    vmin = np.nanmin(all_diffs_flat)
    vmax = np.nanmax(all_diffs_flat)

    for band, diff in enumerate(all_diffs):
        z = diff.reshape((nqx, nqy))
        normed = (z - vmin) / (vmax - vmin + 1e-12)
        facecolors = cmap(normed)
        surf = ax.plot_surface(
            qx,
            qy,
            z,
            facecolors=facecolors,
            alpha=0.7,
            linewidth=0,
            antialiased=True,
            shade=False,
            label=f"Band {band + 1}",
            picker=True,
        )
        surfaces.append(surf)
        handles.append(
            Patch(
                facecolor=cmap((band + 0.5) / freqs_no_vac.shape[1]),
                edgecolor="k",
                label=f"Band {band + 1}",
                alpha=0.7,
            )
        )

    ax.set_xlabel(r"$q_x$")
    ax.set_ylabel(r"$q_y$")
    ax.set_zlabel("Frequency Difference (THz)")
    ax.set_title(
        f"Difference in mesh frequencies (vacancy vs no vacancy): {system.n_repeatsa}x{system.n_repeatsb} supercell"
    )

    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Bands",
        fancybox=True,
    )
    fig.tight_layout()

    # Interactivity: toggle band visibility on legend click
    def legend_pick(event) -> None:
        for i, legpatch in enumerate(leg.legend_handles):
            if event.artist == legpatch:
                surf = surfaces[i]
                visible = not surf._visible
                surf.set_visible(visible)
                legpatch.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw_idle()
                break

    for legpatch in leg.legend_handles:
        legpatch.set_picker(True)
    fig.canvas.mpl_connect("pick_event", legend_pick)

    # Add colorbar for the difference magnitude (all bands)
    from matplotlib.cm import ScalarMappable

    mappable_for_cb = ScalarMappable(cmap=cmap)
    mappable_for_cb.set_array(all_diffs_flat)
    cbar = fig.colorbar(mappable_for_cb, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Frequency Difference (THz)")

    return fig, ax


def plot_vacancy_lattice(
    result: PhononSystem2DResult,
    system: Lattice2DSystem,
) -> tuple[Figure, Axes]:
    """
    Plot a 2D lattice structure with a vacancy, highlighting the vacancy site and showing the distortion of bonds around it.

    Parameters
    ----------
    result : PhononSystem2DResult
        The result object from calculate_2d_modes_with_vacancy.
    system : Lattice2DSystem
        The 2D lattice system being plotted.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib Figure and Axes objects for the plot.
    """
    positions = result.get_positions()

    # Calculate where the vacancy would be (center of supercell)
    a_vec = np.array(system.lattice_vector_a)
    b_vec = np.array(system.lattice_vector_b)
    cellsizea = system.n_repeatsa * a_vec
    cellsizeb = system.n_repeatsb * b_vec
    (cellsizea + cellsizeb) / 2.0

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot atoms
    ax.scatter(positions[:, 0], positions[:, 1], s=60, c="black", label="Atoms")

    # Find bonds
    nn_pairs, nnn_pairs = find_lattice_bond_pairs(positions, system, vacancy=True)

    # Get force constants for all bonds
    fc = build_force_constants_2d(system, result, vacancy=True)

    # Plot nearest neighbor bonds (orange, fixed alpha)
    for i, j in nn_pairs:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            color="orange",
            linewidth=1.5,
            alpha=0.8,
            solid_capstyle="round",
            label="NN bond" if (i, j) == nn_pairs[0] else None,
        )

    # Calculate relative size of NNN force constants for transparency
    nnn_strengths = []
    for i, j in nnn_pairs:
        # Use Frobenius norm of the force constant matrix for this bond
        fc_val = np.linalg.norm(fc[i, j])
        nnn_strengths.append(fc_val)
    nnn_strengths = np.array(nnn_strengths)
    # Avoid division by zero
    if np.max(nnn_strengths) > 0:
        nnn_alphas = nnn_strengths / np.max(nnn_strengths)
    else:
        nnn_alphas = np.ones_like(nnn_strengths)

    # Plot next-nearest neighbor bonds with alpha proportional to force constant
    for idx, (i, j) in enumerate(nnn_pairs):
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
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
