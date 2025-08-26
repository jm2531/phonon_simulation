from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phonon_simulation.vacancy_calculations import (
    DispersionPath,
    Lattice2DSystem,
    calculate_normal_modes,
)
from phonon_simulation.vacancy_plotting import (
    animate_eigenvectors_time,
    plot_dispersion_along_path,
    plot_eigenvector_decomposition_coefficients,
    plot_eigenvectors,
    plot_frequencies,
)

# TO DO LIST: Clean up code and move some plots over from plotting.py and calculation.py into vacancy_plotting.py and vacancy_plotting.py.
# The code for vacancies only uses code form vacancy_plotting.py and vacancy_calculations.py.
start_time = time.time()

dispersion_path = DispersionPath(
    points=np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ]
    ),
    labels=[r"$\Gamma$", "X"],
)

system = Lattice2DSystem(
    element="Si",  # Change to any symbol as required
    lattice_vector_a=(1, 0, 0.0),
    lattice_vector_b=(0, 1, 0.0),
    n_repeatsa=20,  # preferably use even number so that diagonals of reciprocal in mesh lattice lie along the exact diagonals, due to Monkhorst Pack grid.
    n_repeatsb=1,
    k_nn=12.5,  # Nearest neighbour force constant
    k_nnn=0,  # Next nearest neighbour force constant
)

desired_q_point: np.ndarray = np.array([0.25, 0, 0])
band_index = 3 * system.n_repeatsa * system.n_repeatsb - 1


mesh_dict_pristine, result_pristine, normal_modes_pristine = calculate_normal_modes(
    system=system, vacancy=False
)
folder = Path("./examples")
modes_output2 = folder / "2d_grid.txt"
modes_output2.write_text(normal_modes_pristine.to_human_readable(), encoding="utf-8")


mesh_dict_vac, result_vac, normal_modes_vac = calculate_normal_modes(
    system=system, vacancy=True
)
folder = Path("./examples")
modes_output = folder / "2d_grid.vacancy.txt"
modes_output.write_text(normal_modes_vac.to_human_readable(), encoding="utf-8")

# These plots that are commented out are from plotting.py and calculations.py which are do not have working vacancy support. I will update them and move them over to vacancy_plotting.
"""
path = dispersion_path
result = result_pristine
mesh_dict = mesh_dict_pristine

fig, ax = plot_dispersion_path(path, system)
fig.show()
plot_2d_lattice(result, system, vacancy=False)

# Calculate band structure (smooth curve)
q_path_band = interpolate_path(path.points, n_points=100)
result.get_phonon().run_band_structure([q_path_band], with_eigenvectors=True)
bands = result.get_phonon().get_band_structure_dict()

plot_2d_dispersion_band_and_mesh(
    bands, mesh_dict, path.points, path.labels, system
)  # Plots both band and mesh in one figure
plot_2d_dispersion_band(
    bands, path.points, path.labels, system
)  # Plots only the band structure
plot_2d_dispersion_mesh(
    bands, mesh_dict, path.points, path.labels, system
)  # Plots only the mesh structure


plot_2d_mesh_3d_scatter(mesh_dict, system)  # Plots 3D scatter plot of mesh points.
plot_2d_mesh_3d_surface(
    mesh_dict, system
)  # Plots 3D surface plot of mesh points, doesn't work for 1D but great for 2D as it visualises the phonon modes.
"""

fig, ax = plot_dispersion_along_path(
    system=system, path=dispersion_path, mesh_dict=mesh_dict_pristine
)  # Input either mesh_dict_vac for vacancy dispersion relation or mesh_dict_pristine for pristine surface dispersion relation along customizable path


plot_frequencies(
    mesh_dict_vac=mesh_dict_vac,
    mesh_dict_pristine=mesh_dict_pristine,
    desired_q_point=desired_q_point,
)

plot_eigenvectors(
    system=system,
    mesh_dict_vac=mesh_dict_vac,
    mesh_dict_pristine=mesh_dict_pristine,
    eigenvector_component_type="real",  # Choose "real", "imaginary", or "absolute" for measure
    desired_q_point=desired_q_point,
    band_index=band_index,
)


ani = animate_eigenvectors_time(
    system,
    mesh_dict_vac,
    mesh_dict_pristine,
    eigenvector_component_type="real",  # Choose "real", "imaginary", or "absolute" for measure
    desired_q_point=desired_q_point,
    band_index=band_index,
)


plot_eigenvector_decomposition_coefficients(
    system=system,
    mesh_dict_vac=mesh_dict_vac,
    mesh_dict_pristine=mesh_dict_pristine,
    desired_q_point=desired_q_point,
    band_index=band_index,
)  # Only plots frequencies of with the same q_point because of blochs theorem

print(
    "Process finished ---%s seconds ---" % (time.time() - start_time)
)  # just to check, has  no real use

plt.show()
