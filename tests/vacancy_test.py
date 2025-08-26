from __future__ import annotations

import numpy as np

from phonon_simulation.vacancy_calculations import (
    Lattice2DSystem,
    build_force_constants,
    calculate_normal_modes,
    create_eigenvectors_matrix_from_eigenvectors,
    get_eigenvectors,
)


def test_vacancy_vs_pristine_differences() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a 2D lattice system
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)
    pristine_frequencies = mesh_dict_pristine["frequencies"]
    vacancy_frequencies = mesh_dict_vac["frequencies"]
    pristine_eigenvectors = mesh_dict_pristine["eigenvectors"]
    vacancy_eigenvectors = mesh_dict_vac["eigenvectors"]

    assert not np.allclose(pristine_frequencies, vacancy_frequencies, atol=1e-12), (
        "Frequencies are expected to differ between the pristine and vacancy cases."
    )
    assert not np.allclose(pristine_eigenvectors, vacancy_eigenvectors, atol=1e-12), (
        "Eigenvectors are expected to differ between the pristine and vacancy cases."
    )


def test_vacancy_force_constants_zero() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )
    _, result_vac, _ = calculate_normal_modes(system, vacancy=True)
    positions = result_vac.get_positions()
    desired_vacancy_location = (
        np.array(system.lattice_vector_a) * system.n_repeatsa / 2
        + np.array(system.lattice_vector_b) * system.n_repeatsb / 2
    )
    distances_from_vacancy = np.linalg.norm(
        positions - desired_vacancy_location, axis=1
    )
    vacancy_atom_index = int(np.argmin(distances_from_vacancy))

    fc = build_force_constants(system, result_vac, vacancy=True)

    # Check all bonds to/from the vacancy atom directly
    for i in range(fc.shape[0]):
        for j in range(fc.shape[1]):
            if vacancy_atom_index in {i, j}:
                # All force constants for bonds to/from the vacancy atom should be very small
                assert np.all(
                    np.abs(fc[i, j]) < 1e-2 * max(system.k_nn, system.k_nnn) + 1e-6
                ), (
                    f"Force constants for bond ({i}, {j}) around vacancy (atom {vacancy_atom_index}) are not small."
                )


def test_vacancy_vs_pristine_shape() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)

    assert (
        mesh_dict_pristine["frequencies"].shape == mesh_dict_vac["frequencies"].shape
    ), "Frequencies shape should be the same for pristine and vacancy cases."
    assert (
        mesh_dict_pristine["eigenvectors"].shape == mesh_dict_vac["eigenvectors"].shape
    ), "Eigenvectors shape should be the same for pristine and vacancy cases."


def test_vacancy_vs_pristine_correct_shape() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)

    assert (
        mesh_dict_pristine["eigenvectors"].shape
        == mesh_dict_vac["eigenvectors"].shape
        == (
            64,  # This is the number of ir_qpoints and is currently set as a 10x10 mesh but this will have to be updated with the mesh setup and changed to the correct value or make it so that there is an input for mesh size
            3 * system.n_repeatsa * system.n_repeatsb,
            3 * system.n_repeatsa * system.n_repeatsb,
        )
    ), "Eigenvectors shape should be the correct shape."
    assert (
        mesh_dict_pristine["frequencies"].shape
        == mesh_dict_vac["frequencies"].shape
        == (
            64,  # This is the number of ir_qpoints and is currently set as a 10x10 mesh but this will have to be updated with the mesh setup and changed to the correct value or make it so that there is an input for mesh size
            3 * system.n_repeatsa * system.n_repeatsb,
        )
    ), "Frequencies shape should be the same for pristine and vacancy cases."


def test_force_constant_matrix_check() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a 2D lattice system
    _, result_pristine, _ = calculate_normal_modes(system, vacancy=False)
    _, result_vac, _ = calculate_normal_modes(system, vacancy=True)
    fc1 = build_force_constants(system, result_pristine, vacancy=False)
    fc2 = build_force_constants(system, result_vac, vacancy=True)

    assert not np.isnan(fc1).any() or np.isinf(fc1).any(), (
        "Force constants for pristine case should not be infinite or NaN."
    )
    assert not np.isnan(fc2).any() or np.isinf(fc2).any(), (
        "Force constants for vacancy case should not be infinite or NaN."
    )


def test_frequency_negative() -> None:
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=8,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a 2D lattice system
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)
    qpoints = mesh_dict_pristine["qpoints"]
    q_gamma_index = np.argmin(np.linalg.norm(qpoints - np.array([0, 0, 0]), axis=1))
    freqs_gamma_vac = mesh_dict_vac["frequencies"][q_gamma_index, :]
    freqs_gamma_pristine = mesh_dict_pristine["frequencies"][q_gamma_index, :]
    tolerance = 1e-6
    negative_vac = np.sum(freqs_gamma_vac < -tolerance)
    negative_pristine = np.sum(freqs_gamma_pristine < -tolerance)

    assert negative_vac == 0, (
        "Frequencies are expected to be positive for vacancy case."
    )
    assert negative_pristine == 0, (
        "Frequencies are expected to be positive for pristine case."
    )


def test_dot_product() -> None:
    """Test the dot product of eigenvectors for pristine and vacancy cases."""
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=1,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a 2D lattice system
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)
    desired_q_point = np.array([0, 0, 0])  # Should be true for all qpoints

    eigenvectors_dict = get_eigenvectors(
        system=system,
        mesh_dict_vac=mesh_dict_vac,
        mesh_dict_pristine=mesh_dict_pristine,
        desired_q_point=desired_q_point,
        band_index=3 * system.n_repeatsa * system.n_repeatsb - 1,
    )
    eigenvector_vac = eigenvectors_dict["eigenvectors_vac"]
    eigenvectors_pristine = eigenvectors_dict["eigenvectors_pristine"]
    normalised_vac_eigenvectors_column: complex = (
        eigenvector_vac / np.linalg.norm(eigenvector_vac)
    ).reshape(-1, 1)
    normalised_pristine_eigenvectors_column: complex = (
        eigenvectors_pristine / np.linalg.norm(eigenvectors_pristine)
    ).reshape(-1, 1)

    normalised_vac_eigenvectors_horizontal: complex = eigenvector_vac / np.linalg.norm(
        eigenvector_vac
    )
    normalised_pristine_eigenvectors_horizontal: complex = (
        eigenvectors_pristine / np.linalg.norm(eigenvectors_pristine)
    )
    pristine_dot = np.vdot(
        normalised_pristine_eigenvectors_horizontal,
        normalised_pristine_eigenvectors_column,
    )
    vacancy_dot = np.vdot(
        normalised_vac_eigenvectors_horizontal, normalised_vac_eigenvectors_column
    )

    assert np.isclose(np.real(pristine_dot), 1, atol=1e-12), (
        "Dot product expected to be real and 1 for pristine case."
    )
    assert np.isclose(np.real(vacancy_dot), 1, atol=1e-12), (
        "Dot product expected to be real and 1 for vacancy case."
    )
    assert np.isclose(np.imag(pristine_dot), 0, atol=1e-12), (
        "Dot product imaginary part expected to be 0 for pristine case."
    )
    assert np.isclose(np.imag(vacancy_dot), 0, atol=1e-12), (
        "Dot product imaginary part expected to be 0 for vacancy case."
    )


def test_reconstruction_error() -> None:
    """Test the error from the reconstruction of pristine eigenvectors from the matrix and coefficients calculated."""
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=1,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a lattice system
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)
    desired_q_point = np.array([0, 0, 0])  # Should be true for all qpoints
    band_index = 3 * system.n_repeatsa * system.n_repeatsb - 1

    info = create_eigenvectors_matrix_from_eigenvectors(
        system=system,
        mesh_dict_pristine=mesh_dict_pristine,
        mesh_dict_vac=mesh_dict_vac,
        desired_q_point=desired_q_point,
        band_index=band_index,
    )
    matrix = info["matrix"]
    coefficients = info["coefficients"]
    u_n = info["u_n"]
    u_reconstructed = np.matmul(matrix, coefficients)
    error = np.linalg.norm(u_n - u_reconstructed)
    tolerance = 1e-10

    assert error < tolerance, (
        "Error of reconstruction of pristine eigenvectors from matrix and coefficients is too large."
    )


def test_matrix_shape() -> None:
    """Test the shape of the matrix used for calculation of coefficients."""
    system = Lattice2DSystem(
        element="Si",
        lattice_vector_a=(1, 0, 0.0),
        lattice_vector_b=(0, 1, 0.0),
        n_repeatsa=8,
        n_repeatsb=1,
        k_nn=12.5,
        k_nnn=2.50,
    )  # This is an arbitrary example of a lattice system
    mesh_dict_pristine, _, _ = calculate_normal_modes(system, vacancy=False)
    mesh_dict_vac, _, _ = calculate_normal_modes(system, vacancy=True)
    desired_q_point = np.array([0, 0, 0])  # Should be true for all qpoints
    band_index = 3 * system.n_repeatsa * system.n_repeatsb - 1

    info = create_eigenvectors_matrix_from_eigenvectors(
        system=system,
        mesh_dict_pristine=mesh_dict_pristine,
        mesh_dict_vac=mesh_dict_vac,
        desired_q_point=desired_q_point,
        band_index=band_index,
    )
    matrix = info["matrix"]

    assert matrix.shape == (
        (system.n_repeatsa * system.n_repeatsb) ** 2,
        (system.n_repeatsa * system.n_repeatsb) ** 2,
    ), (
        "Matrix shape is not as expected, should be ((n_repeatsa * n_repeatsb) ** 2, (n_repeatsa * n_repeatsb) ** 2)."
    )
