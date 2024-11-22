# -*- coding: utf-8 -*-
"""
This code is designed to extract and visualize data from the European Synchrotron Radiation Facility (ESRF)
at beamline BM02.

The code is structured as follows:
    - Functions for extracting and visualizing data from the raw .h5data files
    - Functions for extracting and visualizing data from the processed .h5 files
    - Functions for saving data to .xy files

@author: williamrigaut
"""

import h5py, pathlib, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _convert_from_q_to_theta(q_list, wavelength=0.495937):
    """
    Convert a list of scattering vector q to a list of 2theta angles.

    Parameters
    ----------
    q_list : list of floats
        List of q values to convert
    wavelength : float, optional
        The wavelength of the X-ray beam. By default, 0.495937 Angstroms (25 keV, ESRF BM02 setup)

    Returns
    -------
    two_theta_list : list of floats
        List of 2theta angles corresponding to the input q values
    """
    to_theta = lambda q: np.arcsin(q * wavelength / (4 * np.pi))  # in rad
    to_degree = lambda t: t * 180 / np.pi

    two_theta_list = [
        2 * to_degree(to_theta(q)) for q in q_list
    ]  # factor 2 to have 2theta angle and not just theta

    return two_theta_list


def _display_data(
    data, fig_size=(5, 5), plot_img=False, x_label=None, y_label=None, dpi=200
):
    """
    Display the data in a plot.

    Parameters
    ----------
    data : tuple or 2D array
        The data to display. If a tuple, the first element is the x axis and the second element is the y axis.
    fig_size : tuple of two floats, optional
        The size of the figure in inches
    plot_img : bool, optional
        If True, the data is displayed as an image
    x_label : str, optional
        The label of the x axis
    y_label : str, optional
        The label of the y axis
    dpi : int, optional
        The resolution of the figure in dots per inch

    Returns
    -------
    None
    """
    plt.figure(figsize=fig_size, dpi=dpi)

    if not plot_img:
        plt.plot(data[0], data[1], linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    else:
        plt.imshow(data, interpolation="none")
        plt.clim(0, 50)
        plt.colorbar(orientation="vertical")

    plt.show()

    return None


def create_folders(*args):
    """
    Create a folder or folders if they do not already exist.

    Parameters
    ----------
    *args : str
        The names of the folders to create

    Returns
    -------
    None
    """
    for arg in args:
        try:
            print(f"Creating folder {arg}")
            os.mkdir(arg)
        except FileExistsError:
            print(f"Folder {arg} already exists.")

    return None


def extract_CdTe_data(
    foldername, scan_number, display=True, raw_data_path="./ESRF_data/RAW_DATA/"
):
    """
    Extract 2D camera (CdTe) data from a raw .h5data file of a scan.

    Parameters
    ----------
    foldername : str
        The name of the folder containing the raw data file
    scan_number : int
        The number of the scan to extract
    display : bool, optional
        If True, display the extracted data
    raw_data_path : str, optional
        The path to the raw data folder. By default, "./ESRF_data/RAW_DATA/"

    Returns
    -------
    data : 2D array
        The extracted 2D camera image acquisition

    """
    file = f"{foldername}_0001.h5"
    CdTe_img_group_path = f"{scan_number}.1/measurement/CdTe/"
    fullpath = raw_data_path / pathlib.Path(
        foldername + "/" + foldername + "_0001/" + file
    )
    print(f'Looking inside "{fullpath}" in group path "{CdTe_img_group_path}"')

    with h5py.File(fullpath, "r") as fh5:
        img = fh5[CdTe_img_group_path]

        if img.shape[0] != 1:
            print(
                "Warning : Calibration scan, measured data starts at scan number > 26"
            )

        data = img[()][0]
        if display:
            _display_data(img[0], plot_img=True)

    return data


def extract_integrated_data(
    foldername,
    scan_number,
    display=True,
    processed_data_path="./ESRF_data/PROCESSED_DATA/",
    scaling=1000,
):
    """
    Extract integrated CdTe data from a processed .h5 data file of a scan.

    Parameters
    ----------
    foldername : str
        The name of the folder containing the processed data file
    scan_number : int
        The number of the scan to extract
    display : bool, optional
        If True, display the extracted data as a plot
    processed_data_path : str, optional
        The path to the processed data folder. By default, "./ESRF_data/PROCESSED_DATA/"
    scaling : int or float, optional
        The factor by which to scale the counts data

    Returns
    -------
    theta_values : list of floats
        The list of 2Theta angle values corresponding to the q values
    counts : list of floats
        The list of scaled counts from the integrated data
    """
    file = f"{foldername}_0001.h5"

    fullpath = processed_data_path / pathlib.Path(
        foldername + "/" + foldername + "_0001/" + file
    )
    # print(fullpath)

    with h5py.File(fullpath, "r") as fh5:
        counts = [
            count * scaling
            for count in fh5[f"{scan_number}.1/measurement/CdTe_integrated"][0]
        ]
        q_values = fh5[f"{scan_number}.1/CdTe_integrate/integrated/q"][()]
        theta_values = _convert_from_q_to_theta(q_values)

    if display:
        _display_data(
            [theta_values, counts],
            fig_size=(8, 5),
            x_label="2Theta (°)",
            y_label="Counts (au)",
        )

    return theta_values, counts


def save_integrated_data(
    foldername, scan_number, data, saved_data_path="./ESRF_data/SAVED_DATA/"
):
    """
    Save integrated data to a file.

    Parameters
    ----------
    foldername : str
        The name of the folder to create and save the data in
    scan_number : int
        The number of the scan to extract
    data : list of lists
        The list of theta values and the list of corresponding counts
    saved_data_path : str, optional
        The path to the directory to save the data in. Defaults to "./ESRF_data/SAVED_DATA/"

    Returns
    -------
    None
    """
    if foldername not in os.listdir(saved_data_path):
        try:
            os.mkdir(saved_data_path / pathlib.Path(foldername))
        except FileExistsError:
            pass
    fullpath = saved_data_path / pathlib.Path(
        f"{foldername}/{foldername}_{scan_number}.xy"
    )

    with open(fullpath, "w") as sf:
        for i, line in enumerate(data[0]):
            sf.write(f"{line}\t{data[1][i]}\n")

    return None


def save_all_integrated(foldername, saved_data_path="./ESRF_data/SAVED_DATA/"):
    """
    Save all integrated data from a folder to .xy files.

    Parameters
    ----------
    foldername : str
        The name of the folder to save the data in
    saved_data_path : str, optional
        The path to the directory to save the data in. Defaults to "./ESRF_data/SAVED_DATA/"

    Returns
    -------
    None
    """
    scan_list = range(27, 316)

    for scan_number in tqdm(scan_list):
        data = extract_integrated_data(foldername, scan_number, display=False)
        save_integrated_data(
            foldername, scan_number, data, saved_data_path=saved_data_path
        )

    print(
        f"All .xy spectrum saved in {saved_data_path / pathlib.Path(foldername)} succesfully !"
    )

    return None
