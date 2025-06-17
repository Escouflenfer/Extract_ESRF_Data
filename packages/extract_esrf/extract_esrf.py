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
import fabio
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pathlib
from ipywidgets import interact, fixed, IntSlider
from matplotlib.colors import LogNorm, Normalize
from pylab import figure, cm
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


def _get_all_datasets(name, node):
    """
    Recursively iterate over all datasets in an HDF5 group and add their
    names and values to a dictionary.

    Parameters
    ----------
    name : str
        The name of the current group
    node : h5py.Group or h5py.Dataset
        The current group or dataset in the HDF5 file

    Returns
    -------
    None
    """
    global metadata_dict

    if isinstance(node, h5py.Dataset):
        name = "/".join((node.name).split("/")[2:])

        if isinstance(node[()], bytes):
            val = node[()].decode()
        elif node.ndim == 0:
            val = str(node[()])
        elif node.ndim == 1:
            if node.size == 1:
                val = str(node[0])
            else:
                val = ", ".join([f"{i:.5f}" for i in node[:].tolist()])
        else:
            return None

        metadata_dict[name] = val
        return None


def _extract_integrated_metadata(h5_file, scan_number):
    """
    Extracts metadata from an HDF5 file corresponding to a given scan number.

    Parameters
    ----------
    h5_file : h5py.File
        The HDF5 file to extract metadata from
    scan_number : int
        The scan number to extract metadata for

    Returns
    -------
    metadata_dict : dict
        A dictionary of metadata
    """
    global metadata_dict
    metadata_dict = {}

    h5_file[f"{scan_number}.1/"].visititems(_get_all_datasets)

    return metadata_dict


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
    foldername,
    scan_number,
    display=True,
    output_metadata=False,
    raw_data_path="./ESRF_data/RAW_DATA/",
):
    """
    Extract data from a CdTe detector at ESRF BM02.

    Parameters
    ----------
    foldername : str
        The name of the folder containing the data
    scan_number : int
        The number of the scan to extract
    display : bool, optional
        If True, display the extracted data as an image
    raw_data_path : str, optional
        The path to the directory containing the raw data. Defaults to "./ESRF_data/RAW_DATA/"

    Returns
    -------
    data : 2D array
        The extracted data
    metadata : dict
        A dictionary of the metadata extracted from the file
    """
    file = f"{foldername}_0001.h5"
    CdTe_img_group_path = f"{scan_number}.1/measurement/CdTe/"
    fullpath = raw_data_path / pathlib.Path(
        foldername + "/" + foldername + "_0001/" + file
    )
    if not output_metadata:
        print(f'Looking inside "{fullpath}" in group path "{CdTe_img_group_path}"')

    with h5py.File(fullpath, "r") as fh5:
        metadata = _extract_integrated_metadata(fh5, scan_number)

        img = fh5[CdTe_img_group_path]

        if img.shape[0] != 1:
            print(
                "Warning : Calibration scan, measured data starts at scan number > 26"
            )

        data = img[()][0]
        if display:
            _display_data(img[0], plot_img=True)

    if output_metadata:
        return data, metadata

    return data


def extract_integrated_data(
    foldername,
    scan_number,
    display=True,
    processed_data_path="./ESRF_data/PROCESSED_DATA/",
    scaling=1e9,
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

    with h5py.File(fullpath, "r") as fh5:
        counts = [
            float(count) * scaling
            for count in fh5[f"{scan_number}.1/measurement/CdTe_integrated"][0]
        ]
        q_values = fh5[f"{scan_number}.1/CdTe_integrate/integrated/q"][()]
        unit_q = fh5[f"{scan_number}.1/CdTe_integrate/integrated/q"].attrs["units"]

        # correction patch for q values in nm^-1, needs to be cleaner
        if unit_q == "nm^-1":
            q_values = [float(q) / 10 for q in q_values]
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
    foldername,
    scan_number,
    data,
    save_metadata=False,
    raw_data_path="./ESRF_data/RAW_DATA/",
    saved_data_path="./ESRF_data/SAVED_DATA/",
):
    """
    Save the integrated data from a scan to a .xy file.

    Parameters
    ----------
    foldername : str
        The name of the folder containing the raw data file
    scan_number : int
        The number of the scan to save
    data : list or tuple
        The data to save, where the first element is the theta values and the second element is the counts
    save_metadata : bool, optional
        If True, save the metadata for the scan in a separate file
    raw_data_path : str, optional
        The path to the raw data folder. By default, "./ESRF_data/RAW_DATA/"
    saved_data_path : str, optional
        The path to the saved data folder. By default, "./ESRF_data/SAVED_DATA/"

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
        # f"{foldername}/{foldername.replace('_v2', '')}_{scan_number+221}.xy"
        f"{foldername}/{foldername}_{scan_number}.xy"
    )

    if save_metadata:
        _raw_data, metadata = extract_CdTe_data(
            foldername,
            scan_number,
            display=False,
            output_metadata=True,
            raw_data_path=raw_data_path,
        )

    with open(fullpath, "w") as sf:
        if save_metadata:
            for meta in metadata:
                sf.write(f"#{meta}: {metadata[meta]}\n")
        for i, line in enumerate(data[0]):
            sf.write(f"{line}\t{data[1][i]}\n")

    return None


def save_CdTe_data(
    foldername,
    scan_number,
    image,
    custom_format="img",
    saved_data_path="./ESRF_data/SAVED_DATA/",
):
    """
    Save a CdTe image to a specified file format.

    Parameters
    ----------
    foldername : str
        The name of the folder to save the image in
    scan_number : int
        The number of the scan for the image to save
    image : 2D array
        The image data to be saved
    custom_format : str, optional
        The file format to save the image in. Defaults to "img"
    saved_data_path : str, optional
        The path to the directory to save the image in. Defaults to "./ESRF_data/SAVED_DATA/"

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
        # f"{foldername}/{foldername.replace('_v2', '')}_{scan_number+221}.{custom_format}"
        f"{foldername}/{foldername}_{scan_number}.{custom_format}"
    )

    # Creating a new image with fabio
    img_file = fabio.dtrekimage.DtrekImage()
    img_file.data = image

    img_file.save(fullpath)

    return None


def save_all_integrated(
    foldername,
    raw_data_path="./ESRF_data/RAW_DATA/",
    processed_data_path="./ESRF_data/PROCESSED_DATA/",
    saved_data_path="./ESRF_data/SAVED_DATA/",
    custom_range=range(27, 316),
):
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
    scan_list = custom_range

    for scan_number in tqdm(scan_list):
        data = extract_integrated_data(
            foldername,
            scan_number,
            processed_data_path=processed_data_path,
            display=False,
        )
        save_integrated_data(
            foldername,
            scan_number,
            data,
            save_metadata=True,
            raw_data_path=raw_data_path,
            saved_data_path=saved_data_path,
        )

    print(
        f"All .xy spectrum saved in {saved_data_path / pathlib.Path(foldername)} succesfully !"
    )

    return None


def save_all_images(
    foldername,
    raw_data_path="./ESRF_data/RAW_DATA/",
    saved_data_path="./ESRF_data/SAVED_DATA/",
    custom_range=range(27, 316),
    custom_format="img",
):
    """
    Save all CdTe data from a folder to .img files.

    Parameters
    ----------
    foldername : str
        The name of the folder to save the data in
    saved_data_path : str, optional
        The path to the directory to save the data in. Defaults to "./ESRF_data/SAVED_DATA/"
    custom_range : list or range, optional
        The range of scan numbers to save the data for. Defaults to range(27, 316)
    custom_format : str, optional
        The file format to save the data in. Defaults to "img"

    Returns
    -------
    None
    """
    scan_list = custom_range

    for scan_number in tqdm(scan_list):
        image, metadata = extract_CdTe_data(
            foldername,
            scan_number,
            display=False,
            output_metadata=True,
            raw_data_path=raw_data_path,
        )

        save_CdTe_data(
            foldername,
            scan_number,
            image=image,
            custom_format=custom_format,
            saved_data_path=saved_data_path,
        )

    print(
        f"All CdTe data saved in {saved_data_path / pathlib.Path(foldername)} succesfully !"
    )


def plot_img(
    folderpath_str,
    index,
    img_data="None",
    vmin=0,
    vmax=60,
    scale="normal",
    aspect="1",
    plot=True,
):
    """
    Plot an image from either an .img file specified by folderpath_str and index, or a numpy array passed as img_data.

    Parameters
    ----------
    folderpath_str : str
        The path to the folder containing the .img files
    index : int
        The index of the .img file to read
    img_data : str or numpy array, optional
        The data to plot. If a string, read from an .img file. If a numpy array, plot that. Defaults to "None"
    vmin : float, optional
        The minimum value of the color scale. Defaults to 0
    vmax : float, optional
        The maximum value of the color scale. Defaults to 60
    scale : str, optional
        The scale of the color scale. Either "log" or "normal". Defaults to "normal"
    aspect : str, optional
        The aspect ratio of the plot. Defaults to "1"

    Returns
    -------
    None
    """

    if isinstance(img_data, str):
        folderpath = pathlib.Path(folderpath_str)
        filepath = folderpath / f"{folderpath.name}_{index}.img"
        img_file = fabio.open(filepath)
        img_data = img_file.data

    elif isinstance(img_data, np.ndarray):
        img_file = fabio.dtrekimage.DtrekImage()
        img_file.data = img_data

    if not plot:
        return img_data

    f = figure(figsize=(6, 4), dpi=200)
    ax = f.add_axes([0, 0, 1, 1])

    if vmin > vmax:
        vmin, vmax = vmax, vmin

    if scale.lower() == "log":
        im = ax.matshow(
            img_data,
            cmap=cm.rainbow,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            aspect=aspect,
        )  # Log scale
    elif scale.lower() == "normal":
        im = ax.matshow(
            img_data,
            cmap=cm.rainbow,
            norm=Normalize(vmin=vmin, vmax=vmax),
            aspect=aspect,
        )  # Normal scale
    else:
        print("Invalid scale option. Please choose 'log' or 'normal'.")
        return 1

    f.colorbar(im, format="%1.0f")

    return None


def display_all_img(
    folderpath_str, index=25, img_data="None", scale="normal", aspect="1"
):
    """
    Display all .img files in a folder as an interactive plot.

    Parameters
    ----------
    folderpath_str : str
        The path to the folder containing the image files.
    img_data : str or 2D array, optional
        The data to be displayed. If a string, it should be the path to the folder containing the image files.
        If a 2D array, it should be the data to be displayed directly. Defaults to "None".
    scale : str, optional
        The scale for the colorbar. Defaults to "normal".
    aspect : str, optional
        The aspect ratio of the image. Defaults to "1".

    Returns
    -------
    None
    """

    index_slider = IntSlider(min=25, max=273, step=1, value=index)
    vmin_slider = IntSlider(min=0, max=60, step=1, value=0)
    vmax_slider = IntSlider(min=0, max=255, step=1, value=60)

    interact(
        plot_img,
        folderpath_str=fixed(folderpath_str),
        index=index_slider,
        img_data=fixed(img_data),
        vmin=vmin_slider,
        vmax=vmax_slider,
        scale=fixed(scale),
        aspect=fixed(aspect),
        plot=fixed(True),
    )


def fuse_all_img(folderpath_str, idx_range=range(25, 274)):
    """
    Fuse multiple image files into a single image by summing their data.

    Parameters
    ----------
    folderpath_str : str
        The path to the folder containing the image files.
    idx_range : range, optional
        The range of indices for the image files to be fused. Defaults to range(25, 274).

    Returns
    -------
    fused_img : 2D array
        The fused image data obtained by summing the individual image data.
    """

    folderpath = pathlib.Path(folderpath_str)

    fused_img = []
    for index in idx_range:
        try:
            filepath = folderpath / f"{folderpath.name}_{index}.img"
            img_file = fabio.open(filepath)

            if len(fused_img) == 0:
                fused_img = img_file.data
            else:
                fused_img += img_file.data
        except FileNotFoundError:
            print(f"Warning, img file not found at position {index}")

    return fused_img


def rewrite_to_hdf5(processed_data_path, foldername, scan_number, data, filter_name):

    filepath = processed_data_path / pathlib.Path(
        f"{foldername}/{foldername}_0001/{foldername}_0001.h5"
    )

    with h5py.File(filepath, "a") as h5f:
        group_path = f"{scan_number}.1/CdTe_integrate/integrated/"
        try:
            del h5f[group_path]["q"]
            del h5f[group_path]["intensity"]
            h5f[group_path]["q"] = data[0]
            h5f[group_path]["intensity"] = [data[1]]
            h5f[group_path]["intensity"].attrs["filter"] = filter_name
        except KeyError:
            print(f"group path {group_path} not found")
            return 1

    return 0
