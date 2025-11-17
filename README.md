# MDC_cut: An ARPES data analysis tool
![Python Version](https://img.shields.io/badge/Python-3.12_|_3.13-3776AB?logo=python&logoColor=%233776AB)
![GitHub License](https://img.shields.io/github/license/alex20000910/main?logo=licenses&logoColor=white)
![GitHub Release](https://img.shields.io/github/v/release/alex20000910/main)
![GitHub Release Date](https://img.shields.io/github/release-date/alex20000910/main)
![GitHub Created At](https://img.shields.io/github/created-at/alex20000910/main)

This application is specifically designed to process data produced by [**Spectrium**](https://prevac.pl/produkt/spectrium/), a software product of [**PREVAC sp. z o.o.**](https://prevac.pl/)

[**MDC_cut.py**](MDC_cut.py) provides a user-friendly GUI for handling datasets acquired with the [**PREVAC EA15**](https://prevac.eu/product/hemispherical-energy-analyser-ea15/) hemispherical electron energy analyzer.

See [`MDC_cut_UserManual.pdf`](MDC_cut_UserManual.pdf) for detailed instructions.
The user manual is currently out of date. For the latest changes and new features, refer to the release notes.

- **One Click to Go**: Launch instantly — no setup, no friction.
- **Intuitive Interface**: Clean, user-friendly design with zero learning curve.
- **Comprehensive Toolkit**: All-in-one solution for ARPES data visualization and analysis.
    - **Sample Offset Fitter**: Precisely calibrate sample orientation.
    - **k-plane Conversion**: Transform raw data into momentum space seamlessly.
    - **Volume Viewer**: Explore and slice 3D k-space data with ease.
    - **MDC Fitter**: Analyze momentum distribution curves efficiently.
    - **Spectrogram Tool**: Generate and select spectrograms from your data.
    - **Versatile Export Options**: Export data in graph(.png, .jpg...), HDF5, CasaXPS (.vms), and OriginPro (.opj/.opju) formats.

## Contents
- [**Usage**](#usage)
- [**What will `MDC_cut.py` do?**](#what-will-mdc_cutpy-do)
- [**Work Flow**](#work-flow)
  - [**Geometry Definition**](#geometry-definition)
  - [**E-Angle to E-k Conversion**](#e-angle-to-e-k-conversion)
- [**Requirements**](#requirements)
- [**Snapshots**](#snapshots)
## Usage
> [!WARNING]
> It is highly recommended to run `MDC_cut.py` in a virtual environment to avoid conflicts with existing packages in your main Python environment.
> 
> `MDC_cut.py` will attempt to install the required dependencies automatically by **pip** if they are not already present.
>
> Make sure you have an environment with **Python 3.12 or above**.
- Please download [`MDC_cut.py`](MDC_cut.py) and place it in the directory you want.
- Create a virtual environment (e.g., via [![Anaconda](https://img.shields.io/badge/Anaconda-white?logo=anaconda&logoColor=%23)](https://www.anaconda.com/download)) with the required Python version to avoid changing your main environment, then run `MDC_cut.py` to automatically install the dependencies.
- If you don’t mind the environment, execute `MDC_cut.py` and check that it automatically installs the required dependencies.

## What will `MDC_cut.py` do?
1. Download the required files from this repository and place them into the working PATH of `MDC_cut.py`.
2. Try to use the pip installer to install the required python packages.
3. Start the GUI.

## Work Flow
![alt text](src/img/img_raw.png)
![alt text](src/img/img_k_plane.png)
![alt text](src/img/img_Volume_Viewer.png)

> [!IMPORTANT]
> Always specify the file name with clear geometric suffixes like
>> **Name1_R1_18_R2_85.h5**, **Name2_r1_12_r2_45.h5**
>> 
> for Angular LensMode
>> 
> or
>> **Name3_X_12_Z_45.h5**, **Name4_x_11_z_48.h5**
>> 
> for Transmission LensMode
>> 
> to well define the geometry of the spectum acquired.

### Geometry Definition
The geometry of the data is defined by the angles or positions of the sample manipulator motors.
The following table illustrates the relationship between motor positions and the corresponding energy and angles resolved by the EA15 analyzer.
> [!NOTE]
> The values shown in the images below are for illustration purposes only and do not represent actual measurements.
> The actual positions and angles of the motors might not be exactly the same as those depicted.
>
> For example, R1=-31° corresponds to $\psi$=0° and R2=85° corresponds to $\phi$=0° in the **Spectrium** software.
>
> However, users might not suffer from this offset since they can always calibrate the offset angle in the **k-plane** tool.
> Additionally, there would be no difference in the k-space conversion as long as the relative angles are correct.

<div align="center">

|Energy|Angle (resolved by EA15 lens)|
|:---:|:---:|
|![](src/img/geo_E.gif)|![](src/img/geo_theta.gif)|

|$\phi$ (Rotation around Y-axis, **R2** Motor)|$\psi$ (Tilt around Z-axis, **R1** Motor)|
|:---:|:---:|
|![](src/img/geo_phi.gif)|![](src/img/geo_psi.gif)|

</div>

### E-Angle to E-k Conversion
The application will parse the file name to extract the geometry information. R1 and R2 represent the motor of the sample manipulator, while the numbers denote the angles(in degrees) or position(in millimeters) of the respective motors.

Let's say we have some ARPES raw data files (`.h5`/`.json` format) from the PREVAC EA15 analyzer.

The typical workflow would be:
1. Launch `MDC_cut.py`.
2. Use the GUI to load your raw data files.
3. Open **k-plane** tool in **Batch Master** to convert raw data into k-space.
> [!NOTE]
> The **Batch Master** button only appears when loading multiple files.
4. Set the calibration parameters(Offset angle) in the **k-plane** tool and export the data.
> [!NOTE]
> **Sample Offset Fitter** can help you find the offset angle with ease.
5. After exporting, you can visualize the data in the **Volume Viewer** tool. It allows you to slice through the 3D k-space data export the desired 2D cuts in HDF5 format.
6. Further analysis like MDC fitting can be performed in the **MDC Fitter** tool by loading the exported 2D cuts back into `MDC_cut.py` using the **Load Raw Data** button.
>[!TIP]
> Many tools in `MDC_cut.py` have HOTKEY bindings for quick access. Try **Enter**, **Up Arrow**, **Down Arrow**, **Left Arrow**, **Right Arrow**, and **Scroll Wheel** for navigation, confirmation, and adjustments.

In addition to the steps mentioned above, `MDC_cut.py` provides a variety of tools for data visualization, exporting to [**CasaXPS**](https://www.casaxps.com/)(.vms), and generating [**OriginPro**](https://www.originlab.com/) projects(.opj). Explore the GUI to discover more features!
> [!NOTE]
> Change the OriginPro project format to `.opju` by directly modify the keyword in `MDC_cut.py` if you are using OriginPro 2018 or later. Typically, the older `.opj` format is still supported by later versions of OriginPro.

## Requirements
You don't need to manually install the dependencies. `MDC_cut.py` will automatically install them via pip if they are not already present in your environment.
The following are the tested Python versions and their corresponding package versions:

- ![Python 3.12.x](https://img.shields.io/badge/Python-3.12.x-3376AB?logo=python)
  - ![numpy 1.26.4](https://img.shields.io/badge/numpy-1.26.4-013243?logo=numpy&logoColor=013243)
  - ![opencv-python 4.10.0.84](https://img.shields.io/badge/opencv--python-4.10.0.84-5C3EE8?logo=opencv&logoColor=5C3EE8)
  - ![matplotlib 3.10.5](https://img.shields.io/badge/matplotlib-3.10.5-11557C)
  - ![xarray 2025.7.1](https://img.shields.io/badge/xarray-2025.7.1-4423ab)
  - ![h5py 3.14.0](https://img.shields.io/badge/h5py-3.14.0-ad2222)
  - ![Pillow 11.3.0](https://img.shields.io/badge/Pillow-11.3.0-32aa66)
  - ![scipy 1.16.1](https://img.shields.io/badge/scipy-1.16.1-8CAAE6?logo=scipy)
  - ![lmfit 1.3.4](https://img.shields.io/badge/lmfit-1.3.4-78bc99)
  - ![tqdm 4.67.1](https://img.shields.io/badge/tqdm-4.67.1-FFC107?logo=tqdm)
  - ![pywin32 311](https://img.shields.io/badge/pywin32-311-0012ae)
  - ![originpro 1.1.13](https://img.shields.io/badge/originpro-1.1.13-ae2399)
  - ![py-cpuinfo 9.0.0](https://img.shields.io/badge/py--cpuinfo-9.0.0-90fc36)
  - ![psutil 7.0.0](https://img.shields.io/badge/psutil-7.0.0-36aeff)
  - ![zarr 3.1.1](https://img.shields.io/badge/zarr-3.1.1-65cc11)
  - ![PyQt5 5.15.11](https://img.shields.io/badge/PyQt5-5.15.11-5c70ff)
  - ![pyqtgraph 0.13.7](https://img.shields.io/badge/pyqtgraph-0.13.7-dd4488)
  - ![tkinterdnd 0.4.3](https://img.shields.io/badge/tkinterdnd-0.4.3-35abaa)
- ![Python 3.13.x](https://img.shields.io/badge/Python-3.13.x-3376AB?logo=python)
  - ![numpy 2.2.6](https://img.shields.io/badge/numpy-2.2.6-013243?logo=numpy&logoColor=013243)
  - ![opencv-python 4.12.0.88](https://img.shields.io/badge/opencv--python-4.12.0.88-5C3EE8?logo=opencv&logoColor=5C3EE8)
  - ![matplotlib 3.10.5](https://img.shields.io/badge/matplotlib-3.10.5-11557C)
  - ![xarray 2025.7.1](https://img.shields.io/badge/xarray-2025.7.1-4423ab)
  - ![h5py 3.14.0](https://img.shields.io/badge/h5py-3.14.0-ad2222)
  - ![Pillow 11.3.0](https://img.shields.io/badge/Pillow-11.3.0-32aa66)
  - ![scipy 1.16.1](https://img.shields.io/badge/scipy-1.16.1-8CAAE6?logo=scipy)
  - ![lmfit 1.3.4](https://img.shields.io/badge/lmfit-1.3.4-78bc99)
  - ![tqdm 4.67.1](https://img.shields.io/badge/tqdm-4.67.1-FFC107?logo=tqdm)
  - ![pywin32 311](https://img.shields.io/badge/pywin32-311-0012ae)
  - ![originpro 1.1.13](https://img.shields.io/badge/originpro-1.1.13-ae2399)
  - ![py-cpuinfo 9.0.0](https://img.shields.io/badge/py--cpuinfo-9.0.0-90fc36)
  - ![psutil 7.0.0](https://img.shields.io/badge/psutil-7.0.0-36aeff)
  - ![zarr 3.1.1](https://img.shields.io/badge/zarr-3.1.1-65cc11)
  - ![PyQt5 5.15.11](https://img.shields.io/badge/PyQt5-5.15.11-5c70ff)
  - ![pyqtgraph 0.13.7](https://img.shields.io/badge/pyqtgraph-0.13.7-dd4488)
  - ![tkinterdnd 0.4.3](https://img.shields.io/badge/tkinterdnd-0.4.3-35abaa)

The only difference between Python 3.12.x and 3.13.x environments is the numpy and opencv-python versions due to pip version compatibility.
You can find the full list of required packages in the beginning section of `MDC_cut.py`.(**REQUIREMENTS**)
The highest tested Python version is 3.13.5.

## Snapshots
![alt text](src/img/img_main.png)
![alt text](src/img/img_SO_Fitter.png)
![alt text](src/img/img_CEC.png)
![alt text](src/img/img_DataViewer.png)
![alt text](src/img/img_Spectrogram.png)
![alt text](src/img/img_MDC_Fitter.png)
