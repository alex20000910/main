# MDC_cut: An ARPES data analysis tool
![GitHub License](https://img.shields.io/github/license/alex20000910/main)
![GitHub Release](https://img.shields.io/github/v/release/alex20000910/main)
![GitHub Release Date](https://img.shields.io/github/release-date/alex20000910/main)

This application is specifically designed to process data produced by [**Spectrium**](https://prevac.pl/produkt/spectrium/), a software product of [**PREVAC sp. z o.o.**](https://prevac.pl/)

[**MDC_cut.py**](MDC_cut.py) provides a user-friendly GUI for handling datasets acquired with the [**PREVAC EA15**](https://prevac.eu/product/hemispherical-energy-analyser-ea15/) hemispherical electron energy analyzer.

See [`MDC_cut_UserManual.pdf`](MDC_cut_UserManual.pdf) for detailed instructions.
The user manual is currently out of date. For the latest changes and new features, refer to the release notes.

## Usage
- Please download [`MDC_cut.py`](MDC_cut.py) and place it in the directory you want.
- Create a virtual environment (e.g., via [Anaconda](https://www.anaconda.com/download)) with the required Python version to avoid changing your main environment, then run `MDC_cut.py` to automatically install the dependencies.
- If you don’t mind the environment, execute `MDC_cut.py` and check that it automatically installs the required dependencies.

## What will `MDC_cut.py` do?
1. Download the required files from this repository and place them into the working PATH of `MDC_cut.py`.
2. Try to use the pip installer to install the required python packages.
3. Start the GUI.

## Work Flow
<img src="src\img\img_raw.png" width="50%">
<img src="src\img\img_k_plane.png" width="50%">
<img src="src\img\img_Volume_Viewer.png" width="50%">

> [!IMPORTANT]
> Always specify the file name with clear geometric suffixes like **File1_R1_18_R2_85** for Angular LensMode or **File2_X_12_Z_45** for Transmission LensMode to well define the geometry of the spectum acquired.

The application will parse the file name to extract the geometry information. R1 and R2 represent the motor of the sample manipulator, while 18 and 85 denote the angles(in degrees) of the respective motors.
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

In addition to the steps mentioned above, `MDC_cut.py` provides a variety of tools for data visualization, exporting to [**CasaXPS**](https://www.casaxps.com/)(.vms), and generating [**OriginPro**](https://www.originlab.com/) projects(.opj). Explore the GUI to discover more features!
> [!NOTE]
> Change the OriginPro project format to `.opju` by directly modify the keyword in `MDC_cut.py` if you are using OriginPro 2023 or later. Typically, the older `.opj` format is still supported by later versions of OriginPro.

## Snapshots
<img src="src\img\img_main.png" width="75%">
<img src="src\img\img_SO_Fitter.png" width="75%">
<img src="src\img\img_CEC.png" width="75%">
<img src="src\img\img_DataViewer.png" width="75%">
<img src="src\img\img_Spectrogram.png" width="75%">
<img src="src\img\img_MDC_Fitter.png" width="75%">
