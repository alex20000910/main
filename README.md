# MDC_cut: An ARPES data analysis tool

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
3. Strat the GUI.
