# FileSelector Analysis

## Overview
This module provides utilities for selecting files, processing and combining dataframes, and visualizing data related to photometry analysis. It uses various libraries to facilitate interactive selection, data manipulation, and visualization.

## Dependencies
Ensure the following packages are installed:

- **Standard Libraries:** `os`
- **Numerical and Scientific:** `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, `scipy.stats`, `scipy.sparse`
- **Machine Learning:** `sklearn`
- **File Handling:** `h5py`
- **Interactive Widgets:** `ipywidgets`
- **Custom Modules:** `dataexplorer`, `photometry_functions`

### Utility Function: `load_dependencies()`
Loads necessary libraries for data handling, visualization, and numerical processing.

## Classes

### `FileSelector`

This class provides an interface to select `.doric` files, load their data, and process it interactively.

- **Attributes:**
  - `file_path`: Path to the selected file.
  - `file_name`: Name of the selected file.
  - `start_dir`: Initial directory for file selection.
  - `file_selector`: Dropdown widget for file selection.
  - `isos`: DataFrame for isos signals.
  - `grabda`: DataFrame for grabda signals.
  - `event`: DataFrame for event signals.
  - `explorer`: Instance of `H5DataExplorer` for handling `.doric` files.

- **Methods:**
  - `_setup_widget()`: Initializes the file selection widget.
  - `_on_change()`: Event handler for file selection change, sets up the widget and loads the selected file.
  - `get_selected_file()`: Returns the selected file path.
  - `get_selected_file_name()`: Returns the name of the selected file.
  - `get_isos_df()`: Returns the isos DataFrame.
  - `get_grabda_df()`: Returns the grabda DataFrame.
  - `get_event_df()`: Returns the event DataFrame.

### `H5DataExplorer`
Handles `.doric` files, extracting relevant datasets and preparing them for analysis.

- **Attributes:**
  - `file_selector`: Instance of `FileSelector` for file management.

- **Methods:**
  - `open_file()`: Opens and processes the selected `.doric` file.
  - `get_isos_df()`: Returns a DataFrame for isos signals.
  - `get_grabda_df()`: Returns a DataFrame for grabda signals.
  - `get_event_df()`: Returns a DataFrame for event signals.

## Functions

### `save_combined_dataframe(isos_df, grabda_df, new_event_df, file_selector)`

Combines `isos_df`, `grabda_df`, and `new_event_df` into a single DataFrame and provides an option to save it to a file.

- **Parameters:**
  - `isos_df`: DataFrame with isos signals.
  - `grabda_df`: DataFrame with grabda signals.
  - `new_event_df`: DataFrame with event signals.
  - `file_selector`: Instance of `FileSelector` for managing the selected file.

- **Description:**
  - Checks if all DataFrames have the same length and time alignment.
  - Combines them into a single DataFrame with either all-time matching or distinct time columns.
  - Provides a widget for specifying the file path and a button for saving the combined DataFrame.
