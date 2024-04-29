# Stim Doric PSTH Documentation

This documentation provides an overview of the functionality and features implemented in the Jupyter Notebook. It covers data import, processing, visualization, and saving results to CSV. Below is a comprehensive outline of each section and its corresponding code functionality.

## 1. Import Data

The initial section imports necessary libraries and modules, including:

- `numpy` and `pandas` for data manipulation.
- `fileselector` and `ipywidgets` for interactive widgets and file selection.
- Functions from `dataexplorer` and `photometry_functions` for specific data operations.

### Data Loading and Visualization

The notebook allows users to select datasets interactively, displaying information about them and enabling loading via a "Load Data" button. After this, the notebook:

- Loads and displays dataframes: `isos_df`, `grabda_df`, and `event_df`.
- Reloads necessary modules.
- Plots data separately and combined, and saves visualizations using `dataexplorer` functions.

## 2. Stim Events

This section focuses on processing event data:

- It uses `find_rising_edges()` to identify event edges.
- `classify_events2()` categorizes events by type.
- The section prints event information and counts.

## 3. Data Preparation

### Z-dF/F Calculation

A specialized function, `get_zdFF`, computes the z-dF/F metric, a measure of data deviation. The notebook references the `photometry_functions.py` file for this functionality, along with relevant research literature.

### Saving Signals

The notebook uses a `PhotometryAnalysis` class from `photometry_functions` to save signals into a global dataframe `signal_df`.

## 4. PSTH Preparation

### Data Centralization

The `cut_and_center_signals_modified()` function slices and aligns data around events within a defined time window. The `event_type` parameter determines the type of event processed.

### Plotting Signals

- **All Signals:** `plot_cut_signals()` visualizes all signals with specified Y-axis limits, saving the plot to a file.
- **Separate Signals:** `plot_cut_signals_separated()` visualizes signals individually and saves to another file.
- **Selected Signals for PSTH:** `SignalExplorer` provides an interface to select specific signals, saving them with a specified callback.

## 5. Normalization

The notebook offers normalization functionality:

- `normalize_signal()` scales data between -1 and 1, preparing it for PSTH analysis.
- Additional adjustments are shown, though not recommended due to potential impact on AUC.

## 6. Analyze Results

### PSTH

- Functions `create_psth_with_min_max()`, `create_psth_with_std()`, and `create_psth_with_ci()` generate various PSTH metrics.
- These are plotted in subplots, saved, and presented comprehensively.

### Heatmap

- A `heatmap_plot()` visualizes the normalized signal dataframe, saving the result as a heatmap figure.

### Area Under Curves (AUC)

- `plot_psth_auc()` visualizes the AUC for the PSTH data.
- `plot_auc_bars()` further visualizes this as bar plots.

## 7. Saving DataFrames to CSV

The notebook provides a function to save all global DataFrames to CSV files:

- It dynamically constructs filenames based on current datetime, preventing overwriting.
- Loops through global variables to save each dataframe to a unique file.

