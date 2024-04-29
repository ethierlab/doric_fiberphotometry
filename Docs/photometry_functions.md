# Photometry Analysis Functions

This module provides various functions for processing, analyzing, and visualizing photometry data. The code includes algorithms for calculating standardized ΔF/F signals, baseline corrections, signal smoothing, and visualizations.

## Overview

- **Module Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`, `sklearn`
- **Author:** Ekaterina Martianova (ekaterina.martianova.1@ulaval.ca)
- **Reference:** Martianova, E., Aronson, S., Proulx, C.D. "Multi-Fiber Photometry to Record Neural Activity in Freely Moving Animal." J. Vis. Exp. (152), e60278 (2019).

## Functions

### `get_zdFF(reference, signal, smooth_win=10, remove=0, lambd=5e11, porder=10, itermax=50)`
Calculates a z-score ΔF/F signal based on fiber photometry recordings.

- **Parameters:**
  - `reference`: 1D array of calcium-independent signals.
  - `signal`: 1D array of calcium-dependent signals.
  - `smooth_win`: (Optional) Window size for smoothing the signal. Default is 10.
  - `remove`: (Optional) Number of data points to remove from the beginning. Default is 0.
  - `lambd`: (Optional) Regularization parameter for baseline correction. Default is 5e11.
  - `porder`: (Optional) Polynomial order for baseline correction. Default is 10.
  - `itermax`: (Optional) Max iterations for baseline correction. Default is 50.

- **Returns:** A 1D numpy array of the z-scored ΔF/F signal.

### `smooth_signal(x, window_len=10, window='blackman')`
Smooths the data using a specified window.

- **Parameters:**
  - `x`: Input signal (1D array).
  - `window_len`: Dimension of the smoothing window.
  - `window`: Type of window (e.g., 'flat', 'hanning', etc.).

- **Returns:** A smoothed signal.

### `WhittakerSmooth(x, w, lambda_, differences=1)`
Performs penalized least squares background fitting.

- **Parameters:**
  - `x`: Input data.
  - `w`: Binary masks for peaks.
  - `lambda_`: Regularization parameter.
  - `differences`: Order of difference for penalties.

- **Returns:** A fitted background vector.

### `airPLS(x, lambda_=100, porder=1, itermax=15)`
Adaptive iteratively reweighted penalized least squares for baseline fitting.

- **Parameters:**
  - `x`: Input data.
  - `lambda_`: Regularization parameter.
  - `porder`: Polynomial order for baseline fitting.
  - `itermax`: Max iterations for baseline fitting.

- **Returns:** A fitted background vector.

### `calculate_sample_rate(df)`
Calculates the sample rate from a DataFrame.

- **Parameters:** `df`: DataFrame with a 'Time' column.
- **Returns:** Sample rate as a float.

### `calculate_window_len(cutoff_freq, sample_rate)`
Calculates the window length for a flat window given a cutoff frequency.

- **Parameters:**
  - `cutoff_freq`: Cutoff frequency in Hz.
  - `sample_rate`: Sample rate in Hz.

- **Returns:** Window length as an odd integer.

### `normalize_signal(df, column='Data')`
Normalizes a signal in the DataFrame using z-score.

- **Parameters:**
  - `df`: DataFrame containing signal data.
  - `column`: Column name of the signal to normalize.

- **Returns:** A new DataFrame with normalized signal data.

### `create_psth_with_std(filtered_signal_df, decimal_places=2)`
Creates a PSTH DataFrame with standard deviation and mean values.

- **Parameters:**
  - `filtered_signal_df`: DataFrame with filtered signal data.
  - `decimal_places`: Number of decimal places for rounding 'Time' values.

- **Returns:** A DataFrame with 'Time', 'Average Signal', and 'Standard Deviation'.

### `create_psth_with_ci(filtered_signal_df, decimal_places=2)`
Creates a PSTH DataFrame with 95% confidence intervals.

- **Parameters:**
  - `filtered_signal_df`: DataFrame with filtered signal data.
  - `decimal_places`: Number of decimal places for rounding 'Time' values.

- **Returns:** A DataFrame with 'Time', 'Average Signal', 'Lower 95% CI', and 'Upper 95% CI'.

### `create_psth_with_min_max(filtered_signal_df, decimal_places=2)`
Creates a PSTH DataFrame with min, max, and mean values.

- **Parameters:**
  - `filtered_signal_df`: DataFrame with filtered signal data.
  - `decimal_places`: Number of decimal places for rounding 'Time' values.

- **Returns:** A DataFrame with 'Time', 'Average Signal', 'Minimum Signal', and 'Maximum Signal'.

### `plot_cut_signals(cut_signals_df, y_min, y_max)`
Plots each cut signal from `cut_signals_df`.

- **Parameters:**
  - `cut_signals_df`: DataFrame of cut signals.
  - `y_min`: Minimum y-axis value.
  - `y_max`: Maximum y-axis value.

- **Returns:** None.

### `plot_cut_signals_separated(cut_signals_df, y_min, y_max)`
Plots each cut signal in a grid layout.

- **Parameters:** Same as `plot_cut_signals`.

### `heatmap_plot(df)`
Plots a heatmap of signal data by time and event.

- **Parameters:** `df`: DataFrame with signal data.

- **Returns:** None.

### `plot_psth_with_std_cloud(psth_df, y_min, y_max, color, signal)`
Plots a PSTH with standard deviation as a shaded cloud.

- **Parameters:**
  - `psth_df`: DataFrame with PSTH data.
  - `y_min`: Minimum y-axis value.
  - `y_max`: Maximum y-axis value.
  - `color`: Line color.
  - `signal`: Signal name.

- **Returns:** None.

### `plot_psth_with_min_max_range(psth_df, y_min, y_max, color, signal)`
Plots a PSTH with min and max values as a shaded range.

- **Parameters:** Same as `plot_psth_with_std_cloud`.

### `plot_psth(psth_df, y_min, y_max, color, signal, pos)`
Plots a PSTH with various data.

- **Parameters:**
  - `pos`: Axes position for plotting.

- **Returns:** None.

### `plot_psth_auc(psth_df, y_min, y_max, color, signal)`
Plots a PSTH with AUC highlighted.

- **Parameters:** Same as `plot_psth_with_std_cloud`.

### `plot_auc_bars(psth_df, y_min, y_max)`
Plots AUC values before and after time zero.

- **Parameters:**
  - `psth_df`: DataFrame with PSTH data.
  - `y_min`: Minimum y-axis value.
  - `y_max`: Maximum y-axis value.

- **Returns:** None.

## Classes

### `PhotometryAnalysis`

A class for photometry data analysis.

- **Attributes:**
  - `isos_df`: DataFrame with isos signals.
  - `grabda_df`: DataFrame with grabda signals.
  - `zdFF`: z-score ΔF/F signal.
  - `signal_df`: DataFrame with processed signals.
  - `save_callback`: Callback function for saving data.

- **Methods:**
  - `check_install()`: Checks if a package is installed.
  - `setup_widgets()`: Sets up widgets for analysis.
  - `on_button_clicked()`: Handles button click event.
  - `min_max_normalize()`: Normalizes a DataFrame.
  - `plot_and_save()`: Plots and saves signals.
  - `display_widgets()`: Displays widgets.
  - `save_selected_data()`: Saves the selected data.

### `SignalExplorer`

A class for exploring photometry signals.

- **Attributes:**
  - `centralized_signals_df`: DataFrame with centralized signals.
  - `filtered_signal_df`: DataFrame with filtered signals.
  - `save_callback`: Callback function for saving data.
  - `checkboxes`: Checkboxes for selecting events.

- **Methods:**
  - `save_selected_data()`: Saves selected data.
  - `plot_selected_signals()`: Plots selected signals.
  - `on_button_clicked()`: Handles button click event.
  - `select_all_handler()`: Handles "Select All" checkbox event.
  - `init_widgets()`: Initializes widgets.
  - `display_widgets()`: Displays widgets.

### Utility Functions

#### `find_nearest(array, value)`
Finds the nearest value in an array.

- **Parameters:**
  - `array`: Array of values.
  - `value`: Target value.

- **Returns:** Nearest value.

#### `cut_and_center_signals(signal_df, event_df, time_window, event_type="All")`
Cuts and centers signals around event times.

- **Parameters:**
  - `signal_df`: DataFrame with signal data.
  - `event_df`: DataFrame with event data.
  - `time_window`: Tuple of time window.
  - `event_type`: Type of event to consider.

- **Returns:** DataFrame with cut and centered signals.

#### `cut_and_center_signals2()`

 and `cut_and_center_signals_modified()`
Alternatives for cutting and centering signals.
