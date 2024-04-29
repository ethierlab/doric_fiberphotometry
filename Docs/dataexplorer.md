# Data Explorer Analysis

## Overview:

The `dataexplorer.py` file is designed to explore and manage data from HDF5 files, focusing on processing and analyzing datasets from fiber photometry experiments.

## Classes:

### `H5DataExplorer`

A class designed to explore and manage data from HDF5 files, processing and visualizing datasets from fiber photometry experiments.

#### Methods:

- **`__init__`:**
  - Initializes the `H5DataExplorer` instance, setting up necessary attributes, including the file path, data frames, and configurations for dataset exploration.

- **`_populate_datasets2`:**
  - Populates datasets in an alternative format, managing different data structures and configurations.

- **`on_dropdown_change`:**
  - Handles changes in the dropdown menu, allowing users to select and manage datasets interactively.

- **`_populate_datasets`:**
  - Populates datasets in a standard format, processing data into usable forms for further analysis and visualization.

- **`on_dataset_info`:**
  - Provides information about the selected dataset, displaying relevant details.

- **`on_load_button_clicked`:**
  - Handles the load button event, loading the selected dataset for processing and analysis.

- **`get_isos_df`:**
  - Retrieves the dataset associated with the isos metric, converting it into a pandas DataFrame for further manipulation.

- **`get_grabda_df`:**
  - Retrieves the dataset associated with the grabda metric, converting it into a pandas DataFrame.

- **`get_event_df`:**
  - Retrieves the dataset associated with events, converting it into a pandas DataFrame for event analysis.

- **`set_data_frames`:**
  - Configures multiple data frames for processing and analysis.

- **`ish5dataset`:**
  - Checks if a given dataset is an HDF5 dataset, ensuring compatibility with the project's functions.

- **`h5getDatasetR`:**
  - Recursively retrieves datasets from an HDF5 file, processing them into usable forms.

- **`open_file`:**
  - Opens an HDF5 file, setting it up for further exploration and manipulation.

## Functions:

- **`find_rising_edges`:**
  - Identifies and returns rising edges from a given dataset, aiding in event classification.

- **`classify_events`:**
  - Classifies events based on predefined metrics, organizing them into appropriate categories.

- **`classify_events2`:**
  - Provides an alternative event classification method, using different metrics or configurations.

- **`find_optimal_time_window`:**
  - Determines the optimal time window for analysis, enhancing the precision of statistical measurements.

- **`plot_and_save_seperated`:**
  - Generates and saves visualizations for separated events, aiding in the analysis of different experiment conditions.

- **`min_max_normalize`:**
  - Applies min-max normalization to a dataset, scaling values to a defined range.

- **`plot_and_save`:**
  - Generates and saves general visualizations for a given dataset, including various graphs and charts.
