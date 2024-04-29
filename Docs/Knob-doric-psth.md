# Knob Doric PSTH Documentation

This Jupyter Notebook is designed to process and analyze photometry data. Below is a comprehensive breakdown of its key functionalities and the workflow it follows:

### 1. Import Data

The notebook imports essential packages for data manipulation, visualization, and custom functions for photometry data processing. It then utilizes a file selector utility to choose the datasets for analysis.

1. **Loading the Necessary Packages:**  
   Various packages like `numpy`, `pandas`, and `ipywidgets` are imported to handle data manipulation, visualization, and GUI creation. Additionally, modules like `fileselector` and `dataexplorer` provide specialized functionalities for working with photometry data.

2. **Selecting and Loading Data:**  
   A `FileSelector` object is instantiated, allowing users to choose datasets from a specified directory. After selection, data is loaded into three separate DataFrames: `isos_df`, `grabda_df`, and `event_df`.

3. **Plotting and Saving Data:**  
   The `plot_and_save_separated` and `plot_and_save` functions from the `dataexplorer` module are used to visualize and save plots of the datasets.

### 2. Event Classification

The notebook moves on to classifying events in the photometry data:

1. **Identifying Rising Edges:**  
   The DataFrame `event_df` is analyzed to identify rising edges, which are then used to create a `rising_edge_df`.

2. **Optimal Time Window:**  
   The `find_optimal_time_window` function calculates an optimal window around each event.

3. **Classifying Events:**  
   The `classify_events2` function classifies events into types such as "Init," "Success," and "Fail," producing a `classified_events_df`.

### 3. Data Preparation

This section prepares the photometry data for further analysis:

1. **Calculating z-dF/F:**  
   The `get_zdFF` function from the `photometry_functions` module calculates a normalized signal, which is stored in `signal_df`.

2. **Cutting and Centering:**  
   The `cut_and_center_signals_modified` function centralizes the signal around each event, creating separate DataFrames for each event type.

### 4. Visualization

Several types of visualizations are generated:

1. **Plotting Signals:**  
   Functions like `plot_cut_signals` and `plot_cut_signals_separated` visualize the centralized signals, both collectively and individually, and save them as image files.

2. **Creating PSTH:**  
   The `create_psth_with_min_max`, `create_psth_with_std`, and `create_psth_with_ci` functions generate PSTH (Peri-Stimulus Time Histogram) plots with different statistics, which are then visualized and saved.

3. **Generating Heatmaps:**  
   Heatmap plots for different signal types are created and saved.

4. **Area Under Curves (AUC):**  
   Functions like `plot_psth_auc` and `plot_auc_bars` are used to visualize the AUC of PSTH plots, providing a quantitative measure of the signals.

### 5. Saving Results

Finally, all generated DataFrames are saved to CSV files:

1. **Saving DataFrames:**  
   The `save_dataframes_to_csv` function iterates through all global variables, saving those that are DataFrames to CSV files with timestamps to avoid overwriting.
