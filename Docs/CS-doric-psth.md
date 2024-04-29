### CS Doric PSTH Documentation

#### 1. Importing Data
The initial code cell includes necessary libraries and packages such as numpy, pandas, and matplotlib for handling data and plotting graphs. It also initializes a file selection module (`fileselector`) to load datasets from a specified directory. This enables the subsequent code cells to handle specific datasets directly from the chosen files.

#### 2. Loading and Displaying Data
After selecting the datasets, functions are defined to read and load three types of data: `isos_df`, `grabda_df`, and `event_df`. Additionally, functions from the `dataexplorer` module are reloaded and used to generate plots for each dataset and save them separately and collectively.

#### 3. CS Events
A section dedicated to detecting and classifying events from `event_df`. Functions from the `dataexplorer` module handle the identification of rising edges and then classify these events into types, with an example output displayed for the user.

#### 4. Data Preparation
The notebook introduces `get_zdFF` from `photometry_functions.py` to calculate z-dF/F, which is used in the subsequent analysis. This section includes necessary citations for the source of this functionality.

#### 5. PSTH Preparation
Data is cut and centered around specified time windows, handling different event types like "Reward" or "Omit," which are processed into separate datasets. These datasets are then explored using several plotting functions:

- **All Signals:** Plots and saves all signals within each dataset.
- **Signals Separately:** Separately plots and saves signals for each event type.
- **Signal Selection for PSTH:** Allows interactive filtering and selection of signals, with visual outputs saved accordingly.

#### 6. Normalization
Introduces functions from `photometry_functions` for normalizing signals. Both standard normalization and a two-step normalization (not recommended due to AUC interference) are covered. These ensure data is consistent for subsequent analysis.

#### 7. Results Analysis
Functions are defined to create and plot PSTHs (Peri-Stimulus Time Histograms) for the reward and omit datasets, using different metrics (CI, min-max, and std). These histograms are then visualized across a 3x1 grid layout.

#### 8. Heatmap
Generates heatmaps for the normalized datasets (`Reward` and `Omit`), displaying them and saving their visualizations.

#### 9. Area Under the Curve (AUC)
This section provides functions to compute and plot AUCs and AUC bars, handling both the reward and omit datasets.

#### 10. Saving Results
A final utility function to save all DataFrames to CSV files with timestamps to avoid overwriting. This section loops through global variables and saves any `pandas` DataFrame into the specified directory.
