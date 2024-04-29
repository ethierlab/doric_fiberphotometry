# Doric Fiber Photometry Data Analysis

This project provides a complete workflow for processing and analyzing data from fiber photometry experiments using Doric files in HDF5 format. The analysis is structured around three different tasks: "Stim," "CS," and "Knob," with results visualized in a variety of formats.

## Project Overview

This project aims to process, analyze, and visualize fiber photometry data, producing comprehensive insights into experimental results. It provides:

- **Data Processing:** Handling various formats, including CSV and HDF5, processing them into usable forms.
- **Data Analysis:** Generating PSTH (Peri-Stimulus Time Histogram) data, confidence intervals, and other statistical measures.
- **Visualization:** Creating bar charts, heatmaps, and other visualizations for raw and processed data.

## Installation Instructions

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd FiberPhotometryDataAnalysis
   ```

2. **Install Dependencies:**

   The project requires various Python packages. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**

   Ensure you have your experimental data organized in a folder structure similar to the one described in the project.

## Walkthrough: Jupyter Notebooks

### 1. [CS-doric-psth.ipynb](Docs/CS-doric-psth.md)

   - **Data Loading:** Loads datasets from the "CS" Folder from "Data".
   - **Data Processing:** Generates PSTH data, confidence intervals, and standard deviations, along with raw data processing.
   - **Visualization:** Creates bar charts, heatmaps, and PSTH graphs using matplotlib.
   - **Results:** Processed data and visualizations are saved in the corresponding data and figs directories.

### 2. [Knob-doric-psth.ipynb](Docs/Knob-doric-psth.md)

   - **Data Loading:** Loads datasets from the "Knob" Folder from "Data".
   - **Data Processing:** Generates PSTH data and associated statistics.
   - **Visualization:** Creates various visualizations including bar charts, heatmaps, and PSTH graphs.
   - **Results:** Processed data and visualizations are saved in the corresponding data and figs directories.

### 3. [Stim-doric-psth.ipynb](Docs/Stim-doric-psth.md)

   - **Data Loading:** Loads datasets from the "Stim" Folder from "Data".
   - **Data Processing:** Generates statistical analyses, including PSTH data and related statistics.
   - **Visualization:** Creates visualizations such as bar charts, heatmaps, and PSTH graphs.
   - **Results:** Processed data and visualizations are saved in the corresponding data and figs directories.

## Python Files Overview

1. **[fileselector.py](Docs/fileselector.md):**
   - **Purpose:** Facilitates file selection and data loading for analysis tasks.
   - **Features:** Integrates with `ipywidgets` to create interactive UI elements for selecting files and loads datasets using pandas.

2. **[photometry_functions.py](Docs/photometry_functions.md):**
   - **Purpose:** Provides utility functions for processing and analyzing fiber photometry data.
   - **Features:** Functions to compute standardized dF/F signals and integrates with numpy and pandas for data manipulations.

3. **[dataexplorer.py](Docs/dataexplorer.md):**
   - **Purpose:** Designed to explore and process data from HDF5 files used in fiber photometry.
   - **Features:** Uses `h5py` to handle HDF5 files and integrates with `numpy` and `pandas` for data manipulations.

---

Is there anything specific you'd like to modify or add? Let me know if you'd like more details on any part of the documentation.

## Project Structure

``````
├── Data
│   ├── CS
│   │   ├── CS_file
│   │   │   ├── Data
│   │   │   │   └── *.csv
│   │   │   └── Figs
│   │   │       └── *.png
│   │   └── *.doric
│   ├── Knob
│   │   ├── Knob_file
│   │   │   ├── Data
│   │   │   │   └── *.csv
│   │   │   └── Figs
│   │   │       └── *.png
│   │   └── *.doric
│   └── stim
│   │   ├── Knob_file
│   │   │   ├── Data
│   │   │   │   └── *.csv
│   │   │   └── Figs
│   │   │       └── *.png
│   │   └── *.doric
├── Docs
├── .gitignore
├── CS-doric-psth.ipynb
├── Knob-doric-psth.ipynb
├── Stim-doric-psth.ipynb
├── dataexplorer.py
├── fileselector.py
└── photometry_functions.py
``````

## Contribution

For contributions, please reach out to the project maintainers through the repository's issue tracker or directly.
