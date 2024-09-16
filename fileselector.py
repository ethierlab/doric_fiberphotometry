import os
import ipywidgets as widgets
from IPython.display import display
import pandas as pd
import dataexplorer as de
from importlib import reload

def load_dependencies():
    # Standard library imports
    import os
    
    # Third-party imports for numerical operations
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from scipy.stats import zscore
    from scipy.sparse import csc_matrix, eye, diags
    from scipy.sparse.linalg import spsolve
    from sklearn.linear_model import Lasso
    import h5py
    
    # Imports for interactive widgets and IPython integration
    import ipywidgets as widgets
    from ipywidgets import GridBox, Layout
    from IPython.display import display, clear_output
    
    # Imports for custom or local modules
    import dataexplorer as de
    import photometry_functions as pf
    
    # Utility to reload modules during development
    from importlib import reload

class FileSelector:
    def __init__(self, start_dir, reload_data=False):
        self.file_path = None
        self.file_name = None
        self.reload_data = reload_data
        self.start_dir = start_dir
        self._setup_widget()
        self.isos = pd.DataFrame()
        self.grabda = pd.DataFrame()
        self.event = pd.DataFrame()
        self.explorer = de.H5DataExplorer(self)
        self.dataframes = pd.DataFrame()
    

    def _setup_widget(self):
        reload(de)
        file_list = [f for f in os.listdir(self.start_dir) if os.path.isfile(os.path.join(self.start_dir, f)) and f.endswith('.doric')]
        self.file_selector = widgets.Dropdown(options=[('Select a file')] + file_list, description='Select File:', disabled=False)
        self.file_selector.observe(self._on_change, names='value')
        display(self.file_selector)
    def load_all_csv_files(self):
        """
        Loads all CSV files from the specified directory and returns them as a dictionary of DataFrames.

        The directory is derived from the selected file's path.
        """
        directory_path = self.file_path.split('.')[0] + '/Data/'
        dataframes = {}
        
        # Iterate through each file in the directory
        for file_name in os.listdir(directory_path):
            # Check if the file is a CSV file
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)
                df = pd.read_csv(file_path)
                
                # Removing the '.csv' extension from the file name for the variable name
                var_name = file_name[:-20]  # Adjust as per your naming convention
                
                # Store the DataFrame in the dictionary
                dataframes[var_name] = df
                
                print(f"Loaded {file_name} into variable: {var_name}")
        
        return dataframes
    def _on_change(self, change):
        from IPython.display import clear_output
        from importlib import reload
        reload(de)
        
        if change['type'] == 'change' and change['name'] == 'value':
            clear_output(wait=True)
            self._setup_widget()
            self.file_path = os.path.join(self.start_dir, change['new'])
            self.file_name = change['new']
            print("Selected file:", self.file_path)
            
            # explorer = de.H5DataExplorer(self,self.isos_df,self.grabda_df,self.event_df)
            if self.reload_data:
                path = self.file_path.split('.')[0]+'/Data/'
                print(path)
                print("Reloading dataset from:", self.file_path)
                self.dataframes = self.load_all_csv_files()
            else:
                print("loading dataset .....")
                self.explorer.open_file()
            
    def get_dataframes(self):
        return self.dataframes
    def get_selected_file(self):
        return self.file_path
    def get_selected_file_name(self):
        return self.file_name
    def get_isos_df(self):
        return self.explorer.get_isos_df()
    def get_grabda_df(self):
        return self.explorer.get_grabda_df()
    def get_event_df(self):
        return self.explorer.get_event_df()



def save_combined_dataframe(isos_df, grabda_df, new_event_df,file_selector):
    # Check if all DataFrames have the same length
    if len(isos_df) == len(grabda_df) == len(new_event_df):
        # Compare 'time' columns
        same_time_isos_grabda = isos_df['Time'] == grabda_df['Time']
        same_time_isos_event = isos_df['Time'] == new_event_df['Time']
        same_time_grabda_event = grabda_df['Time'] == new_event_df['Time']

        # Combining all comparisons (if you want to check if all three are the same)
        all_same_time = same_time_isos_grabda & same_time_isos_event & same_time_grabda_event

        if (all_same_time.sum() == len(isos_df)):
            print("all time are the same across all dataframes")
            # Combine the DataFrames
            combined_df = pd.DataFrame({
                'Index': isos_df.index,  # or a common column like 'Time' if available
                'Isos': isos_df['Data'],
                'Grabda': grabda_df['Data'],
                'Event': new_event_df['Data'],
                'Time': new_event_df['Time']
            })
        else:
            # Combine the DataFrames
            combined_df = pd.DataFrame({
                'Index': isos_df.index,  # or a common column like 'Time' if available
                'Isos_Data': isos_df['Data'],
                'Isos_Time': isos_df['Time'],
                'Grabda_Data': grabda_df['Data'],
                'Grabda_Time': grabda_df['Time'],
                'Event_Data': new_event_df['Data'],
                'Event_Time': new_event_df['Time']
            })
        
        csvname = 'Data/'+file_selector.get_selected_file_name()+'_data.csv'
        # Create text input for the file path
        file_path_input = widgets.Text(value=csvname, description='File path:', disabled=False)

        # Create a button to trigger the save operation
        save_button = widgets.Button(description="Save Combined DataFrame")

        def save_csv(b):
            combined_df.to_csv(file_path_input.value, index=False)
            print(f"Combined DataFrame saved as {file_path_input.value}")

        save_button.on_click(save_csv)

        # Display the widgets
        display(file_path_input, save_button)
    else:
        print("DataFrames do not have the same length and cannot be combined directly.")
