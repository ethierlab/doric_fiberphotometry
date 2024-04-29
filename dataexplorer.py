import h5py
import numpy as np
import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from ipywidgets import GridBox, Layout


class H5DataExplorer:
    def __init__(self, file_selector):        
        self.file_selector = file_selector
        self.file = None
        self.isos_selector = None
        self.grabda_selector = None
        self.event_selector = None
        self.load_button = widgets.Button(description="Load Data")
        self.load_button.on_click(self.on_load_button_clicked)# Define a consistent layout for the labels
        label_layout = widgets.Layout(width='100%', height='100px')
        self.isos_info = widgets.Textarea(value='', description='Isos Info:', disabled=True,  layout=label_layout)
        self.grabda_info = widgets.Textarea(value='', description='Grabda Info:', disabled=True, layout=label_layout)
        self.event_info = widgets.Textarea(value='', description='Event Info:', disabled=True, layout=label_layout)
        self.isos = pd.DataFrame()
        self.grabda = pd.DataFrame()
        self.event = pd.DataFrame()
        
    def _populate_datasets2(self):
        if self.datasets:
            dataset_options = []
            auto_select_map = {'Isos': None, 'Grabda': None, 'Event': None}
            
            for d in self.datasets:
                for data_item in d['Data']:
                    
                    option = (f"{d['Name']} - {data_item['Name']}", (d['Name'], data_item['Name']))
                    
                    dataset_options.append(option)
                    # Check for automatic selection
                    
                    
                    if data_item['DataInfo'].get('Username', {}):
                        # print(data_item['DataInfo'].get('Username', {}))
                        if 'isos' in data_item['DataInfo'].get('Username', {}):
                            auto_select_map['Isos'] = option[1]
                        elif 'sig' in  data_item['DataInfo'].get('Username', {}):
                            auto_select_map['Grabda'] = option[1]
                        elif 'event' in  data_item['DataInfo'].get('Username', {}).lower():
                            auto_select_map['Event'] = option[1]
                        elif 'knobevent' in  data_item['DataInfo'].get('Username', {}).lower():
                            auto_select_map['Event'] = option[1]
                        elif 'bl' in  data_item['DataInfo'].get('Username', {}).lower():
                            auto_select_map['Event'] = option[1]

             
            # Create Dropdowns
            dropdown_layout = widgets.Layout(width='100%', height='100px')
            self.isos_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Isos:', layout=dropdown_layout)
            self.grabda_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Grabda:', layout=dropdown_layout)
            self.event_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Event:', layout=dropdown_layout)

            self.isos_selector.observe(self.on_dataset_info, names='value')
            self.grabda_selector.observe(self.on_dataset_info, names='value')
            self.event_selector.observe(self.on_dataset_info, names='value')
            
            # Set automatic selections if found
            if auto_select_map['Isos']:
                self.isos_selector.value = auto_select_map['Isos']
            if auto_select_map['Grabda']:
                self.grabda_selector.value = auto_select_map['Grabda']
            if auto_select_map['Event']:
                self.event_selector.value = auto_select_map['Event']

            # Show selectors and set event listeners
            self.isos_selector.observe(self.on_dataset_info, names='value')
            self.grabda_selector.observe(self.on_dataset_info, names='value')
            self.event_selector.observe(self.on_dataset_info, names='value')
            

            grid = GridBox(children=[self.isos_selector, self.grabda_selector, self.event_selector, 
                                    self.isos_info, self.grabda_info, self.event_info],
                        layout=Layout(
                            width='100%',
                            grid_template_rows='auto auto', # two rows
                            grid_template_columns='33% 33% 33%', # three columns of equal size
                            grid_gap='10px 10px' # spacing between widgets
                        ))

            display(grid)
            # Attach the change handler to the dropdowns
            self.isos_selector.observe(self.on_dropdown_change, names='value')
            self.grabda_selector.observe(self.on_dropdown_change, names='value')
            self.event_selector.observe(self.on_dropdown_change, names='value')
            
            self.load_button.layout.display = 'none' 
            display(self.load_button)
            # Check if all selections were successful, if not, show GUI
            if not all([auto_select_map['Isos'], auto_select_map['Grabda'], auto_select_map['Event']]):
                print("Not all datasets were automatically selected, please review and select manually.")
            

            else :
                self.on_load_button_clicked(self)
        else:
            print("No datasets available.")
        
    def on_dropdown_change(self,change):
        
        self.on_load_button_clicked(self)
        
    def _populate_datasets(self):
        if self.datasets:
            dataset_options = []
            for d in self.datasets:
                for data_item in d['Data']:
                    dataset_options.append((f"{d['Name']} - {data_item['Name']}", (d['Name'], data_item['Name'])))
            
            self.isos_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Isos:')
            self.grabda_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Grabda:')
            self.event_selector = widgets.Dropdown(options=[('', 'None')] + dataset_options, description='Event:')
            
        # if self.datasets:
        #     dataset_names = [d['Name'] for d in self.datasets]
        #     self.isos_selector = widgets.Dropdown(options=[('', 'None')] + dataset_names, description='Isos:')
        #     self.grabda_selector = widgets.Dropdown(options=[('', 'None')] + dataset_names, description='Grabda:')
        #     self.event_selector = widgets.Dropdown(options=[('', 'None')] + dataset_names, description='Event:')
            self.isos_selector.observe(self.on_dataset_info, names='value')
            self.grabda_selector.observe(self.on_dataset_info, names='value')
            self.event_selector.observe(self.on_dataset_info, names='value')
            display(widgets.VBox([self.isos_selector, self.isos_info, self.grabda_selector, self.grabda_info, self.event_selector, self.event_info]))
            display(self.load_button)
        else:
            print("No datasets available.")

    def on_dataset_info(self, change):
        selected_data_tuple = change['owner'].value
        if selected_data_tuple:
            dataset_name, data_name = selected_data_tuple  # Unpacking the tuple directly
            # Find the selected dataset
            selected_dataset = next((d for d in self.datasets if d['Name'] == dataset_name), None)
            if selected_dataset is not None:
                # Find the selected data item within the dataset and get its DataInfo
                selected_data = next((data for data in selected_dataset['Data'] if data['Name'] == data_name), None)
                if selected_data:
                    data_info = selected_data['DataInfo']
                    info_text = '\n'.join([f"{key}: {value}" for key, value in data_info.items()])
                else:
                    info_text = "Data not found."
                if change['owner'] is self.isos_selector:
                    self.isos_info.value = info_text
                elif change['owner'] is self.grabda_selector:
                    self.grabda_info.value = info_text
                elif change['owner'] is self.event_selector:
                    self.event_info.value = info_text
            else:
                # Handle the case where no dataset is found
                if change['owner'] is self.isos_selector:
                    self.isos_info.value = "Dataset not found."
                elif change['owner'] is self.grabda_selector:
                    self.grabda_info.value = "Dataset not found."
                elif change['owner'] is self.event_selector:
                    self.event_info.value = "Dataset not found."

    def on_load_button_clicked(self, b):
        if self.isos_selector.value:
            dataset_name, data_name = self.isos_selector.value
            selected_dataset = next(d for d in self.datasets if d['Name'] == dataset_name)
            selected_data = next(data for data in selected_dataset['Data'] if data['Name'] == data_name)
            # Assuming 'Time' data is stored in a specific way, adjust as needed
            time_data = next(data['Data'] for data in selected_dataset['Data'] if data['Name'] == 'Time')
            self.isos = pd.DataFrame({'Time': time_data, 'Data': selected_data['Data']})
        if self.grabda_selector.value:
            dataset_name, data_name = self.grabda_selector.value
            selected_dataset = next(d for d in self.datasets if d['Name'] == dataset_name)
            selected_data = next(data for data in selected_dataset['Data'] if data['Name'] == data_name)
            # Assuming 'Time' data is stored in a specific way, adjust as needed
            time_data = next(data['Data'] for data in selected_dataset['Data'] if data['Name'] == 'Time')
            self.grabda = pd.DataFrame({'Time': time_data, 'Data': selected_data['Data']})
        if self.event_selector.value:
            dataset_name, data_name = self.event_selector.value
            selected_dataset = next(d for d in self.datasets if d['Name'] == dataset_name)
            selected_data = next(data for data in selected_dataset['Data'] if data['Name'] == data_name)
            # Assuming 'Time' data is stored in a specific way, adjust as needed
            time_data = next(data['Data'] for data in selected_dataset['Data'] if data['Name'] == 'Time')
            self.event = pd.DataFrame({'Time': time_data, 'Data': selected_data['Data']})
        print("Datasets loaded.")
        
    def get_isos_df(self):
        return self.isos
    def get_grabda_df(self):
        return self.grabda
    def get_event_df(self):
        return self.event
    def set_data_frames(self, df1, df2, df3):
        self.data_frames = [df1, df2, df3]

    def ish5dataset(self, item):
        return isinstance(item, h5py.Dataset)

    def h5getDatasetR(self, item, leading=''):
        r = []
        for key in item:
            firstkey = list(item[key].keys())[0]
            if self.ish5dataset(item[key][firstkey]):
                r += [{'Name': leading + '_' + key, 'Data': [{'Name': k, 'Data': np.array(item[key][k]),
                                                              'DataInfo': {atrib: item[key][k].attrs[atrib] for atrib in item[key][k].attrs}} for k in item[key]]}]
            else:
                r += self.h5getDatasetR(item[key], leading + '_' + key)
        return r
    def open_file(self):
        file_path = self.file_selector.get_selected_file()
        if file_path:
            self.file = h5py.File(file_path, 'r')
            # print(self.file['DataAcquisition']['NC500']['Signals']['Series0001']['DigitalIO']['DIO01'])
            self.datasets = self.h5getDatasetR(self.file['DataAcquisition'], file_path)
            self._populate_datasets2()
        else:
            print("No file selected or file path is invalid.")

def find_rising_edges(event_df,time_diff_threshold=0):
    rising_edges = event_df['Data'].diff() == 1
    rising_edge_df = event_df[rising_edges].reset_index().rename(columns={'index': 'Sample_Number'})

    # Calculate time differences
    new_rising_edges = rising_edge_df['Time'].diff() > time_diff_threshold

    # Ensure the first item is always included
    new_rising_edges.iloc[0] = True

    # Create new DataFrame including the first item
    new_rising_edge_df = rising_edge_df[new_rising_edges].reset_index(drop=True)

    return new_rising_edge_df

def classify_events(df, time_window=2,Event_type=None):
    events = {'Time': [], 'Type': [], 'Sample_Number': []}
    group_start_sample = df.iloc[0]['Sample_Number']
    group_start_time = df.iloc[0]['Time']
    pulse_count = 1
    last_time = 0
    if Event_type.lower() == 'stim':
        print("stim")
        for index, row in df.iterrows():
            if index == 0:
                continue
            current_time = row['Time']
            current_sample = row['Sample_Number']
            if current_time - group_start_time <= time_window:
                pulse_count += 1
                if last_time > 0:
                    freq = 1 / (current_time - last_time)
                last_time = current_time
            else:
                events['Time'].append(group_start_time)
                events['Type'].append('Stim '+str(round(freq, 2))+'Hz')
                events['Sample_Number'].append(group_start_sample)
                last_time = 0
                group_start_time = current_time
                group_start_sample = current_sample
    elif Event_type.lower() == 'knob' or Event_type.lower() == 'cs':
        if Event_type.lower() == 'knob':
            Event_types = ['Init','Success','Fail']
        if Event_type.lower() == 'cs':
            Event_types = ['Que','Reward','Omit']
        for index, row in df.iterrows():
            if index == 0:
                continue
            current_time = row['Time']
            current_sample = row['Sample_Number']
            if current_time - group_start_time <= time_window:
                pulse_count += 1
            else:
                if pulse_count == 1:
                    events['Time'].append(group_start_time)
                    events['Type'].append(Event_types[0])
                    events['Sample_Number'].append(group_start_sample)
                elif pulse_count == 2:
                    events['Time'].append(group_start_time)
                    events['Type'].append(Event_types[1])
                    events['Sample_Number'].append(group_start_sample)
                elif pulse_count == 3:
                    events['Time'].append(group_start_time)
                    events['Type'].append(Event_types[2])
                    events['Sample_Number'].append(group_start_sample)
                group_start_time = current_time
                group_start_sample = current_sample
                pulse_count = 1
        # Classify the last group
        if pulse_count == 1:
            events['Time'].append(group_start_time)
            events['Type'].append(Event_types[0])
            events['Sample_Number'].append(group_start_sample)
        elif pulse_count == 2:
            events['Time'].append(group_start_time)
            events['Type'].append(Event_types[1])
            events['Sample_Number'].append(group_start_sample)
        elif pulse_count == 3:
            events['Time'].append(group_start_time)
            events['Type'].append(Event_types[2])
            events['Sample_Number'].append(group_start_sample)
    else:
        raise TypeError("event type is wrong please choose valid event type 'cs','Knob' and 'stim' are acceptable")
    return pd.DataFrame(events)

def find_optimal_time_window(df,Event_type=['Init','Success','Fail']):
    """
    Find the optimal time window for classifying events into 'Init', 'Success', and 'Fail' categories.
    The optimal window is the one that minimizes the number of 'Init' events not followed by 'Success' or 'Fail'.
    """
    time_diffs = df['Time'].diff().dropna()
    optimal_window = None
    min_unmatched_inits = float('inf')
    min_delta = float('inf')
    # Test different time windows
    for window in np.arange(0.001, 2, 0.001):
        # Classify events using the current time window
        events = classify_events(df, window)
        # Count 'Init' events not followed by 'Success' or 'Fail'
        unmatched_inits = sum((events['Type'] == Event_type[0]) & (events['Type'].shift(-1) == Event_type[0]))
        event_counts = events['Type'].value_counts()
        # delta = event_counts.get('Init', 0) - event_counts.get('Success', 0) - event_counts.get('Fail', 0)
        delta = abs(event_counts.get(Event_type[0], 0) - event_counts.get(Event_type[1], 0) - event_counts.get(Event_type[2], 0))
        # Update optimal window if better
        if unmatched_inits < min_unmatched_inits:
            # if delta < min_delta:
            min_unmatched_inits = unmatched_inits
            min_delta = delta
            optimal_window = window

    return optimal_window

def plot_and_save_seperated(isos_df,grabda_df,event_df,file_selector):
    # Create the plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting data (replace isos_df, grabda_df, event_df with your actual DataFrames)
    axs[0].plot(isos_df.Time, isos_df.Data)
    axs[0].set_title('Isos')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Data')

    axs[1].plot(grabda_df.Time, grabda_df.Data)
    axs[1].set_title('Grabda')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Data')

    axs[2].plot(event_df.Time, event_df.Data)
    axs[2].set_title('Event')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Data')

    plt.tight_layout()

    # Save file input and button
    pltname = file_selector.file_path.split('.')[0]+'/Figs/'+'/raw_data_seperated.png'
    
    save_path_input = widgets.Text(
        value=pltname,
        description='Seperated Plot File path:',
        disabled=False
    )
    # Extract directory from the full file path
    directory = os.path.dirname(save_path_input.value)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(save_path_input.value, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path_input.value}")

def min_max_normalize(df):
    return 2 * ((df - df.min()) / (df.max() - df.min())) - 1

def plot_and_save(isos_df, grabda_df, event_df,file_selector):
    # Normalize the data
    isos_normalized = min_max_normalize(isos_df['Data'])
    grabda_normalized = min_max_normalize(grabda_df['Data'])
    event_normalized = min_max_normalize(event_df['Data'])
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plotting normalized data
    ax.plot(isos_df['Time'], isos_normalized, label='Isos')
    ax.plot(grabda_df['Time'], grabda_normalized, label='Grabda')
    ax.plot(event_df['Time'], event_normalized, label='Event')
    # Adding labels and title
    ax.set_title('Normalized Signals (-1 to 1)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized Data')
    ax.legend()
    plt.tight_layout()
    filename = file_selector.file_path.split('.')[0]+'/Figs/'+'/raw_data_merged.png'
    # Save file input and button
    save_path_input = widgets.Text(
        value= filename,
        description='Plot File path:',
        disabled=False
    )
    # Extract directory from the full file path
    directory = os.path.dirname(save_path_input.value)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(save_path_input.value, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path_input.value}")

    
