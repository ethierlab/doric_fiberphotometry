import numpy as np
import pandas as pd



'''
get_zdFF.py calculates standardized dF/F signal based on calcium-idependent 
and calcium-dependent signals commonly recorded using fiber photometry calcium imaging

Ocober 2019 Ekaterina Martianova ekaterina.martianova.1@ulaval.ca 

Reference:
  (1) Martianova, E., Aronson, S., Proulx, C.D. Multi-Fiber Photometry 
      to Record Neural Activity in Freely Moving Animal. J. Vis. Exp. 
      (152), e60278, doi:10.3791/60278 (2019)
      https://www.jove.com/video/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving

'''

def get_zdFF(reference,signal,smooth_win=10,remove=0,lambd=5e11,porder=10,itermax=50): 
  '''
  Calculates z-score dF/F signal based on fiber photometry calcium-idependent 
  and calcium-dependent signals
  
  Input
      reference: calcium-independent signal (usually 405-420 nm excitation), 1D array
      signal: calcium-dependent signal (usually 465-490 nm excitation for 
                   green fluorescent proteins, or ~560 nm for red), 1D array
      smooth_win: window for moving average smooth, integer
      remove: the beginning of the traces with a big slope one would like to remove, integer
      Inputs for airPLS:
      lambd: parameter that can be adjusted by user. The larger lambda is,  
              the smoother the resulting background, z
      porder: adaptive iteratively reweighted penalized least squares for baseline fitting
      itermax: maximum iteration times
  Output
      zdFF - z-score dF/F, 1D numpy array
  '''
#   !pip install scikit-learn

  import numpy as np
  from sklearn.linear_model import Lasso

 # Smooth signal
  reference = smooth_signal(reference, smooth_win)
  signal = smooth_signal(signal, smooth_win)
  
 # Remove slope using airPLS algorithm
  r_base=airPLS(reference,lambda_=lambd,porder=porder,itermax=itermax)
  s_base=airPLS(signal,lambda_=lambd,porder=porder,itermax=itermax) 

 # Remove baseline and the begining of recording
  reference = (reference[remove:] - r_base[remove:])
  signal = (signal[remove:] - s_base[remove:])   

 # Standardize signals    
  reference = (reference - np.median(reference)) / np.std(reference)
  signal = (signal - np.median(signal)) / np.std(signal)
  
 # Align reference signal to DA signal using non-negative robust linear regression
  lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000,
              positive=True, random_state=9999, selection='random')
  n = len(reference)
  lin.fit(reference.reshape(n,1), signal.reshape(n,1))
  reference = lin.predict(reference.reshape(n,1)).reshape(n,)

 # z dFF    
  zdFF = (signal - reference)
 
  return zdFF


def smooth_signal(x,window_len=10,window='blackman'):

    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    The code taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                'flat' window will produce a moving average smoothing.

    output:
        the smoothed signal        
    """
    
    import numpy as np

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': # Moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y[(int(window_len/2)-1):-int(window_len/2)]


'''
airPLS.py Copyright 2014 Renato Lombardo - renato.lombardo@unipa.it
Baseline correction using adaptive iteratively reweighted penalized least squares

This program is a translation in python of the R source code of airPLS version 2.0
by Yizeng Liang and Zhang Zhimin - https://code.google.com/p/airpls

Reference:
Z.-M. Zhang, S. Chen, and Y.-Z. Liang, Baseline correction using adaptive iteratively 
reweighted penalized least squares. Analyst 135 (5), 1138-1146 (2010).

Description from the original documentation:
Baseline drift always blurs or even swamps signals and deteriorates analytical 
results, particularly in multivariate analysis.  It is necessary to correct baseline 
drift to perform further data analysis. Simple or modified polynomial fitting has 
been found to be effective in some extent. However, this method requires user 
intervention and prone to variability especially in low signal-to-noise ratio 
environments. The proposed adaptive iteratively reweighted Penalized Least Squares
(airPLS) algorithm doesn't require any user intervention and prior information, 
such as detected peaks. It iteratively changes weights of sum squares errors (SSE) 
between the fitted baseline and original signals, and the weights of SSE are obtained 
adaptively using between previously fitted baseline and original signals. This 
baseline estimator is general, fast and flexible in fitting baseline.


LICENCE
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is, 
                 the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.matrix(x)
    m=X.size
    i=np.arange(0,m)
    E=eye(m,format='csc')
    D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*D.T*D))
    B=csc_matrix(W*X.T)
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,
                 the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    m=x.shape[0]
    w=np.ones(m)
    for i in range(1,itermax+1):
        z=WhittakerSmooth(x,w,lambda_, porder)
        d=x-z
        dssn=np.abs(d[d<0].sum())
        if(dssn<0.001*(abs(x)).sum() or i==itermax):
            if(i==itermax): print('WARING max iteration reached!')
            break
        w[d>=0]=0 # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d<0]=np.exp(i*np.abs(d[d<0])/dssn)
        w[0]=np.exp(i*(d[d<0]).max()/dssn) 
        w[-1]=w[0]
    return z


def calculate_sample_rate(df):
    # Ensure the DataFrame is sorted by time
    df = df.sort_values(by='Time')
    
    # Calculate the time differences between consecutive rows
    time_diffs = df['Time'].diff().dropna()
    
    # Calculate the average time difference
    avg_time_diff = time_diffs.mean()
    
    # The sample rate is the inverse of the average time difference
    if avg_time_diff != 0:
        sample_rate = 1 / avg_time_diff
    else:
        sample_rate = float('inf')
    
    return sample_rate


def calculate_window_len(cutoff_freq, sample_rate):
    """
    Calculate the window length for a flat window given a cutoff frequency.

    :param cutoff_freq: The desired cutoff frequency in Hz
    :param sample_rate: The sample rate of the signal in Hz
    :return: The estimated window length as an odd integer
    """
    nyquist_rate = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist_rate  # Normalize the cutoff frequency
    window_len = int(0.44 / normalized_cutoff)

    # Ensure window_len is odd
    if window_len % 2 != 0:
        window_len += 1
    
    if window_len < 4:
        window_len = 4
    return window_len
    # # Example usage:
    # # sample_rate = 100  # Replace with your actual sample rate in Hz
    # cutoff_freq = 10  # Replace with your desired cutoff frequency in Hz
    # window_len = pf.calculate_window_len(cutoff_freq, sample_rate)
    # print("Estimated window length:", window_len)



def normalize_signal(df, column='Data'):
    """
    Normalize the signal data in the DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame containing the signal data.
    column (str): The name of the column containing the signal data to normalize.

    Returns:
    DataFrame: A new DataFrame with the normalized signal data.
    """
    import pandas as pd
    from scipy.stats import zscore

    normalized_df = df.copy()
    normalized_df[column] = zscore(normalized_df[column])
    
    return normalized_df

def normalize_signal2(df, column='Data'):
    import pandas as pd
    max=df['Data'].max()
    min=df['Data'].min()
    diff = max - min

    normal_df=(2*(df['Data'] - min)/diff)-1
    
    return normal_df


def create_psth_with_std(filtered_signal_df, decimal_places=2):
    """
    Create a PSTH DataFrame from the filtered signal data with standard deviation and mean values for each time point,
    aggregating across all events. Time values are rounded to a specified number of decimal places.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure that 'Time' is a column and not part of a MultiIndex
    if 'Time' in filtered_signal_df.index.names:
        filtered_signal_df = filtered_signal_df.reset_index()

    # Round the 'Time' values to the specified number of decimal places
    filtered_signal_df['Time'] = filtered_signal_df['Time'].round(decimal_places)

    # Aggregate data for each time point across all events
    psth_data = filtered_signal_df.groupby('Time')['Data'].agg(['mean', 'std']).reset_index()

    # Rename columns for clarity
    psth_data.columns = ['Time', 'Average Signal', 'Standard Deviation']

    return psth_data
def create_psth_with_ci(filtered_signal_df, decimal_places=2):
    """
    Create a PSTH DataFrame from the filtered signal data with 95% confidence interval and mean values for each time point,
    aggregating across all events. Time values are rounded to a specified number of decimal places.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    # Ensure that 'Time' is a column and not part of a MultiIndex
    if 'Time' in filtered_signal_df.index.names:
        filtered_signal_df = filtered_signal_df.reset_index()

    # Round the 'Time' values to the specified number of decimal places
    filtered_signal_df['Time'] = filtered_signal_df['Time'].round(decimal_places)

    # Aggregate data for each time point across all events and calculate count
    psth_data = filtered_signal_df.groupby('Time')['Data'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate Standard Error of the Mean (SEM)
    psth_data['sem'] = psth_data['std'] / np.sqrt(psth_data['count'])

    # Determine the t critical value for 95% confidence level
    psth_data['t_critical'] = psth_data['count'].apply(lambda x: stats.t.ppf(0.975, x-1))

    # Calculate the Margin of Error (MOE)
    psth_data['moe'] = psth_data['t_critical'] * psth_data['sem']

    # Calculate the Confidence Intervals
    psth_data['Lower 95% CI'] = psth_data['mean'] - psth_data['moe']
    psth_data['Upper 95% CI'] = psth_data['mean'] + psth_data['moe']

    # Select and rename columns for clarity
    psth_data = psth_data[['Time', 'mean', 'Lower 95% CI', 'Upper 95% CI']]
    psth_data.columns = ['Time', 'Average Signal', 'Lower 95% CI', 'Upper 95% CI']

    return psth_data

def create_psth_with_min_max(filtered_signal_df, decimal_places=2):
    """
    Create a PSTH DataFrame from the filtered signal data with minimum, maximum, and mean values for each time point,
    aggregating across all events. Time values are rounded to a specified number of decimal places.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure that 'Time' is a column and not part of a MultiIndex
    if 'Time' in filtered_signal_df.index.names:
        filtered_signal_df = filtered_signal_df.reset_index()

    # Round the 'Time' values to the specified number of decimal places
    filtered_signal_df['Time'] = filtered_signal_df['Time'].round(decimal_places)

    # Aggregate data for each time point across all events
    psth_data = filtered_signal_df.groupby('Time')['Data'].agg(['mean', 'min', 'max']).reset_index()

    # Rename columns for clarity
    psth_data.columns = ['Time', 'Average Signal', 'Minimum Signal', 'Maximum Signal']

    return psth_data

def plot_cut_signals(cut_signals_df,y_min,y_max):
    """
    Plot each cut signal from cut_signals_df.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Get the unique event indices
    event_indices = cut_signals_df.index.get_level_values('Event').unique()

    for event_index in event_indices:
        # Extract the signal for the current event
        event_signal = cut_signals_df.xs(event_index, level='Event')

        # Plot the signal
        plt.plot(event_signal['Time'], event_signal['Data'], label=f'Event {event_index}')

    plt.xlabel('Time(S) relative to event')
    plt.ylabel('z-dF/F')
    plt.title('Cut Signals Around Each Event')
    plt.legend()
    plt.ylim(y_min, y_max)

    plt.show()
def plot_cut_signals_seperated(cut_signals_df,y_min,y_max):
    """
    Plot each cut signal from cut_signals_df in a grid layout with 3 columns.
    """
    import matplotlib.pyplot as plt
    import math

    # Get the unique event indices
    event_indices = cut_signals_df.index.get_level_values('Event').unique()
    num_events = len(event_indices)

    # Determine the number of rows needed for 3 columns
    rows = math.ceil(num_events / 3)

    # Create the subplot grid
    fig, axs = plt.subplots(rows, 3, figsize=(18, 6 * rows))
    
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Plot each signal in its subplot
    for i, event_index in enumerate(event_indices):
        # Extract the signal for the current event
        event_signal = cut_signals_df.xs(event_index, level='Event')

        # Plot the signal
        axs[i].plot(event_signal['Time'], event_signal['Data'], color = 'limegreen', label=f'Event {event_index}')
        axs[i].axvline(x=0, color='red', linestyle='--')  # Event occurrence line
        axs[i].set_xlabel('Time(S) relative to event')
        axs[i].set_ylabel('z-dF/F')
        axs[i].set_ylim(y_min, y_max)
        axs[i].legend()
        

    # Turn off any unused subplots
    for i in range(num_events, len(axs)):
        axs[i].axis('off')    
    plt.suptitle('Cut Signals Around Each Event')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust the layout to make room for the suptitle
      # Show the time of the peak with a vertical line
    
    plt.show()

def heatmap_plot(df):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    # Assuming normalized_signal_df is your DataFrame loaded with data
    # Pivot the DataFrame to get 'Time' as columns, 'Event' as rows, and 'Data' as values
    ## you can use normalized or non normalized signal
    pivoted_df = df.reset_index().pivot(index='Event', columns='Time', values='Data')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("Spectral", as_cmap=True)

    sns.heatmap(pivoted_df, cmap=cmap.reversed())
    plt.title('Heatmap of Signal Data by Time and Event')
    plt.xlabel('Time(S)')
    plt.ylabel('Trial number')

    # Customize x-axis ticks
    time_labels = pivoted_df.columns.values  # Get the time values
    
    # Find the overall maximum value in the DataFrame and its location
    # important note: by changing the .loc[:, 0: 2] you can change the search aira for max value
    max_value = pivoted_df.loc[:, 0: 2].max().max()
    max_location = pivoted_df.loc[:, 0: 2].stack().idxmax()  # This gives you a tuple (Event, Time)

    # Convert Event and Time to positions on the plot
    event_pos = pivoted_df.index.get_loc(max_location[0])  # Convert Event to its index position
    time_pos = pivoted_df.columns.get_loc(max_location[1])  # Convert Time to its column position

    search_range = np.arange(time_labels.min(),time_labels.max()+1) # This will create an array of integers from -3 to 8

    # Vectorized operation to find indices
    # Note: This assumes time_labels contains unique elements and each element in search_range exists in time_labels
    indices = np.nonzero(np.in1d(time_labels, search_range))[0]

    # Ensure 0 is included in the ticks if it exists in the time labels
    zero_time_index = None
    if 0 in time_labels:
        zero_time_index = list(time_labels).index(0)

    # If zero time index is not already included, add it to the tick indices
    if zero_time_index is not None and zero_time_index not in indices:
        indices = sorted(np.append(indices, zero_time_index))
        
    # add max point to indeces
    indices = sorted(np.append(indices,list(time_labels).index(max_location[1]) ))

    ticks_to_show = indices
    plt.xticks(ticks_to_show, [time_labels[i] for i in ticks_to_show])  # Set custom ticks and labels
    
    ydata = len(pivoted_df)
    plt.yticks(np.arange(0.5, ydata+0.5, round(math.log(ydata))), np.arange(1, ydata+1, round(math.log(ydata))))

    # Add a vertical line to indicate zero time occurrence if zero time is present
    if zero_time_index is not None:
        plt.axvline(x=zero_time_index, color='red', linestyle='--', label='Zero Time')
        plt.legend()

    # Highlight the maximum value on the heatmap
    # For better visibility, you can use 'scatter' with a marker or 'annotate' to directly show the value
    plt.scatter(time_pos, event_pos+0.5, color='lime', s=50, label=f'Peak:{max_value:.2f}', edgecolor='black', zorder=5)
    plt.axvline(x=time_pos, color='lime', linestyle='--', label='Peak Time')

    plt.legend()

    plt.show()

def plot_psth_with_std_cloud(psth_df,y_min,y_max,color,signal):

    """
    Plot the Peri-Stimulus Time Histogram (PSTH) with standard deviation as a shaded cloud for each time point,
    and highlight the peak value.
    """
    import matplotlib.pyplot as plt

    # Filter the DataFrame for Time between 0 and 2 inclusive
    filtered_df = psth_df[psth_df['Time'].between(0, 2)]

    # Prepare data from the filtered DataFrame
    time_points = filtered_df['Time']
    mean_values = filtered_df['Average Signal']
    # Find the peak value and its corresponding time within the filtered range
    peak_value = mean_values.max()
    peak_time = time_points[mean_values.idxmax()]
    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']
    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']
    std_deviation = psth_df['Standard Deviation']
    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot mean values
    plt.plot(time_points, mean_values, label=f'Average {signal}', color=color)
    # Add standard deviation as a shaded area
    plt.fill_between(time_points, mean_values - std_deviation, mean_values + std_deviation, color=color, alpha=0.3)
    # Highlight the peak value
    plt.scatter(peak_time, peak_value, color=color, marker='o', label=f'Peak {peak_value:.2f}')  # Mark the peak

    # Show the time of the peak with a vertical line
    plt.axvline(x=peak_time, color='lime', linestyle='--')

    # Customize x-axis to include peak time as a tick
    current_ticks = np.array(plt.xticks()[0])
    new_ticks = np.sort(np.append(current_ticks, peak_time))
    plt.xticks(new_ticks, labels=[f'{tick:.2f}' if tick != peak_time else f'{tick:.2f}' for tick in new_ticks])
    plt.xlabel('Time')
    plt.ylabel('z-dF/F')
    plt.title('Peri-Stimulus Time Histogram (PSTH) with Standard Deviation')
    plt.axvline(x=0, color='red', linestyle='--')  # Event occurrence line
    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def plot_psth_with_min_max_range(psth_df, y_min, y_max,color,signal):
    """
    Plot the Peri-Stimulus Time Histogram (PSTH) with minimum and maximum values as a shaded range for each time point,
    and highlight the peak value. Additionally, show the time of the peak as a specific x-axis tick.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter the DataFrame for Time between 0 and 2 inclusive
    filtered_df = psth_df[psth_df['Time'].between(0, 2)]

    # Prepare data from the filtered DataFrame
    time_points = filtered_df['Time']
    mean_values = filtered_df['Average Signal']

    # Find the peak value and its corresponding time within the filtered range
    peak_value = mean_values.max()
    peak_time = time_points[mean_values.idxmax()]

    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']

    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']
    min_signal = psth_df['Minimum Signal']
    max_signal = psth_df['Maximum Signal']
    

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot mean values
    plt.plot(time_points, mean_values, label=f'Average {signal}', color=color)

    # Add minimum and maximum values as a shaded area
    plt.fill_between(time_points, min_signal, max_signal, color=color, alpha=0.3)

    # Highlight the peak value
    plt.scatter(peak_time, peak_value, color=color, marker='o', label=f'Peak {peak_value:.2f}')  # Mark the peak

    # Show the time of the peak with a vertical line
    plt.axvline(x=peak_time, color=color, linestyle='--')

    # Customize x-axis to include peak time as a tick
    current_ticks = np.array(plt.xticks()[0])
    new_ticks = np.sort(np.append(current_ticks, peak_time))
    plt.xticks(new_ticks, labels=[f'{tick:.2f}' if tick != peak_time else f'{tick:.2f}' for tick in new_ticks])

    plt.xlabel('Time')
    plt.ylabel('z-dF/F')
    plt.title('Peri-Stimulus Time Histogram (PSTH) with Min-Max Range')
    plt.axvline(x=0, color='red', linestyle='--')  # Event occurrence line

    plt.legend()
    plt.grid(True)
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    
    plt.show()

def plot_psth(psth_df, y_min, y_max,color,signal,pos):
    """
    Plot the Peri-Stimulus Time Histogram (PSTH) with minimum and maximum values as a shaded range for each time point,
    and highlight the peak value. Additionally, show the time of the peak as a specific x-axis tick.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter the DataFrame for Time between 0 and 2 inclusive
    filtered_df = psth_df[psth_df['Time'].between(0, 2)]

    # Prepare data from the filtered DataFrame
    time_points = filtered_df['Time']
    mean_values = filtered_df['Average Signal']

    # Find the peak value and its corresponding time within the filtered range
    peak_value = mean_values.max()
    peak_time = time_points[mean_values.idxmax()]

    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']
    
    # Plot mean values
    pos.plot(time_points, mean_values, label=f'{signal}', color=color)

    if 'Standard Deviation' in psth_df.columns:        
        std_deviation = psth_df['Standard Deviation']
        # Add standard deviation as a shaded area
        pos.fill_between(time_points, mean_values - std_deviation, mean_values + std_deviation, color=color, alpha=0.3)
        pos.set_title('Peri-Stimulus Time Histogram (PSTH) with SDT')
    elif 'Lower 95% CI' in psth_df.columns:        
        min_signal = psth_df['Lower 95% CI']
        max_signal = psth_df['Upper 95% CI']
        # Add minimum and maximum values as a shaded area
        pos.fill_between(time_points, min_signal, max_signal, color=color, alpha=0.3)
        pos.set_title('Peri-Stimulus Time Histogram (PSTH) with 95% CI')
    else:
        min_signal = psth_df['Minimum Signal']
        max_signal = psth_df['Maximum Signal']
        # Add minimum and maximum values as a shaded area
        pos.fill_between(time_points, min_signal, max_signal, color=color, alpha=0.3)
        pos.set_title('Peri-Stimulus Time Histogram (PSTH) with Min and Max')

    # Highlight the peak value
    pos.scatter(peak_time, peak_value, color=color, marker='o', label=f'Peak {peak_value:.2f}')  # Mark the peak

    # Show the time of the peak with a vertical line
    pos.axvline(x=peak_time, color=color, linestyle='--')
    # Customize x-axis to include peak time as a tick
    current_ticks = np.array(plt.xticks()[0])
    print(current_ticks)
    new_ticks = np.sort(np.append(current_ticks, peak_time))
    pos.set_xticks(new_ticks, labels=[f'{tick:.2f}' if tick != peak_time else f'{tick:.2f}' for tick in new_ticks])

    pos.set_xlabel('Time(S)')
    pos.set_ylabel('z-dF/F')
    pos.axvline(x=0, color='red', linestyle='--')  # Event occurrence line

    pos.legend()
    pos.grid(True)
    pos.set_ylim(y_min, y_max)

    plt.tight_layout()


def plot_psth_auc(psth_df,y_min,y_max,color,signal):
    """
    Plot the Peri-Stimulus Time Histogram (PSTH) with standard deviation as a shaded cloud for each time point.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Prepare data for plotting
    time_points = psth_df['Time']
    mean_values = psth_df['Average Signal']
    std_deviation = psth_df['Standard Deviation']

    # Plotting PSTH
    plt.plot(time_points, mean_values, label=f'Average {signal}', color=color)
    plt.fill_between(time_points[time_points < 0], mean_values[time_points < 0], color='purple', alpha=0.3)
    plt.fill_between(time_points[time_points >= 0], mean_values[time_points >= 0], color='purple', alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--')  # Event occurrence line
    plt.xlabel('Time(S)')
    plt.ylabel('z-dF/F')
    plt.title('PSTH with Standard Deviation and Highlighted Areas')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(y_min, y_max)

def plot_auc_bars(psth_df,y_min,y_max):
    """
    Plot the AUC values before and after time 0 as separate bars.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Separate the dataset into before and after time 0
    before_zero = psth_df[psth_df['Time'] < 0]
    after_zero = psth_df[psth_df['Time'] >= 0]

    # Calculate AUC using the trapezoidal rule
    auc_before_zero = np.trapz(before_zero['Average Signal'], before_zero['Time'])
    auc_after_zero = np.trapz(after_zero['Average Signal'], after_zero['Time'])

    # Plotting AUC bars
    auc_values = [auc_before_zero, auc_after_zero]
    labels = ['AUC Before event', 'AUC After event']
    colors = ['gray', 'green']

    plt.figure(figsize=(12, 6))
    plt.bar(labels, auc_values, color=colors)
    plt.title('Area Under Curve')
    plt.ylabel('Averaged Area')
    plt.tight_layout()
    plt.ylim(y_min, y_max)
    plt.show()

def plot_auc_bars_with_duration(psth_df,y_min,y_max,bd=(-1,0),ad=(0,3)):
    """
    Plot the AUC values before and after time 0 as separate bars.
    """
    import matplotlib.pyplot as plt
    import numpy as np


    # Separate the dataset into before duration and after duration time 

    start, end = bd
    before_zero = psth_df[psth_df['Time'].between(start, end)]
    start, end = ad
    after_zero = psth_df[psth_df['Time'].between(start, end)]

    # Calculate AUC using the trapezoidal rule
    auc_before_zero = np.trapz(before_zero['Average Signal'], before_zero['Time'])
    auc_after_zero = np.trapz(after_zero['Average Signal'], after_zero['Time'])

    # Plotting AUC bars
    auc_values = [auc_before_zero, auc_after_zero]
    labels = ['AUC Before event', 'AUC After event']
    colors = ['gray', 'green']

    plt.figure(figsize=(12, 6))
    plt.bar(labels, auc_values, color=colors)
    plt.title('Area Under Curve')
    plt.ylabel('Averaged Area')
    plt.tight_layout()
    plt.ylim(y_min, y_max)
    plt.show()


import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class PhotometryAnalysis:
    def __init__(self,isos_df,grabda_df,save_callback=None):
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        from importlib import reload
        import photometry_functions as pf

        self.isos_df = isos_df
        self.grabda_df = grabda_df
        self.save_callback = save_callback  # Callback function to handle the saved DataFrame
        self.zdFF = np.array([])
        self.signal_df = pd.DataFrame()
        self.setup_widgets()
        self.display_widgets()

    def check_install(self, package_name):
        try:
            __import__(package_name)
            print(f"{package_name} is installed.")
        except ImportError:
            print(f"{package_name} is not installed. Please run `!pip install {package_name}`.")

    def setup_widgets(self):
        self.cutoff_freq_widget = widgets.IntText(value=10, description='Cutoff Freq:')
        self.remove_widget = widgets.IntText(value=0, description='remove(ms):')
        self.lambd_widget = widgets.FloatLogSlider(value=5e11, base=10, min=5, max=16, step=0.1, description='lambd:')
        self.porder_widget = widgets.IntText(value=10, description='porder:')
        self.itermax_widget = widgets.IntText(value=50, description='itermax:')
        self.button = widgets.Button(description='Run Function')
        self.button.on_click(self.on_button_clicked)
        self.plot_output = widgets.Output()

    def on_button_clicked(self, b):
        ref_df = self.isos_df  # Need to define or pass `isos_df` when initializing class
        sig_df = self.grabda_df  # Need to define or pass `grabda_df` when initializing class
        sample_rate = calculate_sample_rate(sig_df)
        cutoff_freq = self.cutoff_freq_widget.value
        window_len = calculate_window_len(cutoff_freq, sample_rate)
        remove = int((self.remove_widget.value) * (sample_rate / 1000))
        self.zdFF = get_zdFF(ref_df.Data, sig_df.Data, remove=remove, smooth_win=window_len,
                                lambd=self.lambd_widget.value, porder=self.porder_widget.value,
                                itermax=self.itermax_widget.value)
        self.plot_and_save(ref_df[remove:], sig_df[remove:], self.zdFF)

    def min_max_normalize(self, df):
        return 2 * ((df - df.min()) / (df.max() - df.min())) - 1

    def plot_and_save(self, isos_df, grabda_df, zdFF):
        with self.plot_output:
            clear_output(wait=True)
            # here we are doing the normalization
            isos_normalized = self.min_max_normalize(isos_df['Data'])
            grabda_normalized = self.min_max_normalize(grabda_df['Data'])
            event_normalized = self.min_max_normalize(zdFF)
            fig, ax = plt.subplots(figsize=(20, 4))
            ax.plot(isos_df['Time'], isos_normalized, 'silver', alpha = 0.3,  label='Isos')
            ax.plot(grabda_df['Time'], grabda_normalized, 'black', alpha = 0.3, label='GrabDA')
            ax.plot(isos_df['Time'], event_normalized, 'limegreen', label='corrected DA signal')
            ax.set_title('Normalized Signals (-1 to 1)')
            ax.set_xlabel('Time')
            ax.set_ylabel('z-dF/F')
            ax.legend()
            plt.tight_layout()
            plt.show()
            self.signal_df = pd.DataFrame({'Time': isos_df.Time, 'Data': zdFF})
            self.save_selected_data()

    def display_widgets(self):
        display(self.cutoff_freq_widget, self.remove_widget, self.lambd_widget,
                self.porder_widget, self.itermax_widget, self.button, self.plot_output)
    def save_selected_data(self):
        if self.save_callback is not None:
            self.save_callback(self.signal_df)
class SignalExplorer:
    def __init__(self, centralized_signals_df,event_type, save_callback=None):
        
        self.centralized_signals_df = centralized_signals_df
        self.filtered_signal_df = None
        self.save_callback = save_callback  # Callback function to handle the saved DataFrame
        self.event_type = event_type
        self.checkboxes = {}
        self.plot_output = widgets.Output()
        self.init_widgets()
        self.display_widgets()

    def save_selected_data(self):
        selected_events = [event for event, cb in self.checkboxes.items() if cb.value]
        self.filtered_signal_df = self.centralized_signals_df[self.centralized_signals_df.index.get_level_values('Event').isin(selected_events)]
        print("Filtered data is ready.")
        # If a callback is provided, call it with the filtered DataFrame
        if self.save_callback is not None:
            self.save_callback(self.filtered_signal_df,self.event_type)
    
    def plot_selected_signals(self, selected_events):
        with self.plot_output:
            clear_output(wait=True)
            plt.figure(figsize=(12, 6))
            for event_index in selected_events:
                event_signal = self.centralized_signals_df.xs(event_index, level='Event')
                plt.plot(event_signal['Time'], event_signal['Data'], label=f'Event {event_index}')
            plt.xlabel('Time (relative to event)')
            plt.ylabel('Signal')
            plt.title('Selected Cut Signals')
            plt.legend()
            plt.show()
    
    def on_button_clicked(self, b):
        selected_events = [event for event, cb in self.checkboxes.items() if cb.value]
        self.plot_selected_signals(selected_events)
        self.save_selected_data()
    
    def select_all_handler(self, change):
        for cb in self.checkboxes.values():
            cb.value = change.new
    
    def init_widgets(self):
        event_indices = self.centralized_signals_df.index.get_level_values('Event').unique()
        self.checkboxes = {event: widgets.Checkbox(value=False, description=f'Event {event}') for event in event_indices}
        select_all_checkbox = widgets.Checkbox(description="Select All", value=False)
        select_all_checkbox.observe(self.select_all_handler, names='value')
        
        n_cols = 6
        n_rows = -(-len(self.checkboxes) // n_cols)
        self.grid_layout = widgets.GridspecLayout(n_rows + 1, n_cols)
        self.grid_layout[0, :] = select_all_checkbox
        for i, event in enumerate(self.checkboxes, start=n_cols):
            self.grid_layout[i // n_cols, i % n_cols] = self.checkboxes[event]
        self.refresh_button = widgets.Button(description="Apply Selection")
        self.refresh_button.on_click(self.on_button_clicked)
    
    def display_widgets(self):
        display(self.grid_layout)
        # display(self.save_button)
        display(self.refresh_button)
        display(self.plot_output)


def find_nearest(array, value):
    """
    Find the nearest value in an array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def cut_and_center_signals(signal_df, event_df, time_window, event_type="All"):
    """
    Cut the signal around each event time of a specific type, center the time around the event, 
    and store them separately.
    """
    cut_signals = []

    # Filter the events if a specific type is chosen
    if event_type is not None:
        if event_type != "All":
            event_df = event_df[event_df['Type'] == event_type]

    for _, event in event_df.iterrows():
        nearest_event_time = find_nearest(signal_df['Time'], event['Time'])
        window_start = nearest_event_time + time_window[0]
        window_end = nearest_event_time + time_window[1]
        # Extract signal within the time window
        window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
        windowed_signal = signal_df[window_mask].copy()
        # Center the time around the nearest event time
        windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

        cut_signals.append(windowed_signal)

    # Combine all cut signals into a single DataFrame
    cut_signals_df = pd.concat(cut_signals, keys=range(1,len(cut_signals)+1), names=['Event', 'Row'])

    return cut_signals_df

def cut_and_center_signals2(signal_df, event_df, time_window, event_type="All"):
    """
    Cut the signal around each event time of a specific type, center the time around the event, 
    and store them separately.
    """
    cut_signals = []

    # Filter the events if a specific type is chosen
    if event_type is not None:
        if event_type == "All":
            event_df = event_df[event_df['Type'] == event_type]
            for _, event in event_df.iterrows():
                nearest_event_time = find_nearest(signal_df['Time'], event['Time'])
                window_start = nearest_event_time + time_window[0]
                window_end = nearest_event_time + time_window[1]
                # Extract signal within the time window
                window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
                windowed_signal = signal_df[window_mask].copy()
                # Center the time around the nearest event time
                windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

                cut_signals.append(windowed_signal)

            # Combine all cut signals into a single DataFrame
            cut_signals_df = pd.concat(cut_signals, keys=range(1,len(cut_signals)+1), names=['Event', 'Row'])

        if event_type == "Init":
            event_df = event_df[event_df['Type'] == event_type]
            for _, event in event_df.iterrows():
                nearest_event_time = find_nearest(signal_df['Time'], event['Time'])
                window_start = nearest_event_time + time_window[0]
                window_end = nearest_event_time + time_window[1]
                # Extract signal within the time window
                window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
                windowed_signal = signal_df[window_mask].copy()
                # Center the time around the nearest event time
                windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

                cut_signals.append(windowed_signal)

            # Combine all cut signals into a single DataFrame
            cut_signals_df = pd.concat(cut_signals, keys=range(1,len(cut_signals)+1), names=['Event', 'Row'])
            
        if event_type == "Fail":
            for _, event in event_df.iterrows():
                nearest_event_time = find_nearest(signal_df['Time'], event['Time'])
                window_start = nearest_event_time + time_window[0]
                window_end = nearest_event_time + time_window[1]
                # Extract signal within the time window
                window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
                windowed_signal = signal_df[window_mask].copy()
                # Center the time around the nearest event time
                windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

                cut_signals.append(windowed_signal)

            # Combine all cut signals into a single DataFrame
            cut_signals_df = pd.concat(cut_signals, keys=range(1,len(cut_signals)+1), names=['Event', 'Row'])

        if event_type == "Success":

            for _, event in event_df.iterrows():
                nearest_event_time = find_nearest(signal_df['Time'], event['Time'])
                window_start = nearest_event_time + time_window[0]
                window_end = nearest_event_time + time_window[1]
                # Extract signal within the time window
                window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
                windowed_signal = signal_df[window_mask].copy()
                # Center the time around the nearest event time
                windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

                cut_signals.append(windowed_signal)

            # Combine all cut signals into a single DataFrame
            cut_signals_df = pd.concat(cut_signals, keys=range(1,len(cut_signals)+1), names=['Event', 'Row'])

    return cut_signals_df

def cut_and_center_signals_modified(signal_df, event_df, time_window, event_type="All"):
    """
    Cut the signal around each event time of a specific type, center the time around the event, 
    and store them separately. For 'Fail' or 'Success' types, use the 'Init' time before them.
    """
    cut_signals = []

    # Create a copy to avoid modifying the original DataFrame
    event_df = event_df.copy()

    # Add a column to track the reference time for cutting signals
    event_df['Ref_Time'] = event_df['Time']

    if event_type != "All":
        # If looking for Fail or Success, adjust to use Init time
        if event_type in ["Fail", "Success"]:
            # Iterate over events to find and assign the preceding Init time
            for i, row in event_df.iterrows():
                if row['Type'] in ["Fail", "Success"]:
                    # Find the preceding Init event
                    preceding_init = event_df[(event_df['Time'] < row['Time']) & (event_df['Type'] == "Init")].max()
                    if not preceding_init.empty:
                        event_df.at[i, 'Ref_Time'] = preceding_init['Time']

        event_df = event_df[event_df['Type'] == event_type]

    for _, event in event_df.iterrows():
        nearest_event_time = find_nearest(signal_df['Time'], event['Ref_Time'])
        window_start = nearest_event_time + time_window[0]
        window_end = nearest_event_time + time_window[1]
        # Extract signal within the time window
        window_mask = (signal_df['Time'] >= window_start) & (signal_df['Time'] <= window_end)
        windowed_signal = signal_df[window_mask].copy()
        # Center the time around the nearest event time
        windowed_signal['Time'] = (windowed_signal['Time'] - nearest_event_time).round(5)

        cut_signals.append(windowed_signal)

    # Combine all cut signals into a single DataFrame
    cut_signals_df = pd.concat(cut_signals, keys=range(1, len(cut_signals)+1), names=['Event', 'Row'])

    return cut_signals_df