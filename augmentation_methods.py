import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import random
#import smogn
import pickle as pkl
from matplotlib import rcParams

#np.random.seed(42)

def window_warp_old(signal, modify_type, warps) -> np.array: #
    #print(f"windowwarping")
    new_signal = []
    if signal.ndim > 1:
        for j in range(warps):
            index = np.random.randint(0, int(len(signal[0]) * 0.9))
            for i in range(len(signal)):

                # Choose a random point in the first 90% of the array
                #index = np.random.randint(0, int(len(signal[i]) * 0.9))

                # Define the next 10% of the array after the index
                modify_area = signal[i][index: int(index + len(signal) * 0.1)]

                # Depending on the modify_type, either remove every second data point or double every data point
                if modify_type == "shorten":
                    modify_area = np.delete(modify_area, np.arange(0, len(modify_area), 2))
                elif modify_type == "elongate":
                    print(f"ww 2d elongating")
                    modify_area = np.repeat(modify_area, 2)
                elif modify_type == "random":
                    random_number = random.random()
                    if random_number < 0.5:
                        modify_area = np.delete(modify_area, np.arange(0, len(modify_area), 2))
                    else:
                        modify_area = np.repeat(modify_area, 2)
                    

                # Replace the segment of the signal with the modified version
                #plt.plot(signal[i], label = "original signal")
                signal[i] = np.concatenate((signal[i][:index], modify_area, signal[i][int(index + len(signal) * 0.1):]))
                #new_signal = np.concatenate((signal[i][:index], modify_area, signal[i][int(index + len(signal) * 0.1):]))
                #plt.plot(new_signal, label = "modified signal")
                #plt.legend()
                #plt.show()
                #a = 1/0

        return signal
    else: #TODO: Can never happen, since data and target are ALLWAYS used
        print(f"WINDOW WARP IS RUN WITH ONLY A SINGLE ARRAY. THIS MESSAGE SHOULD NEVER BE SEEN")
        # Choose a random point in the first 90% of the array
        index = np.random.randint(0, int(len(signal) * 0.9))

        # Define the next 10% of the array after the index
        modify_area = signal[index: int(index + len(signal) * 0.1)]

        # Depending on the modify_type, either remove every second data point or double every data point
        if modify_type == "shorten":
            modify_area = np.delete(modify_area, np.arange(0, len(modify_area), 2))
        elif modify_type == "elongate":
            print(f"elongating")
            modify_area = np.repeat(modify_area, 2)

        # Replace the segment of the signal with the modified version
        plt.plot(signal, label = "original signal")
        new_signal = np.concatenate((signal[:index], modify_area, signal[int(index + len(signal) * 0.1):]))
        plt.plot(new_signal, label = "modified signal")
        plt.legend()
        plt.show()
        a = 1/0
        return new_signal#, index, len(modify_area)    

def single_window_warp(time_series, window_fraction=0.1):
    m, n = time_series.shape
    warped_series = np.empty((m, 0))  # Initialize an empty array to concatenate new values

    window_size = int(n * window_fraction)
    start_idx = np.random.randint(0, n - window_size)

    action = np.random.choice(['stretch', 'compress'])

    # Add the part before the window to the warped_series
    warped_series = np.hstack((warped_series, time_series[:, :start_idx]))

    if action == 'stretch':
        stretch_window = np.zeros((m, 2 * window_size))
        for i in range(m):
            stretch_window[i] = np.interp(
                np.linspace(start_idx, start_idx + window_size, 2 * window_size),
                np.arange(start_idx, start_idx + window_size),
                time_series[i, start_idx:start_idx + window_size]
            )
        warped_series = np.hstack((warped_series, stretch_window))
        
    elif action == 'compress':
        compress_window = time_series[:, start_idx:start_idx + window_size][:, ::20] 
        warped_series = np.hstack((warped_series, compress_window))

    # Add the part after the window to the warped_series
    end_idx = start_idx + window_size
    if end_idx < n:
        warped_series = np.hstack((warped_series, time_series[:, end_idx:]))

    return warped_series

def window_warp(time_series, num_parts, window_fraction=0.1):
    #num_parts = 2 #TODO: Change back
    m, n = time_series.shape
    part_size = n // num_parts
    final_series = np.empty((m, 0))

    #plt.rc('font', family='serif')
    #rcParams.update({'font.size': 14})
    #fig = plt.figure(figsize=(10, 5))
    #
    #plt.rc('font', family='serif')
    #rcParams.update({'font.size': 14})
    #ax = fig.add_subplot(111)
    #axes[0].plot(t,spline_signal, color = "#1F77B4")
    #axes[0].set_title("Spline Polynomial")
    #axes[0].set_xlabel("Time (s)", fontsize = 16)
    #axes[0].set_ylabel("Value", fontsize = 16)
    #axes[0].set_yticks(np.arange(-4, 4 + 1, 2))

    for i in range(num_parts):
        start = i * part_size
        end = start + part_size
        segment = time_series[:, start:end]
        warped_segment = single_window_warp(segment, window_fraction)
        final_series = np.hstack((final_series, warped_segment))

    # Append remaining elements if the array size is not a multiple of num_parts
    if n % num_parts != 0:
        final_series = np.hstack((final_series, time_series[:, num_parts * part_size:]))

    #plot_normal = np.copy(time_series[0])
    #plot_augment = np.copy(final_series[0])
    #if len(plot_normal) > len(plot_augment):
    #    maxlen = len(plot_augment)
    #    plot_augment = np.pad(plot_augment, (0, len(plot_normal) - len(plot_augment)), 'constant')
    #else:
    #    maxlen = len(plot_normal)
    #    plot_normal = np.pad(plot_normal, (0, len(plot_augment) - len(plot_normal)), 'constant')
        

    #t = np.arange(0, maxlen) / 50.0
    #ax.plot(t, plot_normal[:maxlen],label = "original signal", color = "#1F77B4")
    #ax.plot(t, plot_augment[:maxlen],label = "window warped signal", color = "#FF7F0E")
#
    ##axes[1].plot(t,signal[i], label = "magnitude warped signal", color = "#FF7F0E")
    ##axes[1].plot(t,base_signal_forplot, label = "original signal", color = "#2CA02C")
    ##axes[1].set_title("Signal")
    #ax.set_xlabel("Time (s)", fontsize = 16)
    #ax.set_ylabel("Value", fontsize = 16)
    #plt.legend()
    ##plt.tight_layout(pad=1.0)
    #savepath = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Auswertung\plots\bilder_fuer_arbeit\WW_explaination.png"
    #plt.savefig(savepath, dpi = 300)
    #plt.show()
    #a = 1/0
    return final_series

def add_noise(signal, magnitude) -> np.array: #DONE
    if signal.ndim > 1:
        for i in range(len(signal)):
            noise = np.random.normal(0, magnitude, len(signal[0]))
            #plt.plot(noise, label = "noise")
            #plt.legend()
            #plt.show()
            #
            #original_signal = np.copy(signal[i])
            signal[i] = signal[i] + noise
            #plt.plot(signal[i], label = "augmented signal")
            #plt.plot(original_signal, label = "original signal")
            #plt.legend()
            #plt.show()
            #a = 1/0
    else:
        noise = np.random.normal(0, magnitude, len(signal))
        plt.plot(signal, label = "noise")
        plt.legend()
        plt.show()
        plt.plot(signal, label = "original signal")
        signal = signal + noise
        plt.plot(signal, label = "augmented signal")
        plt.legend()
        plt.show()
        a = 1/0
    return signal

def flipping(signal) -> np.array:
    base_signal = np.copy(signal)
    max_value = np.max(base_signal)
    min_value = np.min(base_signal)
    mid_value = min_value+ (max_value-min_value)/2
    for i in range(len(base_signal)):
        base_signal[i] = mid_value - (base_signal[i] - mid_value)
    return base_signal

def scale_by_value(signal, scaleing_value):
    return signal * scaleing_value

def fourier_transform(signal):
    fourier_transform = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))

def generate_spline(signal, knots, mu, sigma, min_dist):
    '''generates a spline interpolation with "knots" knots (plus one at the start and end).
    The value and location of the knots are randomly chosen using mu and sigma (using normal dist.) 
    All knots need to be min_dist (0 to 1, in relation to len(signal)) from each other.
    Lenght is the same as signal'''
    #base_signal = signal#np.copy(signal) ###THIS CREATES ERROR!!!
    base_signal = np.copy(signal) #IMPORTANT TO COPY!
    #return base_signal
    array_length = len(base_signal)
    min_distance = int(min_dist * array_length)

    # Choose four random points with minimum distance constraint
    random_indices = []
    while len(random_indices) < knots:
        index = np.random.randint(min_distance, array_length - min_distance)
        if all(abs(index - existing_index) > min_distance for existing_index in random_indices):
            random_indices.append(index)

    # Set the selected points to random values with mu and sigma
    base_signal[random_indices] = np.random.normal(loc=mu, scale=sigma, size=knots)

    # Set the first and last points to mu
    base_signal[0] = base_signal[-1] = mu

    # Get indices and values of the known points
    known_indices = np.concatenate(([0], random_indices, [array_length - 1]))
    known_values = base_signal[known_indices]

    # Create an interpolation function
    interp_func = interp1d(known_indices, known_values, kind='cubic', fill_value='extrapolate')

    # Interpolate the remaining points in the array
    interpolated_values = interp_func(np.arange(array_length))
    #plt.plot(interpolated_values)
    #plt.show()
    return interpolated_values

def generate_resampled_spline(signal, knots, mu, sigma, min_dist, spline_length=400):
    ''' Generates and resamples a spline interpolation to match the length of the signal.
    The value and location of the knots are randomly chosen using mu and sigma (using normal dist.) 
    All knots need to be min_dist (0 to 1, in relation to spline_length) from each other.
    The generated spline is then resampled to match signal_length.
    spline_length=40000 since this is about the length that was used for tests'''

    signal_length = len(signal)
    base_spline = np.zeros(spline_length)
    min_distance = int(min_dist * spline_length)

    # Choose knots with minimum distance constraint
    random_indices = []
    while len(random_indices) < knots:
        index = np.random.randint(min_distance, spline_length - min_distance)
        if all(abs(index - existing_index) > min_distance for existing_index in random_indices):
            random_indices.append(index)

    # Set the selected points to random values with mu and sigma
    base_spline[random_indices] = np.random.normal(loc=mu, scale=sigma, size=knots)
    base_spline[0] = base_spline[-1] = mu

    # Known points for interpolation
    known_indices = np.concatenate(([0], random_indices, [spline_length - 1]))
    known_values = base_spline[known_indices]

    # Create spline interpolation function
    spline_func = interp1d(known_indices, known_values, kind='cubic', fill_value='extrapolate')

    # Interpolate the spline across its entire length
    interpolated_spline = spline_func(np.arange(spline_length))

    # Resample the spline to match the signal length
    resample_func = interp1d(np.linspace(0, 1, spline_length), interpolated_spline, kind='linear')
    resampled_spline = resample_func(np.linspace(0, 1, signal_length))

    #print(resampled_spline.shape)
    #plt.plot(resampled_spline)
    #plt.show()
    return resampled_spline

#Magnitude Warping

def magnitude_warping(signal, knots, sigma, mu = 0.2, min_dist = 0.1):

    if signal.ndim > 1:
        #spline_signal = generate_spline(signal[0], knots=knots, mu=mu, sigma=sigma, min_dist=min_dist)
        spline_signal = generate_resampled_spline(signal[0], knots=knots, mu=mu, sigma=sigma, min_dist=min_dist)

        #plt.rc('font', family='serif')
        #rcParams.update({'font.size': 14})
        #fig, axes = plt.subplots(2, 1, figsize=(10, 6),gridspec_kw={'height_ratios': [1, 2]})
        #t = np.arange(0, len(signal[0])) / 50.0
        #plt.rc('font', family='serif')
        #rcParams.update({'font.size': 14})
        #axes[0].plot(t,spline_signal, color = "#1F77B4")
        #axes[0].set_title("Spline Polynomial")
        #axes[0].set_xlabel("Time (s)", fontsize = 16)
        #axes[0].set_ylabel("Value", fontsize = 16)
        #axes[0].set_yticks(np.arange(-4, 4 + 1, 2))

        for i in range(len(signal)):
            base_signal = np.copy(signal[i])
            base_signal_forplot = np.copy(signal[i])
            base_signal_avg = np.average(base_signal)
            base_signal = base_signal - base_signal_avg
            
            mag_warped_signal = np.multiply(base_signal, spline_signal)
            signal[i] = mag_warped_signal + base_signal_avg
            #axes[1].plot(t,signal[i], label = "magnitude warped signal", color = "#FF7F0E")
            #axes[1].plot(t,base_signal_forplot, label = "original signal", color = "#2CA02C")
            #axes[1].set_title("Signal")
            #axes[1].set_xlabel("Time (s)", fontsize = 16)
            #axes[1].set_ylabel("Value", fontsize = 16)
            #plt.legend()
            #plt.tight_layout(pad=1.0)
            #savepath = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Auswertung\plots\bilder_fuer_arbeit\MagWarp.png"
            #plt.savefig(savepath, dpi = 300)
            #plt.show()
            #a = 1/0
        return signal
    else: #TODO: Can never happen, since data and target are ALLWAYS used
        print(f"magnitude warping IS RUN WITH ONLY A SINGLE ARRAY. THIS MESSAGE SHOULD NEVER BE SEEN")
    

def homogenous_scaling(signal, sigma):
    base_signal = np.copy(signal)
    base_signal_avg = np.average(base_signal)
    base_signal = base_signal - base_signal_avg
    scaled_signal = base_signal * np.random.normal(loc=1, scale=sigma, size=1)
    #scaled_signal = np.multiply(base_signal, multiplier_signal)
    scaled_signal = scaled_signal + base_signal_avg
    return scaled_signal

def split_into_five_pieces(signal): #TODO: Make one function
    base_signal = np.copy(signal)

    # Calculate the length of each piece
    length = len(base_signal) // 5
    
    # Split the array into five pieces
    pieces = []
    for i in range(5):
        start = i * length
        end = (i + 1) * length
        pieces.append(base_signal[start:end])
    
    return pieces

def merge_pieces(pieces):
    # Randomly shuffle the order of the pieces
    np.random.shuffle(pieces)
    
    # Merge the pieces
    merged_array = pieces[0]
    for i in range(1, len(pieces)):
        prev_end_index = len(merged_array) - 1
        next_start_index = 0
        
        # Calculate the offset to match start and end values
        offset = merged_array[prev_end_index] - pieces[i][next_start_index]
        
        # Adjust the next piece by the offset
        pieces[i] += offset
        
        # Merge the adjusted piece with the merged array
        merged_array = np.concatenate((merged_array, pieces[i]))
    
    return merged_array

def permutation(signal):
    return(merge_pieces(split_into_five_pieces(signal)))

def pad_with_last_value(array1, array2):
    pad_length = len(array2) - len(array1)
    if pad_length > 0:
        last_value = array1[-1]
        pad_array = np.full(pad_length, last_value)
        return np.concatenate([array1, pad_array])
    else:
        return array1
    
def time_warping(signal, knots, sigma, mu = 0.2, min_dist = 0.1):
    mu = 0
    #bei spline integrieren, wenn integral der Differenz zu 1:
    # -1: löschen des nächsten Datenpunktes
    # +1: entweder verdopplung des nächsten datenpunktes, oder besser: einfügen eines Datenpunktes zwischen nächsten und übernächsten. Zunächst mit linearer Interpolation
    warped_signal = []
    #spline_signal = generate_spline(signal[0], knots=knots, mu=mu, sigma=sigma, min_dist=min_dist)
    spline_signal = generate_resampled_spline(signal[0], knots=knots, mu=mu, sigma=sigma, min_dist=min_dist)

    #plt.rc('font', family='serif')
    #rcParams.update({'font.size': 14})
    #fig, axes = plt.subplots(2, 1, figsize=(10, 6),gridspec_kw={'height_ratios': [1, 2]})
    #t = np.arange(0, len(signal[0])) / 50.0
    #plt.rc('font', family='serif')
    #rcParams.update({'font.size': 14})
    #axes[0].plot(t,spline_signal, color = "#1F77B4")
    #axes[0].set_title("Spline Polynomial")
    #axes[0].set_xlabel("Time (s)", fontsize = 16)
    #axes[0].set_ylabel("Value", fontsize = 16)
    #axes[0].set_yticks(np.arange(-1, 2, 1))


    #plt.plot(spline_signal)
    #plt.show()
    for channel in signal:
        this_channel = np.copy(channel)
        base_signal_forplot = np.copy(channel)
        integral = 0
        signal_pos = 0
        for i in range(len(spline_signal)-1):
            integral += (spline_signal[i])
            #base_signal[i] = integral
            if integral > 1:
                #add data
                value_to_add = (this_channel[signal_pos]+this_channel[signal_pos+1])/2
                this_channel = np.insert(this_channel,signal_pos, value_to_add)
                signal_pos += 1
                integral -= 1

            if integral < -1:
                #remove data

                integral += 1
                this_channel = np.delete(this_channel, signal_pos)
                signal_pos -= 1
            signal_pos += 1

        plot_normal = np.copy(base_signal_forplot)
        plot_augment = np.copy(this_channel)
        if len(plot_normal) > len(plot_augment):
            
            #plot_augment = np.pad(plot_augment, (0, len(plot_normal) - len(plot_augment)), 'constant')
            plot_augment = pad_with_last_value(plot_augment,plot_normal)
            maxlen = len(plot_normal)
        else:

            #plot_normal = np.pad(plot_normal, (0, len(plot_augment) - len(plot_normal)), 'constant')
            plot_normal = pad_with_last_value(plot_normal, plot_augment)
            maxlen = len(plot_augment)
        #t = np.arange(0, maxlen) / 50.0
        #axes[1].plot(t, plot_normal[:maxlen],label = "original signal", color = "#1F77B4")
        #axes[1].plot(t, plot_augment[:maxlen],label = "time warped signal", color = "#FF7F0E")
        #axes[1].set_xlabel("Time (s)", fontsize = 16)
        #axes[1].set_ylabel("Value", fontsize = 16)
        #axes[1].set_title("Signal")
        #plt.legend()
        #plt.tight_layout(pad=1.0)
        #savepath = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Auswertung\plots\bilder_fuer_arbeit\TW_explaination.png"
        #plt.savefig(savepath, dpi = 300)
        #plt.show()
        #a = 1/0
        ##plt.plot(this_channel, label = "time warped signal")
        ##plt.plot(this_channel_for_plot, label = "original signal")
        ##plt.legend()
        ##plt.show()
        ##a = 1/0
        warped_signal.append(this_channel)
    
    return np.array(warped_signal)

def window_slicing(signal, window_length):
    base_signal = np.copy(signal)
    signal_length = len(base_signal)
    sliced_length = int(len(base_signal)*window_length-1)
    start_area = int((1-window_length)*signal_length)
    start_element = random.randrange(0,start_area)
    sliced_window = base_signal[start_element: sliced_length+start_element]
    #interpolate to same size
    return sliced_window

def TimeVAEgenerated(interpol_type):
    #print(f"TimeVAEgenerated with type {interpol_type}")
    if interpol_type == 1:#'linear':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_linear.pkl"
    elif interpol_type == 2:#'cubic':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_cubic.pkl"
    elif interpol_type == 3:#'fft':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_fft.pkl"
    elif interpol_type == 4:#'nearest':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_nearest.pkl"
    elif interpol_type == 5:#'polynomial':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_polynomial.pkl"
    elif interpol_type == 6:#'spline':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10_upsampled_10_interpolationtype_spline.pkl"
    elif interpol_type == 7:#'10_parts':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_100_samples_merged_from_10_parts.pkl"
    elif interpol_type == 8:#'downsampled':
        data_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE\CMX_Alu_Tr_Air_TimeVAEgenerated_downsampled_10.pkl"
    
    with open(data_path, 'rb') as file:
        data =  pkl.load(file)
    
    random_idx = np.random.randint(data.shape[0])
    selected_sample = data[random_idx]
    selected_sample = np.roll(selected_sample, shift=-1, axis=1)
    selected_sample_reshaped = selected_sample.T
    #print(f"TimeVAEgenerated want to return {selected_sample_reshaped.shape}")
    #i want (5, 48373) as output
    return selected_sample_reshaped
    
def YData_generated():
    paths = [r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_ydata\generated_from_CMX_Alu_Tr_Air_1_all_measured_25_samples.csv",
            r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_ydata\generated_from_CMX_Alu_Tr_Air_2_all_measured_25_samples.csv",
            r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_ydata\generated_from_CMX_Alu_Tr_Air_3_all_measured_25_samples.csv"]
    randpath = random.randint(0, 2)
    randentity = random.randint(0, 24)
    df = pd.read_csv(paths[randpath])
    df = df[df["entity"] == f"entity_{randentity}"]
    df = df.drop(columns=["entity", "Number"])
    #print(df)
    data_array = df.values.T
    
    #print(data_array.shape)
    #a = 1/0
    return data_array


def rotation(signal):
    '''Rotates the signal, so taht the first element is the last and the other way around'''
    rotated_signal = signal[:, ::-1]
    return rotated_signal


#def smogn_augemnt(df_signals, target: str):
#    '''Implementation of smogn from nickkunz's library https://github.com/nickkunz/smogn'''
#    #TODO: Put data in df
#
#    augmented_data = smogn.smoter(data = df_signals, y = target)
#
#    #TODO: move data from df back to arrays
#    return augmented_data
#    noop = 1

