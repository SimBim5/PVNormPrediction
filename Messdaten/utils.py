import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression



def load_data(data, start, end):
    """
    Load the Data from an ascii file for Zyklische Daten
    
    Arguments:      data        Name of Grid you want to load the Data ; e.g. Unteregg
                    start       Start of Time Intervall ; e.g. 09:45:40.000
                    end         End of Time Intervall ; e.g. 09:46:50.000
    
    Returns:        power       Numpy Power Array with all Power Values from the ascii file within the Time Intervall
                    time        Numpy Time Array with all Time Values from the ascii file within the Time Intervall
                    frequency   Numpy Frequency Array with all frequency Values from the ascii file within the Time Intervall
    """
    powerarray, timearray, frequncyarray = ([] for i in range(3))
    
    path = 'Messdaten/ZyklischeDaten/' + data + '_Zyklischedaten.asc'
    
    with open(path) as f:
        lines = f.readlines()[11:]
    
    for l in lines:
        parts = l.split() 
        if len(parts) > 1:
            timepart = parts[1]
            timearray = np.append(timearray, timepart)
            
            powerpart = parts[8]
            powerarray = np.append(powerarray, float(powerpart)/1000)

            frequencypart = parts[10]
            frequncyarray = np.append(frequncyarray, frequencypart)
    
    start = np.where(timearray == start)[0][0]
    end = np.where(timearray == end)[0][0]

    frequncyarray = frequncyarray[start:end]
    frequency = np.array(frequncyarray, dtype=float)
    
    powerarray = powerarray[start:end]
    power = np.array(powerarray, dtype=float)
    
    time = timearray[start:end]

    return power, time, frequency

def load_data_10ms(filename):
    """
    Load the Data from an ascii file for 10msRMS
    
    Arguments:      filename    Name of Grid you want to load the Data ; e.g. Unteregg
    
    Returns:        power       Numpy Power Array with all Power Values from the ascii file
                    time        Numpy Time Array with all Time Values from the ascii file
                    frequency   Numpy Frequency Array with all frequency Values from the ascii file
    """
    power, time, frequency = ([] for i in range(3))
    
    path = 'Messdaten/10msRMS/' + filename
    
    with open(path) as f:
        lines = f.readlines()[11:]

    for l in lines:
        parts = l.split()
        if len(parts) > 1:
            timepart = parts[1]
            time = np.append(time, timepart)
                
            powerpart = float(parts[13])
            power = np.append(power, float(powerpart)/1000)

            frequencypart = float(parts[16])
            frequency = np.append(frequency, frequencypart)
        
    return power, time, frequency


def time_in_sec(time):
    """
    Converts the Time array in seconds
    
    Arguments:      time        Numpy Time Array with all Time Values
    
    Returns:        seconds     Numpy Time Array with all Time Values in seconds
    """
    seconds = []
    for i in time:
        ms = float(i[-3:])
        s = float(i[6:-4])
        m = float(i[3:-7])
        h = float(i[:-10])
        s_gesamt = ms/1000 + s + m *60 + h*60*60
        seconds = np.append(seconds, float(s_gesamt))
        seconds = seconds.reshape(-1, 1)
    return seconds

def lin_regression(seconds, frequency):
    """
    Converts the Time array in seconds
    
    Arguments:      seconds             Numpy Time Array with all Time Values in seconds
                    frequency           Numpy Frequency Array with all frequency Values
    
    Returns:        frequency_lin       Numpy Frequency Array with linear Frequency Values 
                    reg                 Fitted Estimator outputted from LinearRegression().fit()
    """
    
    reg = LinearRegression().fit(seconds, frequency)
    reg.score(seconds, frequency)
    t = (min(seconds), max(seconds))
    a, b = reg.predict(t)
    
    frequency_lin = []
    for h in range(1, frequency.size+1):
        frequency_lin = np.append(frequency_lin, a + (b - a)/(frequency.size)*h)
        
    return frequency_lin, reg

def normalize(power):
    """
    Normalises the power array ; maps power array to (1 : 0)
    
    Arguments:      power       Numpy Power Array with all Power Values
    
    Returns:        power       Normalised Power Array with maximum value = 1 and minimum value = 0
    """
    
    if max(power) > 0: 
        power = power - max(power)
    power = power * -1
    power = (power-np.min(power))/(np.max(power)-np.min(power))        
    return power

def plot_result(frequency_lin, power, start, data):
    """
    Plots the power over the linearised frequency and saves it in Messdaten_Visuell/ZyklischeDaten
    
    Arguments:      frequency_lin       Numpy Frequency Array with linear Frequency Values
                    power               Numpy Power Array with Power Values
                    start               Start of Time Intervall ; e.g. 09:45:40.000
                    data                Name of Grid you want to load the Data ; e.g. Unteregg
    
    Returns:        path                Path where the plotted P(f) graph is getting saved
    """
    pathtime = start[:-7].replace(':', '') 
    filename_extra = "%s_%s.png" % (data + '_ZyklischeDaten', pathtime)
    path = os.path.join('Messdaten_Visuell/ZyklischeDaten', filename_extra)
    
    plt.figure(figsize=(15, 8))
    plt.plot(frequency_lin, power, "darkorange") 
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.axis([min(frequency_lin), max(frequency_lin), min(power), max(power)])
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    return path
    
def plot_everything(second, time, power, frequency, frequency_lin, reg):
    """
    Plots P(t) original, f(t) original, f_lin(t) linearised, P(f_lin)
    
    Arguments:      second              Numpy Time Array with all Time Values in seconds
                    time                Numpy Time Array with all Time Values
                    power               Numpy Power Array with all Power Values
                    frequency           Numpy Frequency Array with all frequency Values
                    frequency_lin       Numpy Frequency Array with linear Frequency Values
                    reg                 Fitted Estimator outputted from LinearRegression().fit()
    """
    
    
    p = (min(time), max(time))
    t = (min(second), max(second))

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1) 
    plt.plot(time, power, "darkgoldenrod") 
    plt.xticks([time[0], time[-1]], visible=True, rotation="horizontal")
    plt.xlim([time[0], time[-1]])
    plt.subplot(2, 2, 2) 
    plt.plot(time, frequency, "burlywood") 
    plt.xticks([time[0], time[-1]], visible=True, rotation="horizontal")
    plt.xlim([time[0], time[-1]])
    plt.subplot(2, 2, 3)
    plt.plot(p, reg.predict(t), "navajowhite")
    plt.axis([min(p), max(p), min(reg.predict(t)), max(reg.predict(t))])
    plt.subplot(2, 2, 4) 
    plt.axis([min(frequency_lin), max(frequency_lin), min(power), max(power)])
    plt.plot(frequency_lin, power, "tan") 


def smooth_frequency(frequency):
    """
    Smooths some non linear frequency array
    
    Arguments:      frequency           Numpy Frequency Array with frequency Values including noise
    
    Returns:        smooth_frequency    Numpy Array with removed noise
    """
    
    smooth_frequency = savgol_filter(frequency, 200, 1)
    
    return smooth_frequency

def plot_everything_10ms(power, norm_power, time, frequency, frequency_smooth):
    Hz = np.linspace(50,51.6,1601)
    
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1) 
    plt.plot(time, power, "darkgoldenrod") 
    plt.xticks([time[0], time[-1]], visible=True, rotation="horizontal")
    plt.xlim([time[0], time[-1]])
    plt.subplot(2, 2, 2) 
    plt.plot(time, frequency, "burlywood") 
    plt.xticks([time[0], time[-1]], visible=True, rotation="horizontal")
    plt.xlim([time[0], time[-1]])
    plt.subplot(2, 2, 3)
    plt.plot(time, frequency_smooth, "burlywood") 
    plt.xticks([time[0], time[-1]], visible=True, rotation="horizontal")
    plt.xlim([time[0], time[-1]])
    plt.subplot(2, 2, 4) 
    plt.plot(Hz, norm_power, "darkorange") 
    plt.axis([min(Hz), max(Hz), min(norm_power), max(norm_power)])
    plt.show()
    plt.close()
    
    return

def save_result(filename, norm_power):
    """
    Saves the Plot in Messdaten_Visuell/10msRMS
    
    Arguments:      filename            Name of the file which is getting saved
                    frequency_smooth    Numpy Array with removed noise
                    norm_power          Normalised Power Array with maximum value = 1 and minimum value = 0
    
    Returns:        path                Path where the Results are gettings saved
    """
    
    Hz = np.linspace(50,51.6,1601)
    path = 'Messdaten_Visuell/10msRMS/' + filename[:-4]
    
    plt.figure(figsize=(15, 8))
    plt.plot(Hz, norm_power, "darkorange") 
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.axis([min(Hz), max(Hz), min(norm_power), max(norm_power)])
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()
    
    return path


def power_1601(power, frequency):
    """
    Takes in a Power Array with random length and gives back a power array with size 1601
    
    Arguments:      power           Power Array with random size
                    frequency       frequency Array with random size
    
    Returns:        path            Power Array with size 1601
    """
    
    Hz = np.linspace(50,51.6,1601)
    new_power = []
    idx = 0
    for y in Hz:
        idx = (np.abs(frequency[idx:] - y)).argmin() + idx
        new_power.append(power[idx])
        
    new_power = savgol_filter(new_power, 10, 1)
    return new_power

def create_pickle(norm_power, filename):
    """
    Creates a pickle file with containing a power array
    
    Arguments:      norm_power          Normalised Power Array with maximum value = 1 and minimum value = 0
                    filename            Name of the file which is getting saved
    
    Returns:        path                Path where the Pickle file is getting saved
    """
    
    diff = len(norm_power)%1601

    if diff != 0:
        print('Power array Size is not divisible by 1601 and therefore not fitable to the DNN. No Pickle File created!')
        return

    filename = filename.split('_')[0] + filename.split('_')[2][:-4] + '.pkl'
    path = 'Pickle/' + filename
    with open(path, 'wb') as f:
        pickle.dump(norm_power, f)
        
    return path