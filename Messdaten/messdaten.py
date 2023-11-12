import numpy as np
import os.path
import pickle 
from datetime import datetime
from distutils.dir_util import copy_tree
from utils import *

mode = str(input("Zyklisch or 10msRMS: "))

def messdaten():
    if mode == "Zyklisch":
        print('Which Zyklische Daten do you want to use?')
        datalist = os.listdir("Messdaten/ZyklischeDaten")
        datalist = [x[:-19] for x in datalist]
        [print(i) for i in datalist]
        data = str(input())
        
        assert data in datalist
        
        start = str(input("Start of Time Intervall (00:00:00.000): "))
        end = str(input("End of Time Intervall (00:00:00.000): "))

        power, time, frequency = load_data(data, start, end)
        
        seconds = time_in_sec(time)

        frequency_lin, reg = lin_regression(seconds, frequency)
        
        power = normalize(power)

        #plot_everything(seconds, time, power, frequency, frequency_lin, reg)

        path = plot_result(frequency_lin, power, start, data)
        
        print('Plot Saved Under ', path)
        
        
    if  mode == '10msRMS':
        datalist = os.listdir("Messdaten/10msRMS")
        list = [*set([i.split('_')[0] for i in datalist])]
        [print(i) for i in list]
        data = str(input('Which 10msRMS Interval do you want to use: '))
        [print((k.split('_')[2][:-4])[:2]+':'+(k.split('_')[2][:-4])[2:]) for k in datalist if data in k]
        time = str(input('Which Time Interval do you want to use: '))
        time = time.replace(':', '')
        
        filename = data + '_10msRMS_' + time + '.asc'
        
        power, time, frequency = load_data_10ms(filename)
        
        frequency_smooth = smooth_frequency(frequency)
        
        power1601 = power_1601(power, frequency_smooth)
        norm_power = normalize(power1601)
        
        plot_everything_10ms(power, norm_power, time, frequency, frequency_smooth)
        
        path = save_result(filename, norm_power)
        
        print('Plot Saved Under', path)
        
        h5_mode = str(input('Do you want to create a pickle file: '))
        
        if h5_mode == 'Yes':
            
            path = create_pickle(norm_power, filename)
            
            from_directory = "Pickle"
            to_directory = "../DNN/Pickle"
            copy_tree(from_directory, to_directory)
            #frequency_smooth = frequency_smooth[::int(frequency_smooth.shape[0]/1601)]
            
            
        return filename, norm_power
    
if __name__ == "__messdaten__":
    messdaten()
messdaten()