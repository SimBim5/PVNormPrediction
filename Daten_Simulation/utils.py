import h5py
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pvlib
import random as rnd

from datetime import datetime
from datetime import timedelta 
from uncertainties import ufloat

def Power():
    """
    Generaters a random number between 2 and 33 (6% chance between 33 and 120) for the Power of a single PV-Module/Norm
    
    Arguments:
    
    Returns:        Power           Predictions for the given dataset X
    """
    Power = rnd.uniform(2, 33)
    if rnd.randint(0,100) < 6:
        Power = rnd.uniform(33, 120)
    return Power


def VDE_AR_N_4105():
    """
    Simulates a single a VDE_AR_N_4105 Norm - Ramp Function
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a VDE_AR_N_4105 Norm with random Power ; len 1601
    """
    Pow = Power()
    ywerte = []
    Hz = np.linspace(50,51.6,1601)
    
    noise1 = rnd.uniform(50.19, 50.21)
    noise2 = rnd.uniform(51.49, 51.51)
    
    for x in Hz: 
        match x:
            case x if x >= noise1 and x <= noise2:
                P = Pow - ((abs(x-noise1) * 0.4) * Pow)
            case x if x < noise1:
                P = Pow
            case x if x > noise2:
                P = 0
        ywerte.append(P)
    return ywerte

def VDEW_2001():
    """
    Simulates a single a VDEW_2001 Norm - 50.5 Hz
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a VDEW_2001 Norm with random Power ; len 1601
    """
    Pow = Power()
    ywerte = []
    Hz = np.linspace(50,51.6,1601)

    noise = rnd.uniform(50.49, 50.51)
    
    for x in Hz:
        match x: 
            case x if x >= noise:
                P = 0
            case x if x < noise:
                P = Pow
        ywerte.append(P)
    return ywerte

    
def DIN_V_VDE_V():
    """
    Simulates a single a DIN_V_VDE_V Norm - 50.2 Hz
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a DIN_V_VDE_V Norm with random Power ; len 1601
    """
    Pow = Power()
    ywerte = []
    Hz = np.linspace(50,51.6,1601)
    
    noise = rnd.uniform(50.19, 50.21)
    
    for x in Hz:
        match x: 
            case x if x >= noise:
                P = 0
            case x if x < noise:
                P = Pow
        ywerte.append(P)
    return ywerte


def SysStabV(Hzvalue):
    """
    Simulates a single a SysStabV Norm - Threshold
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a SysStabV Norm with random Power ; len 1601
    """
    Pow = Power()
    ywerte = []
    threshold = ufloat(Hzvalue, 0.01)
    Hz = np.linspace(50,51.6,1601)
    
    for x in Hz:
        match x: 
            case x if x >= threshold:
                P = 0
            case x if x < threshold:
                P = Pow
        ywerte.append(P)
    return ywerte


def random_date(start, end):
    """
    This function will return a random datetime between two datetime objects
    
    Arguments:      start       Start of the Time Interval ; e.g. 2022-01-01 13:00:00
                    end         End of the Time Interval ; e.g. 2022-12-31 12:00
                    
    Returns:        date        Random Date somewhere between Start and End ; e.g.  2022-12-07 19:59     
    """
    
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = rnd.randrange(int_delta)
    return start + timedelta(seconds=random_second)
    
def rndmnoise(maxvalue):
    """
    This function will create some rndm noise +- 0.0015/0.001 of the maximal Value of Power
    
    Arguments:      maxvalue    The maximum Value of the Power ; single value
                    
    Returns:        x           The noise created ; np.array with shape (1601, )
    """
    
    x = [0]
    it = 0
    
    for i in range(0,800): 
        noise = rnd.uniform(-maxvalue*0.0015, maxvalue*0.001)
        x = np.append(x, x[it] + noise)
        x = np.append(x, x[it+1] + noise)
        it += 2
    return x
   
   
def uniformnoise(maxvalue):
    """
    This function will create some uniform distributed noise depending on the maximum value of the Power
    
    Arguments:      maxvalue    The maximum Value of the Power ; single value
                    
    Returns:        noise       The noise created ; np.array with shape (1601, )
    """
    
    sigma = rnd.uniform(0, maxvalue*0.001)
    mu = 0
    noise = sigma * np.random.randn(1601) + mu
    return noise
   
def wirkungsgrad(randomdate):
    """
    This function calculates some efficiency with which all PV-Moduls work 
    
    Arguments:      randomdate      Random Date somewhere between Start and End ; e.g.  2022-12-07 19:59
                    
    Returns:        wg              Value between 0.02 and 1.00
    """

    strahlung, temperatur = solar(randomdate)
    
    if 0 < strahlung <= 100:
        wirkungsgrad1 = 0.0*100
    if 100 < strahlung <= 300:
        wirkungsgrad1 = 0.02*100
    if 300 < strahlung <= 500:
        wirkungsgrad1 = 0.4*100
    if 500 < strahlung <= 700:
        wirkungsgrad1 = 0.56*100
    if 700 < strahlung <= 900:
        wirkungsgrad1 = 0.8*100
    if 900 < strahlung <= 1100:
        wirkungsgrad1 = 1.0*100
    if 1100 < strahlung:
        wirkungsgrad1 = 1.2*100
        
    wirkungsgrad = 100 - (abs(25 - temperatur) *0.4)
    wg = (wirkungsgrad/100 * wirkungsgrad1/100)
    
    if wg <= 0.02:
        wg = 0.02
    return wg
    
def solar(randomdate):
    """
    This function shows the Radiation and Temperatur at a random Timeslot in New Mexico
    
    Arguments:      randomdate      Random Date somewhere between Start and End ; e.g.  2022-12-07 19:59
                    
    Returns:        strahlung       Single Value representing the current Radiation ; e.g. 1400 kWh/m^2
                    temperatur      Single Value representing the current Temperature ; e.g. 22.8 °C 
    """
    
    # (year-month-day)
    DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
    df_tmy, meta_dict = pvlib.iotools.read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=2022)
    meta_dict  # display the dictionary of metadata

    df = df_tmy[['GHI', 'DHI', 'DNI', 'DryBulb', 'Wspd']]
 
    randomdate = str(randomdate)[:-9]

    day = df.loc[randomdate:randomdate]
    plt.plot(day['DNI'], color='r') 
    plt.plot(day['DHI'], color='g', marker='.') 
    plt.plot(day['GHI'], color='b', marker='s')
    plt.ylabel('Irradiance [W/m$^2$]');
    #plt.show()
    plt.close()
    
    hour = rnd.randint(10, 16)
    hour = str(hour) + ":00:00-05:00"
    
    intervall = df_tmy.loc['%s %s' % (hour , randomdate) ]
    #print(randomdate, hour[:-6])
    #print('Temperatur:', intervall['DryBulb'])
    #print('Gesamtstrahlung:', intervall['DNI'] + intervall['DHI'] + intervall['GHI'] )
    temperatur =  intervall['DryBulb']
    strahlung = intervall['DNI'] + intervall['DHI'] + intervall['GHI']
    
    return strahlung, temperatur

def last(randomdate):
    """
    This function gives back the load off some weekday in either Winter, Summer or Intermediate Time
    
    Arguments:      randomdate      Random Date somewhere between Start and End ; e.g.  2022-12-07 19:59
                    
    Returns:        x               Time Steps ; 0:00, 0:15, 0:30, 0:45, 1:00, 1:15
                    y               The Load at the time x ; in kW
    """
    
    path = "Last/Lastprofil_Haushalt_H0.xlsx"
    
    df = pd.read_excel(path)
    x = df.Jahreszeit
    x = df[["Jahreszeit"]].to_numpy(x)
    
    if 0 <= int(randomdate[6:-3]) <= 4 or 11 <= int(randomdate[6:-3]) <= 12:
        z = [1,1,1,1,1,2,3]
        z = rnd.choice(z)
        if z == 1:
            df = pd.read_excel(path)
            y = df.WinterWerktag
            y = df[["WinterWerktag"]].to_numpy(y)
        if z == 2:
            df = pd.read_excel(path)
            y = df.WinterSamstag
            y = df[["WinterSamstag"]].to_numpy(y)
        if z == 3:
            df = pd.read_excel(path)
            y = df.WinterSonntag
            y = df[["WinterSonntag"]].to_numpy(y)
    if 6 <= int(randomdate[6:-3]) <= 9:
        z = [1,1,1,1,1,2,3]
        z = rnd.choice(z)
        if z == 1:    
            df = pd.read_excel(path)
            y = df.SommerWerktag
            y = df[["SommerWerktag"]].to_numpy(y)
        if z == 2:
            df = pd.read_excel(path)
            y = df.SommerSamstag
            y = df[["SommerSamstag"]].to_numpy(y)
        if z == 3:
            df = pd.read_excel(path)
            y = df.SommerSonntag
            y = df[["SommerSonntag"]].to_numpy(y)
    else: 
        z = [1,1,1,1,1,2,3]
        z = rnd.choice(z)
        if z == 1:
            df = pd.read_excel(path)
            y = df.ÜbergangszeitWerktag
            y = df[["ÜbergangszeitWerktag"]].to_numpy(y)
        if z == 2:
            df = pd.read_excel(path)
            y = df.ÜbergangszeitSamstag
            y = df[["ÜbergangszeitSamstag"]].to_numpy(y)
        if z == 3:   
            df = pd.read_excel(path)
            y = df.ÜbergangszeitSonntag
            y = df[["ÜbergangszeitSonntag"]].to_numpy(y)
    return x, y


def create(all_power_array, all_norms_array, set, nrgesamt_array, iteration):
    """
    This function creates the h5 file used for Neural Networks. It saves the files in 'path2'
    
    Arguments:      all_power_array     Numpy Array Containing the Power Values for all Iterations ; shape (number examples, 1601)
                    all_norms_array     Numpy Array Containing the Norms in a Grid for all Iterations in % ; shape (number examples, 4)
                    set                 Test, Train or Validation      
                    nrgesamt_array      Numpy Array Containing the Norms in a Grid for all Iterations in absolute Values ; shape (number examples, 4)
                    iteration           Number Iterations
    """
    
    filename = str(set) + '_' + str(iteration) + '.h5'
    path2 = r"C:\Users\49152\Desktop\FP\Daten_Simulation\h5"
    path3 = str(os.path.join(path2, filename))

    if set == 'Test':
        test = h5py.File(path3, 'w')
        
        classes = ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV']

        test.create_dataset('list_classes', data = classes)
        test.create_dataset('test_set_x', data = all_power_array)
        test.create_dataset('test_set_y', data = all_norms_array)
        test.create_dataset('test_set_nrgesamt', data = nrgesamt_array)
    
    if set == 'Train':
        train = h5py.File(path3, 'w')
        
        classes = ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV']

        train.create_dataset('list_classes', data = classes)
        train.create_dataset('train_set_x', data = all_power_array)
        train.create_dataset('train_set_y', data = all_norms_array)

    if set == 'Vali':
        vali = h5py.File(path3, 'w')
        
        classes = ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV']

        vali.create_dataset('list_classes', data = classes)
        vali.create_dataset('vali_set_x', data = all_power_array)
        vali.create_dataset('vali_set_y', data = all_norms_array)
     
     
     
     
    #   to read a h5 file:    
    #with h5py.File(filename, "r") as f:
    #print("Keys: %s" % f.keys())
    #a_group_key = list(f.keys())[0]
    #data = list(f[a_group_key])
    #print(data)