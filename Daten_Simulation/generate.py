from distutils.dir_util import copy_tree
import numpy as np
import os.path
import pickle 
import random as rnd
import shutil
from tqdm import tqdm
from pathlib import Path
from utils import *

iteration = int(input("Size of the Data you want to generate: "))
set = str(input("Test or Train or Vali Data: "))

def generate_data(set):
    all_power_array = np.array([np.zeros(1601)])
    all_norms_array = np.array([np.zeros(4)])
    nrgesamt_array = np.array([np.zeros(4)])

    for i in tqdm(range (1, 1+iteration)):
        nr1, nr2, nr3, nr4 = 0, 0, 0, 0
        gesamt = np.zeros(1601)
        x = [1,2,3,4]
        Hzz = [50.25, 50.3,50.35, 50.4,50.45, 50.5,50.55, 50.6,50.6, 50.7,50.75, 50.8,50.85, 50.9,50.95, 51]
        
        for n in range(1, rnd.randint(75, 300)): # define power between 75 and 300 kWp
                rndm = rnd.choice(x)
                if rndm == 1:
                    ywerte = VDE_AR_N_4105()
                    x.append(1)
                    nr1 += 1
                if rndm == 2:     
                    ywerte = VDEW_2001()
                    x.append(2)
                    nr2 += 1
                if rndm == 3: 
                    ywerte = DIN_V_VDE_V()
                    x.append(3)
                    nr3 += 1
                if rndm == 4: 
                    Hzvalue = rnd.choice(Hzz)
                    Hzz.remove(Hzvalue)
                    if Hzz == []:
                        Hzz = [50.25, 50.3,50.35, 50.4,50.45, 50.5,50.55, 50.6,50.6, 50.7,50.75, 50.8,50.85, 50.9,50.95, 51]   
                    ywerte = SysStabV(Hzvalue)
                    x.append(4)
                    nr4 += 1
                gesamt = np.add(ywerte, gesamt)

        Hz = np.linspace(50,51.6,1601)
        
        randomdate = random_date(datetime.strptime('1/1/2022 1:00 PM', '%m/%d/%Y %I:%M %p'), 
                             datetime.strptime('12/31/2022 12:00 AM', '%m/%d/%Y %I:%M %p'))
        
        wg = wirkungsgrad(randomdate)

        maxvalue = gesamt[0]
        noise = rndmnoise(maxvalue)

        finaly = (gesamt+noise)*wg

        minvalue = np.amin(finaly)
        finaly = finaly + abs(minvalue)

        maxvalue = np.amax(finaly)
        finaly = finaly / maxvalue

        all_power_array = np.insert(all_power_array, all_power_array.shape[0], np.array([finaly]), axis=0)

        # create path
        folderpath = 'DataSets/' + str(set) + '_' + str(iteration)
        
        Path(folderpath).mkdir(parents=True, exist_ok=True)
        pathpng = folderpath + '/%i.png' % i
        
        plt.plot(Hz,finaly)
        plt.xlabel("Frequency (HZ)")
        plt.ylabel("Power From all PV Moduls")
        plt.savefig(pathpng, dpi=300)
        plt.close()

        pathtxt = 'DataSets/' + str(set) + '_' + str(iteration) + '/%i.txt' % i
        
        if set == 'Vali':
            pathpkl = folderpath +  '/%i.pkl' % i
            with open(pathpkl, 'wb') as f:
                pickle.dump(finaly, f)
                
                
        sourceFile = open(pathtxt, 'w')        
        nrgesamt = nr1 + nr2 + nr3 + nr4
        print("Iteration:", i, "/" ,iteration,"\n", file = sourceFile)
        # print("Iteration:", i, "/" ,iteration,"\n")
        print("Moduls in Grid:" , nrgesamt,"\n", file = sourceFile)
        #print("Moduls in Grid:" , nrgesamt,"\n")
        print("Wirkungsgrad der Module:", wg,"\n", file = sourceFile)
        #print("Wirkungsgrad der Module:", wg,"\n")
        print("VDE_AR_N_4105:", nr1, file = sourceFile)
        #print("VDE_AR_N_4105:", nr1)
        print("VDEW_2001:", nr2, file = sourceFile)
        #print("VDEW_2001:", nr2)
        print("DIN_V_VDE_V:", nr3, file = sourceFile)
        #print("DIN_V_VDE_V:", nr3)
        print("SysStabV:", nr4, file = sourceFile)
        #print("SysStabV:", nr4)
        
        pnr1 = nr1 / nrgesamt # portion of VDE_AR_N_4105
        pnr2 = nr2 / nrgesamt # portion of VDEW_2001
        pnr3 = nr3 / nrgesamt # portion of DIN_V_VDE_V
        pnr4 = nr4 / nrgesamt # portion of SysStabV
        
        all_norms_array = np.insert(all_norms_array, all_norms_array.shape[0], np.array([np.array([pnr1, pnr2, pnr3, pnr4])]), axis=0)
        nrgesamt_array = np.insert(nrgesamt_array, nrgesamt_array.shape[0], np.array([np.array([nr1, nr2, nr3, nr4])]), axis=0)
        
        create(all_power_array[1:], all_norms_array[1:], set, nrgesamt_array[1:], iteration)

        

    from_directory = "DataSets"
    to_directory = "../DNN/DataSets"
    copy_tree(from_directory, to_directory)

    from_directory = "h5"
    to_directory = "../DNN/h5"
    copy_tree(from_directory, to_directory)
        
if __name__ == "__generate_data__":
    generate_data() 
    
generate_data(set)