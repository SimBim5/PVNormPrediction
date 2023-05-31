<h1 align="center">PV NORM PREDICTION</h1>
<p align="center">
</p>

As a part of the project LINDA 2.0, several field test will be performed on low voltage grids composed of rooftop solar plants and housholds. During the last 20 years in Germany, several norms for connection of photovoltaic power plants have been proposed, whose field implementation have been a direct responsibility of the grid operators. This has led to lack of information regarding the norms that each of the connected solar power plants follow and its rated power. To be able to identify each solar power plant inside a low voltage grid, a machine learning algorithm has been developed in this work. 


# 🚀🔬😎 LINDA 2.0 🔎🔆🔋

![LINDA 2.0](Photos/Linda.png)
The consequences of a long-lasting and widespread power blackout can approach a national catastrophe with serious consequences for civil society. The use of decentralized energy supply systems as an emergency supply for critical infrastructure using island grids can significantly reduce the damage in such scenarios. In the LINDA research project (local stand-alone grid supply and accelerated grid reconstruction with decentralized generation plants), a concept for stable stand-alone grid operation in the event of an emergency supply was developed and tested in a southern German grid area under real conditions. In LINDA 2.0, the LINDA concept will be transferred to another test area and (partially) automated. The object of investigation is a constellation of run-of-river power plant as an island network-forming unit and drinking water supply as critical infrastructure. Several field tests are planned. In addition, as part of the project, a hybrid unit is being developed and tested in the distribution network. The hybrid unit is intended to be an alternative to a conventional emergency power unit and consists of an inverter that forms an island grid with battery storage and a diesel generator as a range extender. In regular operation, the range extender is switched off and the unit works completely emission-free (noise, exhaust gas and CO2).

# ☀️🔋⚡ PV Norms  💡🌞🔌

As already mentioned, there are many norms for connection of photovoltaic power plants in germany, proposed over the last 20 years. Because we want to predict PV norms based on the behavior when the frequency is reduced, the following 4 PV norms are observed (Power per Unit over Frequency):

1. DIN V VDE V (2006)
<img src="Photos/2.png" alt="DIN V" style="width: 60%;">

2. VDEW 2001 (1991)
<img src="Photos/4.png" alt="VDEW 2001" style="width: 60%;">

3. VDE4105 (2001/2005)
<img src="Photos/1.png" alt="VDE 4105" style="width: 60%;">

4. SysStab V (2012)
<img src="Photos/3.png" alt="Sys Stab V" style="width: 60%;">


# 🤖⚙️🔧 Code | DNN 🦾👩‍💻💻
For the predicition of the PV-Norms, a deep neural network is been used. For the usage open the DNN folder in environment you wish and run the dnn.py file. 
Firstly, you will get asked if you want to Train, Test or Predict. 

## Train
If you type 'Train', you will train some parameters (weights, bias) for the neural network. After the code has finished, the resuts will get saved under DNN/parameters. For training, you will have to put some trainings data under DNN/DataSets. To generate some Data Sets, check [Data Simulation](#Simulation)

Next you will be asked, which TrainSet to train on. If the folder DNN/DataSets is not empty, the Data Set should be shown to you. Enter the name to train on this set. 

For Multiple Trainings you can either type Yes or No. 
Yes: Multiple Trainings with the learning rates 0.001, 0.003, 0.005, 0.007, 0.009, 0.011, will be made.
No: Only one Training with the learning rate you enter will be made. 

Next you will get asked about the Amount of Iterations. This means the number of passes of the training data through the network. Usually good performance is made with huge iterations and small learning rate. 

After you enter the Amount of Iterations, finally you can enter the Learning rate (if you didnt chose Yes in Multiple Trainings). Afterwards the training will start, and the trained paramaters will be saved in DNN/Parameters. 

## Test
If you type 'Test', you can now test the performance of specific parameters, you trainied beforehand. For that purpose you have to put an Test Set inside the folder DNN/DataSets. If you do not have one, you can again create one with [Data Simulation](#Simulation). To go on with the code, type the Test Set you want to use. 

Now type the parameters you want to test. 

The code will run and print the Predicted Norms + Ground Truth for every single example in the Test Set. It will also print the % of the wrongly predicted Norms, and the overall % of the wrongly predicted Norms. 

## Predict
If you type Predict, you can run the code on a single measurement. Again, firstly enter the name of the parameters you want to use for your prediction. After that type the name of the measurment you want to feed in the neural network. For that purpose, the measurment (1601 Power Values in form of a .pkl file) has to be inside the folder DNN/Pickle. Now, type one of the .pkl file names to predict Norms on it. 

The Predictions form the Net + the Ground Truth will now be shown. 

If you continue with typing the location, date, and time (when and where the measurments where made), you will also be shown the estimated generated power of all PV Moduls inside the grid (multiplied with a multiplied with a certain efficiency estimated from irradiation and temperature on this day and place).

Also the code will show you the estimated load inside the grid estimated from standard load profiles (H0, G0, L0).

<a name="Simulation"></a>
# 📝🧾📂 Code | Data Simulation 📋📕⚙️


# 📝🧾📂 Code | Messdaten 📋📕⚙️

