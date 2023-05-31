import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os.path
import time
from tqdm import tqdm
import shutil
from utils import *

#plt.ion()  # Enable interactive mode

# Set default plot configurations
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Set random seed
np.random.seed(1)

# Get mode from user input
mode = str(input("Train, Test, or Predict: "))

if mode == 'Train':
    # Get available trainsets and select one
    trainsets = [x for x in os.listdir("h5") if x.startswith("Train")]
    [print(x) for x in trainsets]
    trainset = str(input("Enter the name of the trainset: "))
    train_x, train_y, classes = load_data_train(trainset)
    m_train = train_x.shape[0]

if mode == 'Test':
    # Get available testsets and select one
    testsets = [x for x in os.listdir("h5") if x.startswith("Test")]
    [print(x) for x in testsets]
    testset = str(input("Enter the name of the testset: "))
    test_x, test_y, classes, test_set_pv_norms = load_data_test(testset)
    m_test = test_x.shape[0]


layers_dims = [1601, 200, 100, 50, 25, 12, 4]


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    np.random.seed(1)
    best_parameters = {}
    costs = []
    
    
    parameters = initialize_parameters_deep(layers_dims)
     
    for i in tqdm(range(0, num_iterations)):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Save parameters if cost improves
        if all(i >= cost for i in costs):
            best_parameters = parameters
        
        costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    
    min_cost = min(costs)
    min_index = costs.index(min_cost)
    print('Min Cost:', min_cost, 'in Iteration:', min_index)

    
    return parameters, best_parameters

if __name__ == "__L_layer_model__":
    L_layer_model()


if mode == 'Train': 
    # Check if multiple trainings are desired
    multiple = str(input("Multiple trainings? (Yes/No): "))

    
    if multiple == 'No':
        iterations = int(input("Amount of iterations: "))
        learning_rate = float(input("Learning rate: "))
        filename = f'Parameters/parameters_{m_train}_{iterations}_{learning_rate}.pkl'
        
        parameters, best_parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, iterations,
                                                   print_cost=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(best_parameters, f)
            
        print(f'Parameters saved under {filename}')
       
            
    if multiple == 'Yes':
        iterations = int(input("Amount of iterations: "))
        
        for learning_rate in np.arange(0.001, 0.012, 0.002):
            learning_rate = round(learning_rate, 3)
            filename = f'Parameters/parameters_{m_train}_{iterations}_{learning_rate}.pkl'
            
            parameters, best_parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, iterations, print_cost = True)
            
            with open(filename, 'wb') as f:
                pickle.dump(best_parameters, f)

            print(f'Parameters saved under {filename}')

if mode == 'Test': 
    # Get available parameter files and select one
    parameter_files = os.listdir("Parameters")
    [print(x) for x in parameter_files]
    parameterfile = str(input("Enter the name of the parameters you want to use:\t\t"))

    
    parameterpath = f'Parameters/{parameterfile}'
    with open(parameterpath, 'rb') as f:
        parameters = pickle.load(f)

    
    nr = 0
    all_error = 0
    for test in test_x:
        powerarray = []
        powerarray.append(test)
        powerarray = np.array(powerarray)

        
        pvnorm1 = test_set_pv_norms[nr][0]
        pvnorm2 = test_set_pv_norms[nr][1]
        pvnorm3 = test_set_pv_norms[nr][2]
        pvnorm4 = test_set_pv_norms[nr][3]
        
        nr = nr +1
        result = metric(powerarray, parameters, nr, pvnorm1, pvnorm2, pvnorm3, pvnorm4) 
        
        all_error += result

    print(f'{(all_error / nr):.2f}% PV Norms are not predicted correctly.')

if mode == 'Predict': 
    parameterfile = str(input("Enter the name of the parameters you want to use: \t\t\t"))
    powerarrayfile = str(input("Enter the name of the Powerarray you want to use for the prediction: \t"))
    
    dataframe = pd.read_excel('Anlagendaten/Anlagendaten.xlsx')

    parameterpath = 'Parameters/' + parameterfile
    with open(parameterpath, 'rb') as f:
        parameters = pickle.load(f)

    powerarraypath = 'Pickle/' + powerarrayfile
    with open(powerarraypath, 'rb') as f:
        powerarray = pickle.load(f)

    powerarray2 = []
    powerarray2.append(powerarray)
    powerarray2 = np.array(powerarray2) 

    if powerarrayfile[:-8] in set(dataframe['Ort']):

        vdepower, dinpower, syspower, vdewpower = calculate_power(parameterfile, powerarrayfile)

        location = str(input(Fore.RESET + "Type the location:\t\t\t "))
        date = str(input("Type the date (YYYY-MM-DD):\t\t "))
        time = input('Type the time (HH:MM):\t\t\t ')
        temperature, irradiance = solar(location, date, time)

        wirkungsgrad = wirkungsgrad(temperature, irradiance)
        
        vdepower = wirkungsgrad * vdepower
        vdewpower = wirkungsgrad * vdewpower
        dinpower = wirkungsgrad * dinpower
        syspower = wirkungsgrad * syspower
        
        print('Leistung der PV Anlagen:\t\t', (vdepower + vdewpower + dinpower + syspower), '[kW]')
        terminal_width = shutil.get_terminal_size().columns
        print('#' * terminal_width)
        
        plot_result(vdepower, vdewpower, dinpower, syspower)
        
        load = load(powerarrayfile[:-8], date, time)
        
        plot_result_with_load(vdepower, vdewpower, dinpower, syspower, load)
        
    else:
        nrModuls = None
        vde, vdew, din, sys = predict(powerarray2, parameters, nrModuls)
        plot_result(vde, vdew, din, sys)