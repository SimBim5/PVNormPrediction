import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import pickle 
import os.path


from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from utils import *


modus = str(input("Train,Test or Predict: "))


plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

if modus == 'Train':
    [print(x) for x in os.listdir("h5") if x.startswith("Train") == True]
    trainset = str(input("On which Trainset do you want to Train on "))
    train_x, train_y, classes = load_data_train(trainset)

    m_train = train_x.shape[0]

    #print ("Number of training examples: " + str(m_train))
    #print ("train_x_orig shape: " + str(train_x.shape))
    #print ("train_y shape: " + str(train_y.shape))
    #print ("train_x's shape: " + str(train_x.shape))

if modus == 'Test':
    [print(x) for x in os.listdir("h5") if x.startswith("Test") == True]
    testset = str(input("On which Testset do you want to Test on "))
    test_x, test_y, classes, test_set_pv_norms = load_data_test(testset)
    
    m_test = test_x.shape[0]

    #print ("Number of testing examples: " + str(m_test))
    #print ("test_x_orig shape: " + str(test_x.shape))
    #print ("test_y shape: " + str(test_y.shape))
    #print ("test_x's shape: " + str(test_x.shape))

layers_dims = [1601, 200, 100, 50, 25, 12, 4]

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, print_cost=False):
    
    np.random.seed(1)
    costs = []    # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
     
    for i in tqdm(range(0, num_iterations)):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        #if print_cost and i % 100 == 0:
        #    print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    min_cost = min(costs)
    min_index = costs.index(min_cost)
    print('Min Cost', min(costs), 'in Iteration:', min_index * 100)
    
    return parameters
if __name__ == "__L_layer_model__":
    L_layer_model()


if modus == 'Train': 
    iterations = int(input("Amount of Iterations: "))
    learning_rate = float(input("Learning Rate: "))
    filename = 'Parameters/parameters_' + str(m_train) + '_' + str(iterations) + '_' + str(learning_rate) + '.pkl'
    
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate, iterations, print_cost = True)
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)


if modus == 'Test': 
    [print(x) for x in os.listdir("Parameters")]
    parameterfile = str(input("Type the name of the parameters you want to use: "))
    
    parameterpath = 'Parameters/' + parameterfile
    with open(parameterpath, 'rb') as f:
        parameters = pickle.load(f)
    
    nr = 0
    allerror = 0
    vdeerror = 0
    vdewerror = 0
    dinerror = 0
    syserrir = 0
    for test in test_x:
        powerarray = []
        powerarray.append(test)
        powerarray = np.array(powerarray)
        
        pvnorm1 = test_set_pv_norms[nr][0]
        pvnorm2 = test_set_pv_norms[nr][1]
        pvnorm3 = test_set_pv_norms[nr][2]
        pvnorm4 = test_set_pv_norms[nr][3]
        
        nr = nr +1
        result, errorrateVDE, errorrateVDEW, errorrateDIN, errorrateSys = metric(powerarray, parameters, nr, pvnorm1, pvnorm2, pvnorm3, pvnorm4) 
        
        vdeerror = errorrateVDE + vdeerror
        vdewerror = errorrateVDEW + vdewerror
        dinerror = errorrateDIN + dinerror
        syserrir = errorrateSys + syserrir
        allerror = allerror + result
        
    print(vdeerror, 'VDE_AR_N_4105 PV Norms are not predicted correctly.')
    print(vdewerror, 'VDEW_2001 PV Norms are not predicted correctly.')
    print(dinerror, ' DIN_V_VDE_V PV Norms are not predicted correctly.')
    print(syserrir, 'SysStabV PV Norms are not predicted correctly.')
    print(f'{(allerror/nr):.2f}% PV Norms are not predicted correctly.')

if modus == 'Predict': 
    parameterfile = str(input("Type the name of the parameters you want to use: "))
    powerarrayfile = str(input("Type the name of the Powerarray you want to use for the Prediction: "))

    nrModulsknown = str(input("Do we know how much Moduls are in the grid? (Type either Yes or No): "))
    if nrModulsknown == 'Yes':
        nrModuls = int(input("How many Moduls are in the grid?" ))
    if nrModulsknown == 'No': 
        nrModuls = None

    parameterpath = 'Parameters/' + parameterfile
    with open(parameterpath, 'rb') as f:
        parameters = pickle.load(f)
        
    powerarraypath = 'Pickle/' + powerarrayfile
    with open(powerarraypath, 'rb') as f:
        powerarray = pickle.load(f)

    powerarray2 = []
    powerarray2.append(powerarray)
    powerarray2 = np.array(powerarray2)

    prediction = predict(powerarray2, parameters, nrModuls)