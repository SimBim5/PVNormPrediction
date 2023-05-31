import geopy
import h5py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import shutil
from colorama import init, Fore


def sigmoid(Z):
    """
    Implement Sigmoid Activation Function
    
    Arguments:      Z           Numpy Array of any Shape
    
    Returns:        A           Output of sigmoid(z), same Shape as Z
                    cache       returns Z ; stored for computing the Backward Pass
    """

    # Compute the sigmoid activation
    A = 1 / (1 + np.exp(-Z))

    # Store Z for computing the backward pass in the neural network
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU Function

    Arguments:      Z           Output of the linear layer, of any shape

    Returns:        A           Output of Relu(z), same Shape as Z
                    cache       returns A ; stored for computing the Backward Pass
    """
    
    # Apply the ReLU activation function element-wise
    A = np.maximum(0, Z)
    
    # Ensure the shapes of A and Z are the same
    assert(A.shape == Z.shape)
    
    # Store Z for computing the backward pass in the neural network
    cache = Z
    
    return A, cache


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit

    Arguments:      dA          post-activation gradient, of any shape
                    cache       Z ; which we store for computing backward propagation

    Returns:        dZ          Gradient of the cost with respect to Z
    """
    
    # Retrieve the stored value of Z from cache
    Z = cache
    
    # Compute the sigmoid activation function of Z
    s = 1 / (1 + np.exp(-Z))
    
    # Compute the gradient of the cost with respect to Z using the chain rule
    dZ = dA * s * (1 - s)
    
    # Ensure the shape of dZ matches the shape of Z
    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, cache):
    """
    Implement the Backward Propagation for a single RELU unit

    Arguments:      dA          post-activation gradient, of any shape
                    cache       Z ; which we store for computing backward propagation

    Returns:        dZ          Gradient of the cost with respect to Z
    """
    
    # Retrieve the stored value of Z from cache
    Z = cache

    # Create a copy of dA as dZ to avoid modifying the original dA
    dZ = np.array(dA, copy=True)

    # Set gradients to zero where the corresponding Z value is less than or equal to zero
    dZ[Z <= 0] = 0

    # Ensure the shape of dZ matches the shape of Z
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data_train(trainset):
    """
    Loading the data from the h5 file for Training

    Returns:        train_set_x_orig        Powerarray from all Example Grids ; used for training
                    train_set_y_orig        Norms for all Example Grids ; used for training
                    classes                 Norm Names ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV'] 
    """
    
    # Construct the path to the h5 file
    trainset = 'h5/' + trainset
    
    # Open the h5 file in read mode
    train_dataset = h5py.File(trainset, "r")
    
    # Load the power array for training examples
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    
    # Load the norms for training examples
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    # Load the norm names
    classes = np.array(train_dataset["list_classes"][:])
        
    return train_set_x_orig, train_set_y_orig, classes


def load_data_test(testset):
    """
    Loading the data from the h5 file for Testing

    Returns:        test_set_x_orig         Powerarray from all Example Grids ; used for testing
                    test_set_y_orig         Norms for all Example Grids ; used for testing
                    test_set_pv_norms       Amount of Norms for all Exampele Grids ; used for testing
                    classes                 Norm Names ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV'] 
    """
    
    # Construct the path to the h5 file
    testset = 'h5/' + testset
    
    # Open the h5 file in read mode
    test_dataset = h5py.File(testset, "r")
    
    # Load the power array for testing examples
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    
    # Load the norms for testing examples
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    # Load the amount of PV norms for testing examples
    test_set_pv_norms = np.array(test_dataset["test_set_nrgesamt"][:])
    
    # Load the norm names
    classes = np.array(test_dataset["list_classes"][:])
    
    return test_set_x_orig, test_set_y_orig, classes, test_set_pv_norms



def initialize_parameters_deep(layer_dims):
    """
    Random initialize all parameters inside the Deep Neural Network (weights and biases)
    
    Arguments:      layer_dims      Python Array containing the Amount of Nodes of each Layer
    
    Returns:        parameters      Python Dictionary containing all Parameters "W1", "b1", ..., "WL", "bL":
                    Wl              Weight Matrix ; Shape (layer_dims[l], layer_dims[l-1])
                    bl              Bias Vector ; Shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        # Randomly initialize the weight matrix using He initialization
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        
        # Initialize the bias vector with zeros
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        # Reshape the weight and bias matrices to match the desired dimensions        
        parameters['W' + str(l)] = np.reshape(parameters['W' + str(l)], (((parameters['W' + str(l)]).shape)[1],((parameters['W' + str(l)]).shape)[0]))
        parameters['b' + str(l)] = np.reshape(parameters['b' + str(l)], (((parameters['b' + str(l)]).shape)[1],((parameters['b' + str(l)]).shape)[0]))
        
        # Ensure the shapes of weight and bias matrices are correct
        assert(parameters['W' + str(l)].shape == (layer_dims[l-1], layer_dims[l]))
        assert(parameters['b' + str(l)].shape == (1, (layer_dims[l])))
        
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation ( z = a1w1 + a2w2 + a3w3 + ... + b)

    Arguments:      A           Input in some Layer     (number of examples , size of previous layer / input data)
                    W           Weight Matrix           (size of previous layer , size of current layer)
                    b           Bias Vector             (1 , size of the current layer)

    Returns:        Z           Input for the Activation Function 
                    cache       Python Dictionary containing "A", "W" and "b" ; stored for computing the backward pass
    """
    
    # Compute the linear combination
    Z = A.dot(W) + b

    # Ensure the shape correctness of the output
    assert(Z.shape == (A.shape[0], W.shape[1]))

    # Store the variables needed for the backward pass in the cache
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the whole Forward Propagation ; Input in Layer -> Linear Forward -> Activation Function -> Output of Layer

    Arguments:      A_prev          Input in some Layer     (number of examples , size of previous layer / input data)
                    W               Weight Matrix           (size of previous layer , size of current layer)
                    b               Bias Vector             (1 , size of the current layer)
                    activation      activation function to be used in this layer ; "sigmoid" or "relu"

    Returns:        A               Output of the Activation Function ; A = sigmoid(Z) / relu(Z)
                    cache           Python Dictionary containing "linear_cache" and "activation_cache" ; 
                                    stored for computing the backward pass
    """
    
    # Linear forward propagation with Activation Function = SIGMOID
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    # Linear forward propagation with Activation Function = RELU
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    # Ensure the shape correctness of the output
    assert (A.shape == (A_prev.shape[0], W.shape[1]))
    
    # Store the variables needed for the backward pass in the cache
    cache = (linear_cache, activation_cache)
    
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:          X               Input Data ; (number of examples, input size)
                        parameters      Output of initialize_parameters_deep() ; ("W1", "b1", ..., "WL", "bL")
    
    Returns:            AL              Post-Activation Value
                        caches          list of caches containing:
                                        every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                                        the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    # List to store caches for each layer
    caches = []
    # Initialize input A as X
    A = X
    # Number of layers in the neural network
    L = len(parameters) // 2
    
    # Perform forward propagation for [LINEAR->RELU]*(L-1)
    for l in range(1, L):
        # Previous layer's output
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Perform forward propagation for the last layer with sigmoid activation
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    # Ensure the output shape is correct
    assert(AL.shape == (X.shape[0], 4))
    
    return AL, caches


def compute_cost(AL, Y):
    """
    Compute the Cost for all examples at the end of a iteration

    Arguments:      AL      Predicted Probability Vector    (number of examples , 4)
                    Y       True "Label" Vector             (number of examples , 4)

    Returns:        cost    Mean Squared Error
    """
    
    # Number of classes
    m = Y.shape[1]
    # Reshape AL to match the shape of Y
    AL = AL.reshape(Y.shape[0], 4)
    
    # List to store individual costs for each example
    costarray = []
    
    # Calculate the squared Euclidean distance between each predicted row and true row
    for rowAL, rowY in zip(AL, Y):
        rowcost = np.linalg.norm(rowAL-rowY)
        costarray = np.append(costarray, rowcost)
            
    # Compute the mean squared error (MSE) cost
    cost = (1/Y.shape[0]) * sum(costarray) 
    
    # Remove any unnecessary dimensions (e.g. [[17]] -> 17)
    cost = np.squeeze(cost)
    
    # Ensure the output shape is correct
    assert(cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:      dZ          Gradient of the cost with respect to the linear output (of current layer l)
                    cache       Tuple of Values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:        dA_prev     Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
                    dW          Gradient of the cost with respect to W (current layer l), same shape as W
                    db          Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache
    # Number of examples in the batch
    m = A_prev.shape[1]
        
    # Gradient of the cost with respect to W
    dW = 1./m * np.dot(A_prev.T,dZ)
    # Gradient of the cost with respect to b
    db = 1./m * np.sum(dZ, axis = 0, keepdims = True)
    # Gradient of the cost with respect to the activation of the previous layer (A_prev)
    dA_prev = np.dot(dZ, W.T)
    
    # Ensure the output shape is correct
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:      dA              post-activation gradient for current layer l 
                    cache           tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
                    activation      the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:        dA_prev         Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
                    dW              Gradient of the cost with respect to W (current layer l), same shape as W
                    db              Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        # Compute the gradient of the cost with respect to the linear output Z
        dZ = relu_backward(dA, activation_cache)  
        # Compute the gradients of the cost with respect to A_prev, W, and b
        dA_prev, dW, db = linear_backward(dZ, linear_cache) 
    
    elif activation == "sigmoid":
        # Compute the gradient of the cost with respect to the linear output Z
        dZ = sigmoid_backward(dA, activation_cache) 
        # Compute the gradients of the cost with respect to A_prev, W, and b
        dA_prev, dW, db = linear_backward(dZ, linear_cache) 
    
    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:      AL          Probability Vector ; Output of the Forward Propagation
                    Y           Rrue "Label" Vector ; Ground Truth
                    caches      list of caches containing:
                                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:        grads       A Dictionary with the Gradients
                                grads["dA" + str(l)] = ... 
                                grads["dW" + str(l)] = ...
                                grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches)  # Number of layers
    m = AL.shape[1]  # Number of examples
    Y = Y.reshape(AL.shape)  # Reshape Y to match the shape of AL
    
    # Compute the derivative of the cost with respect to AL
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]  # Retrieve the cache of the last layer
    # Perform backward propagation for the last layer (sigmoid activation)
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    
    # Loop through the remaining layers in reverse order
    for l in reversed(range(L-1)):
        current_cache = caches[l]  # Retrieve the cache of the current layer
        # Perform backward propagation for the current layer (ReLU activation)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp  # Store the gradients for the activation of the previous layer
        grads["dW" + str(l + 1)] = dW_temp  # Store the gradients for the weights of the current layer
        grads["db" + str(l + 1)] = db_temp  # Store the gradients for the biases of the current layer
    
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update the parameters of the neural network using gradient descent.

    Arguments:      parameters       A dictionary containing the parameters of the neural network.
                    grads            A dictionary containing the gradients of the parameters.
                    learning_rate    The learning rate for gradient descent.

    Returns:        parameters       The updated parameters after performing the gradient descent update.
    """

    L = len(parameters) // 2  # Number of layers in the neural network

    for l in range(L):
        # Update the weights and biases of each layer using gradient descent
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters


def predict(X, parameters, amountModuls = None):
    """
    Predict Norms for a given Powerarray Input
    
    Arguments:      X               Example you would like to label
                    parameters      Parameters of the trained model
                    amountModuls    Amount of PV Moduls in the Example Grid if available
    
    Returns:
    """
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    probas = [probas[0] / np.sum(probas[0])]
    
    gesamt = probas[0][0] + probas[0][1] + probas[0][2] + probas[0][3]

    if amountModuls == None:
        # Print the percentages of each norm in the grid
        terminal_width = shutil.get_terminal_size().columns
        text = "  PREDICTIONS  "
        padding = (terminal_width - len(text)) // 2
        print(Fore.GREEN + "#" * padding + text + "#" * padding)
        print('Percentage of \tVDE_AR_N_4105 \t(Rampfunc.)\t Norms in Grid:\t %f %%' %(probas[0][0]*100))
        print('Percentage of \tVDEW_2001 \t(50.5 Hz) \t Norms in Grid:\t %f %%' %(probas[0][1]*100))
        print('Percentage of \tDIN_V_VDE_V \t(50.2 Hz) \t Norms in Grid:\t %f %%' %(probas[0][2]*100))
        print('Percentage of \tSysStabV \t(Thresholds)\t Norms in Grid:\t %f %%' %(probas[0][3]*100))
        terminal_width = shutil.get_terminal_size().columns
        print('#' * terminal_width)
        return (
            probas[0][0] / gesamt,  # Percentage of VDE_AR_N_4105 (Rampfunc.) norms in the grid
            probas[0][1] / gesamt,  # Percentage of VDEW_2001 (50.5 Hz) norms in the grid
            probas[0][2] / gesamt,  # Percentage of DIN_V_VDE_V (50.2 Hz) norms in the grid
            probas[0][3] / gesamt,  # Percentage of SysStabV (Thresholds) norms in the grid
        )
    
    else:
        rounded_values = round_to_nrModuls(
            probas[0][0] * amountModuls,  # Number of modules with VDE_AR_N_4105 (Rampfunc.) norm in the grid
            probas[0][1] * amountModuls,  # Number of modules with VDEW_2001 (50.5 Hz) norm in the grid
            probas[0][2] * amountModuls,  # Number of modules with DIN_V_VDE_V (50.2 Hz) norm in the grid
            probas[0][3] * amountModuls,  # Number of modules with SysStabV (Threshold) norm in the grid
            amountModuls,  # Total amount of PV modules in the grid
        )
        init()

        terminal_width = shutil.get_terminal_size().columns
        text = "  PREDICTIONS  "
        padding = (terminal_width - len(text)) // 2
        print(Fore.GREEN + "#" * padding + text + "#" * padding)
        print('Moduls with \tVDE_AR_N_4105 \t(Rampfunc.)\t Norm in Grid:\t %i' %(rounded_values[0]))
        print('Moduls with \tVDEW_2001 \t(50.5 Hz)\t Norm in Grid:\t %i' %(rounded_values[1]))
        print('Moduls with \tDIN_V_VDE_V \t(50.2 Hz)\t Norm in Grid:\t %i' %(rounded_values[2]))
        print('Moduls with \tSysStabV \t(Threshold)\t Norm in Grid:\t %i' %(rounded_values[3]))
        terminal_width = shutil.get_terminal_size().columns
        print('#' * terminal_width)
        
        return rounded_values[0], rounded_values[1], rounded_values[2], rounded_values[3]



def metric(X, parameters, nr, pvnorm1, pvnorm2, pvnorm3, pvnorm4):
    """
    Prints the Predicted vs Ground Truth Norms of a given Test Set 
    
    Arguments:      X               Example you would like to label
                    parameters      Parameters of the trained model
                    nr              Number X of Test Example Set
                    pvnormX         Real amount of PV-Norms 1, 2, 3, 4 inside the Grid
    
    Returns:        errorate        Predictions for the Given Dataset X
    """
    
    probas, caches = L_model_forward(X, parameters)
    gesamt = sum(probas[0])
    amountModuls = pvnorm1 + pvnorm2 + pvnorm3 + pvnorm4
    
    error1 = int(abs(round(probas[0][0]/gesamt*amountModuls) - pvnorm1))
    error2 = int(abs(round(probas[0][1]/gesamt*amountModuls) - pvnorm2))
    error3 = int(abs(round(probas[0][2]/gesamt*amountModuls) - pvnorm3))
    error4 = int(abs(round(probas[0][3]/gesamt*amountModuls) - pvnorm4))

    # % of how many PV-Normes are predicted wrong
    errorrate = (
        (error1 + error2 + error3 + error4)
        / (
            max(probas[0][0] / gesamt * amountModuls, pvnorm1)
            + max(probas[0][1] / gesamt * amountModuls, pvnorm2)
            + max(probas[0][2] / gesamt * amountModuls, pvnorm3)
            + max(probas[0][3] / gesamt * amountModuls, pvnorm4)
        )
        * 100
    )

    print('Iteration: ', nr)
    print('\t\t Predicted \t Ground Truth')
    print('VDE_AR_N_4105 \t %i\t\t %i' %(round(probas[0][0]/gesamt*amountModuls), pvnorm1))
    print('VDEW_2001 \t %i\t\t %i' %(round(probas[0][1]/gesamt*amountModuls), pvnorm2))
    print('DIN_V_VDE_V \t %i\t\t %i' %(round((probas[0][2]/gesamt*amountModuls)), pvnorm3))
    print('SysStabV \t %i\t\t %i' %(round((probas[0][3]/gesamt*amountModuls)), pvnorm4))
    print('Precentage False Predicted: %f %% \n' %(errorrate))  
                
    return errorrate

    
def VDE_AR_N_4105(vde):
    """
    Simulates a single a VDE_AR_N_4105 Norm - Ramp Function
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a VDE_AR_N_4105 Norm with random Power ; len 1601
    """
    Pow = vde
    Hz = np.linspace(50, 51.6, 1601)

    # Calculate the power values for the VDE_AR_N_4105 Norm
    ywerte = [Pow - ((abs(x - 50.2) * 0.4) * Pow) if 50.2 <= x <= 51.5 else (Pow if x < 50.2 else 0) for x in Hz]

    return ywerte

def VDEW_2001(vdew):
    """
    Simulates a single a VDEW_2001 Norm - 50.5 Hz
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a VDEW_2001 Norm with random Power ; len 1601
    """
    
    Pow = vdew
    Hz = np.linspace(50, 51.6, 1601)
    
    # Calculate the power values for the VDEW_2001 Norm
    ywerte = [0 if x >= 50.5 else Pow for x in Hz]

    return ywerte

    
def DIN_V_VDE_V(din):
    """
    Simulates a single a DIN_V_VDE_V Norm - 50.2 Hz
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a DIN_V_VDE_V Norm with random Power ; len 1601
    """
    Pow = din
    Hz = np.linspace(50, 51.6, 1601)
    
    # Calculate the power values for the DIN_V_VDE_V Norm
    ywerte = [0 if x >= 50.2 else Pow for x in Hz]

    return ywerte


def SysStabV(sys):
    """
    Simulates a single a SysStabV Norm - Threshold
    
    Arguments:      
    
    Returns:        ywerte      Power Array for a SysStabV Norm with random Power ; len 1601
    """
    
    Pow = sys
    Hz = np.linspace(50, 51.6, 1601)
    
    # Calculate the power values for the SysStabV Norm
    ywerte = [1 * Pow if x < 50.25 else
              0.9375 * Pow if x < 50.3 else
              0.875 * Pow if x < 50.35 else
              0.8125 * Pow if x < 50.4 else
              0.75 * Pow if x < 50.45 else
              0.6875 * Pow if x < 50.5 else
              0.625 * Pow if x < 50.55 else
              0.5625 * Pow if x < 50.6 else
              0.5 * Pow if x < 50.65 else
              0.4375 * Pow if x < 50.7 else
              0.375 * Pow if x < 50.75 else
              0.3125 * Pow if x < 50.8 else
              0.25 * Pow if x < 50.85 else
              0.1875 * Pow if x < 50.9 else
              0.125 * Pow if x < 50.95 else
              0.0625 * Pow if x < 51 else
              0 for x in Hz]
    return ywerte


def plot_result(vde, vdew, din, sys):
    """
    Plots the combined power array from different simulation functions.
    
    Arguments:      vde     Random power value for VDE_AR_N_4105
                    vdew    Random power value for VDEW_2001
                    din     Random power value for DIN_V_VDE_V
                    sys     Random power value for SysStabV
        
    Returns:        None
    """
    
    y1 = np.array(VDEW_2001(vdew))
    y2 = np.array(VDE_AR_N_4105(vde))
    y3 = np.array(DIN_V_VDE_V(din))
    y4 = np.array(SysStabV(sys))
    ywerte = y1 + y2 + y3 + y4
    Hz = np.linspace(50, 51.6, 1601)
    
    plt.figure(figsize=(15, 8))
    plt.plot(Hz, ywerte, color='#e37222')
    plt.axis([min(Hz), max(Hz), min(ywerte), max(ywerte) + (max(ywerte) * 0.05)])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power from all PV Modules")
    plt.show()
    plt.close()
    
    
def solar(location, date, time):
    """
    Fetches weather data for a specific location, date, and time, and displays the temperature and solar irradiance.
    
    Arguments:      location        String representing the location.
                    date            String representing the date in the format 'YYYY-MM-DD'.
                    time            String representing the time in the format 'HH:MM'.
    
    Returns:        temperature     Air temperature in degrees Celsius.
                    irradiance      Ground-level solar irradiance in W/m².
    """

    # Geocoding service
    geolocator = geopy.Nominatim(user_agent="myGeocoder")
    location_obj = geolocator.geocode(location)

    if location_obj is None:
        print('Fehler beim Abrufen der Wetterdaten.')
        return None, None

    latitude = location_obj.latitude
    longitude = location_obj.longitude

    location = {
        'latitude': latitude,
        'longitude': longitude
    }

    # Get weather data
    temperature, irradiance = get_weather_data(date, time, location)

    # Display results
    if temperature is not None and irradiance is not None:
        
        terminal_width = shutil.get_terminal_size().columns
        text = "  POWER FROM PV  "
        padding = (terminal_width - len(text)) // 2
        print(Fore.YELLOW + "#" * padding + text + "#" * padding)
        print('Air Temperature:\t\t\t', temperature, '°C')
        print('Ground-level Solar Irradiance:\t\t', irradiance, 'W/m²')
    else:
        print('Fehler beim Abrufen der Wetterdaten.')

    return temperature, irradiance


def get_weather_data(date, time, location):
    """
    Fetches weather data from the Renewables.ninja API for a specific date, time, and location.
    
    Arguments:
        date: String representing the date in the format 'YYYY-MM-DD'.
        time: String representing the time in the format 'HH:MM'.
        location: Dictionary containing the latitude and longitude of the location.
    
    Returns:
        air_temperature: Air temperature in degrees Celsius.
        solar_irradiance: Solar irradiance in W/m².
    """
    token = 'c9ea714bc230eb778c665618ba5bab09ab65fc66'
    api_base = 'https://www.renewables.ninja/api/'

    session = requests.Session()
    session.headers = {'Authorization': 'Token ' + token}

    url = api_base + 'data/weather'

    # Prepare API request
    args = {
        'header': True,
        'lat': location['latitude'],
        'lon': location['longitude'],
        'date_from': date,
        'date_to': date,
        'dataset': 'merra2',
        'var_t2m': True,
        'var_swgdn': True,
        'format': 'json'
    }

    # Send API request
    response = session.get(url, params=args)

    if response.status_code == 200:
        data = json.loads(response.text)

        datetime_str = date + ' ' + time + ':00'
        datetime_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        millisec = int(datetime_object.timestamp() * 1000)

        if str(millisec) in data['data']:
            air_temperature = data['data'][str(millisec)]['t2m']
            solar_irradiance = data['data'][str(millisec)]['swgdn']

            return air_temperature, solar_irradiance
        else:
            print('Daten für den angegebenen Zeitpunkt nicht verfügbar.')
            return None, None
    else:
        print('Fehler beim Abrufen der Daten:', response.text)
        return None, None
    
    
def wirkungsgrad(temperature, irradiance):
    """
    Calculates the efficiency based on temperature and solar irradiance.
    
    Arguments:      temperature     Air temperature in degrees Celsius.
                    irradiance      Solar irradiance in W/m².
    
    Returns:        wg              Efficiency as a decimal value.
    """
    
    if 0 < irradiance <= 100:
        wirkungsgrad1 = 0.0
    elif 100 < irradiance <= 300:
        wirkungsgrad1 = 0.02
    elif 300 < irradiance <= 500:
        wirkungsgrad1 = 0.4
    elif 500 < irradiance <= 700:
        wirkungsgrad1 = 0.6
    elif 700 < irradiance <= 900:
        wirkungsgrad1 = 0.8
    elif 900 < irradiance <= 1100:
        wirkungsgrad1 = 1.0
    else:
        wirkungsgrad1 = 1.2
    
    wirkungsgrad = 100 - (abs(25 - temperature) * 0.4)
    wg = (wirkungsgrad / 100) * wirkungsgrad1

    if wg < 0.02:
        wg = 0.02
    
    print('Wirkungsgrad beträgt:\t\t\t',wg)
    return wg

def round_to_nrModuls(a, b, c ,d, nrModuls):
    """
    Calculates the real amount of Norms inside a Grid, using the predicted probabilities 
    
    Arguments:      a                   predicted probability for a 
                    b                   predicted probability for b 
                    c                   predicted probability for c 
                    d                   predicted probability for d 
                    nrModuls            amount of moduls inside the grid
    
    Returns:        rounded_values      
    """
    
    rounded_values = np.round([a, b, c ,d]).astype(int)

    diff = nrModuls - np.sum(rounded_values)

    if diff == 0:
        return rounded_values
    if diff > 0:
        decimal_parts = np.abs([val % 1 - 0.5 for val in [a, b, c ,d]])
        indices = np.where(decimal_parts == np.min(decimal_parts))[0]
        rounded_values[indices[0]] += diff
        return rounded_values
        
    elif diff < 0:
        # Die gerundete Summe ist zu hoch, verringere den Wert mit der Nachkommastelle am nächsten an 0.5, aber darunter
        decimal_parts = np.abs([val % 1 - 0.5 for val in [a, b, c ,d]])
        indices = np.where(decimal_parts == np.min(decimal_parts))[0]
        rounded_values[indices[0]] += diff
        return rounded_values
    
    
def load(location, date, time):
    """
    Calculates the estimated load inside a grid with the amounts of Housholds H0 /Businesses G0 /Agricultures L0
    
    Arguments:      a                   predicted probability for a 
                    b                   predicted probability for b 
                    c                   predicted probability for c 
                    d                   predicted probability for d 
                    nrModuls            amount of moduls inside the grid
    
    Returns:        rounded_values      
    """

    eingabe_datum = datetime.strptime(date, '%Y-%m-%d')
    eingabe_zeit = datetime.strptime(time, '%H:%M')
    eingabe_zeit = pd.to_datetime(eingabe_zeit).round('15min').strftime('%H:%M')

    winter_start = "01-01"
    winter_end = "03-20"
    ubergang1_start = "03-21"
    ubergang1_end = "05-14"
    sommer_start = "05-15"
    sommer_end = "09-14"
    ubergang2_start = "09-15"
    ubergang2_end = "10-31"
    winter2_start = "11-01"
    winter2_end = "12-31"

    input_month_day = date[5:]

    if winter_start <= input_month_day <= winter_end:
        jahreszeit = "Winter"
    elif ubergang1_start <= input_month_day <= ubergang1_end:
        jahreszeit = "Übergangszeit"
    elif sommer_start <= input_month_day <= sommer_end:
        jahreszeit = "Sommer"
    elif ubergang2_start <= input_month_day <= ubergang2_end:
        jahreszeit = "Übergangszeit"
    elif winter2_start <= input_month_day <= winter2_end:
        jahreszeit = "Winter"
    else:
        print("Invalid date or range not found")

    wochentag = eingabe_datum.weekday()

    if wochentag < 5:
        day = "Werktag"
    elif wochentag == 5:
        day = "Samstag"
    else:
        day = "Sonntag"

    # Pfad zur Excel-Datei angeben
    haushalt = 'Load/Lastprofil_Haushalt_H0.xlsx'
    gewerbe = 'Load/Lastprofil_Gewerbe_G0.xlsx'
    landwirtschaft = 'Load/Lastprofil_Landwirtschaft_L0.xlsx'

    formatted_time = eingabe_zeit + ":00"
    
    dfhaushalt = pd.read_excel(haushalt, parse_dates=['Zeit'])
    desired_row = dfhaushalt[dfhaushalt['Zeit'] == formatted_time]
    valuehaushalt = desired_row[(jahreszeit + day)].values[0]
    
    
    dfgewerbe = pd.read_excel(gewerbe, parse_dates=['Zeit'])
    desired_row = dfgewerbe[dfgewerbe['Zeit'] == formatted_time]
    valuegewerbe = desired_row[(jahreszeit + day)].values[0]
    

    dflandwirtschaft = pd.read_excel(landwirtschaft, parse_dates=['Zeit'])
    desired_row = dflandwirtschaft[dflandwirtschaft['Zeit'] == formatted_time]
    valuelandwirtschaft = desired_row[(jahreszeit + day)].values[0]
        
    # Pfad zur Excel-Datei angeben
    anschlussdaten = 'Anlagendaten/Anschlussdaten.xlsx'

    # Eintrag auslesen
    df = pd.read_excel(anschlussdaten)
    
    H0 = int(df.loc[df['Ort'] == location, 'Anzahl_H0'].values[0])
    G0 = int(df.loc[df['Ort'] == location, 'Anzahl_G0'].values[0])
    L0 = int(df.loc[df['Ort'] == location, 'Anzahl_L0'].values[0])
    
    
    terminal_width = shutil.get_terminal_size().columns
    text = "  LOAD IN GRID  "
    padding = (terminal_width - len(text)) // 2
    print(Fore.BLUE + "#" * padding + text + "#" * padding)
    print('Haushalte im Netz:\t\t\t', H0)
    print('Landwirschaftliche Betriebe im Netz:\t', L0)
    print('Gewerbe im Netz:\t\t\t', G0)
    
    load = H0 * valuehaushalt + G0 * valuegewerbe + L0 * valuelandwirtschaft
    
    print('Gesamte geschätzte Last:\t\t', load/1000, '[kW]')
    terminal_width = shutil.get_terminal_size().columns
    print('#' * terminal_width)
    return load


def plot_result_with_load(vde, vdew, din, sys, load):
    """
    Plots the combined power array from different simulation functions.
    
    Arguments:      vde     Random power value for VDE_AR_N_4105
                    vdew    Random power value for VDEW_2001
                    din     Random power value for DIN_V_VDE_V
                    sys     Random power value for SysStabV
        
    Returns:        None
    """
    
    y1 = np.array(VDEW_2001(vdew))
    y2 = np.array(VDE_AR_N_4105(vde))
    y3 = np.array(DIN_V_VDE_V(din))
    y4 = np.array(SysStabV(sys))
    ywerte = (y1 + y2 + y3 + y4) * 1000
    ywerte = load - ywerte
    
    Hz = np.linspace(50, 51.6, 1601)
    
    plt.figure(figsize=(15, 8))
    plt.plot(Hz, ywerte, color='#e37222')
    plt.axis([min(Hz), max(Hz), min(ywerte), max(ywerte)])
    plt.show()
    plt.close()
    
    
def calculate_power(parameterfile, powerarrayfile):
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

    rows = dataframe.loc[dataframe['Ort'] == powerarrayfile[:-8]]
    nrModuls = len(rows)
    vde, vdew, din, sys = predict(powerarray2, parameters, nrModuls)

    vdeyears = (rows['Jahr der Anmeldung'].nlargest(vde)).index
    if vde != 0 and int(min(rows.loc[vdeyears, 'Jahr der Anmeldung'])) < 2009:
        print('Number of predicted Rampfunc. do not agree with the IBN from Anlagedaten. Take the mean Power of all PV-Moduls')
        years = (rows['Jahr der Anmeldung']).index
        sum_power = sum(rows.loc[years, 'Generator Leistung'])
        vdepower, dinpower, syspower, vdewpower = calculate_mean_power(sum_power, nrModuls, vde, vdew, din, sys)
        return vdepower, dinpower, syspower, vdewpower
    else: 
        vdepower = sum(rows.loc[vdeyears, 'Generator Leistung'])        
        rows = rows.drop(vdeyears)
        
    dinyears = (rows['Jahr der Anmeldung'].nlargest(din)).index
    if din != 0 and int(min(rows.loc[dinyears, 'Jahr der Anmeldung'])) < 2006:
        print('Number of predicted 50.2 Hz do not agree with the IBN from Anlagedaten. Take the mean Power of all PV-Moduls')
        years = (rows['Jahr der Anmeldung']).index
        sum_power = sum(rows.loc[years, 'Generator Leistung'])
        vdepower, dinpower, syspower, vdewpower = calculate_mean_power(sum_power, nrModuls, vde, vdew, din, sys)
        return vdepower, dinpower, syspower, vdewpower
    else: 
        dinpower = sum(rows.loc[dinyears, 'Generator Leistung'])
        rows = rows.drop(dinyears)

    sysyears = (rows['Jahr der Anmeldung'].nlargest(sys)).index
    if sys != 0 and int(min(rows.loc[sysyears, 'Jahr der Anmeldung'])) < 2001:
        print('Number of predicted SysStab do not agree with the IBN from Anlagedaten. Take the mean Power of all PV-Moduls')
        years = (rows['Jahr der Anmeldung']).index
        sum_power = sum(rows.loc[years, 'Generator Leistung'])
        vdepower, dinpower, syspower, vdewpower = calculate_mean_power(sum_power, nrModuls, vde, vdew, din, sys)
        return vdepower, dinpower, syspower, vdewpower
    else: 
        syspower = sum(rows.loc[sysyears, 'Generator Leistung'])
        rows = rows.drop(sysyears)

    vdewyears = (rows['Jahr der Anmeldung'].nlargest(vdew)).index
    vdewpower = sum(rows.loc[vdewyears, 'Generator Leistung'])
    rows = rows.drop(vdewyears)
    
    return vdepower, dinpower, syspower, vdewpower


def calculate_mean_power(sum_power, nrModuls, vde, vdew, din, sys):
    if vde != 0:
        vdepower = sum_power / nrModuls * vde
    else:
        vdepower = 0
    
    if din != 0:
        dinpower = sum_power / nrModuls * din
    else:
        dinpower = 0
    
    if sys != 0:
        syspower = sum_power / nrModuls * sys
    else:
        syspower = 0
    
    if vdew != 0:
        vdewpower = sum_power / nrModuls * vdew
    else:
        vdewpower = 0
 
    return vdepower, dinpower, syspower, vdewpower