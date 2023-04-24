import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import sys


def sigmoid(Z):

    """
    Implement Sigmoid Activation Function
    
    Arguments:      Z           Numpy Array of any Shape
    
    Returns:        A           Output of sigmoid(z), same Shape as Z
                    cache       returns Z ; stored for computing the Backward Pass
    """

    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Implement the RELU Function

    Arguments:      Z           Output of the linear layer, of any shape

    Returns:        A           Output of Relu(z), same Shape as Z
                    cache       returns A ; stored for computing the Backward Pass
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the Backward Propagation for a single RELU unit

    Arguments:      dA          post-activation gradient, of any shape
                    cache       Z ; which we store for computing backward propagation

    Returns:        dZ          Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit

    Arguments:      dA          post-activation gradient, of any shape
                    cache       Z ; which we store for computing backward propagation

    Returns:        dZ          Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def load_data_train(trainset):
    """
    Loading the data from the h5 file for Training

    Returns:        train_set_x_orig        Powerarray from all Example Grids ; used for training
                    train_set_y_orig        Norms for all Example Grids ; used for training
                    classes                 Norm Names ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV'] 
    """
    trainset = 'h5/' + trainset
    
    train_dataset = h5py.File(trainset, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    classes = np.array(train_dataset["list_classes"][:]) # the list of classes
    
    c = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]*train_set_y_orig.shape[1]))
    
    return train_set_x_orig, train_set_y_orig, classes

def load_data_test(testset):
    """
    Loading the data from the h5 file for Testing

    Returns:        test_set_x_orig         Powerarray from all Example Grids ; used for testing
                    test_set_y_orig         Norms for all Example Grids ; used for testing
                    test_set_pv_norms       Amount of Norms for all Exampele Grids ; used for testing
                    classes                 Norm Names ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV'] 
    """
    testset = 'h5/' + testset
    
    test_dataset = h5py.File(testset, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    test_set_pv_norms = np.array(test_dataset["test_set_nrgesamt"][:]) # amount PV Norms inside Grid

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
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
    L = len(layer_dims)            # number of layers in the network == 7 (1601 - 200 - 100 - 50 - 25 - 12 - 4)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        parameters['W' + str(l)] = np.reshape(parameters['W' + str(l)], (((parameters['W' + str(l)]).shape)[1],((parameters['W' + str(l)]).shape)[0]))
        parameters['b' + str(l)] = np.reshape(parameters['b' + str(l)], (((parameters['b' + str(l)]).shape)[1],((parameters['b' + str(l)]).shape)[0]))
        
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
    
    Z = A.dot(W) + b

    assert(Z.shape == (A.shape[0], W.shape[1]))
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
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    assert (A.shape == (A_prev.shape[0], W.shape[1]))
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

    caches = []
    A = X
    #   Number of Layers inside Network
    L = len(parameters) // 2
    
    #   Forward Propagation with Relu Function for all Layers but the first One
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    #   Forward Propagation with Sigmoid Function for the first Layer
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (X.shape[0], 4))     
    return AL, caches


def compute_cost(AL, Y):
    """
    Compute the Cost for all examples at the end of a iteration

    Arguments:      AL      Predicted Probability Vector    (number of examples , 4)
                    Y       True "Label" Vector             (number of examples , 4)

    Returns:        cost    Mean Squared Error
    """
    
    m = Y.shape[1]
    AL = AL.reshape(Y.shape[0], 4)
    
    costarray = []
    
    for rowAL, rowY in zip(AL, Y):
        #   sqrt( (a1 - b1)^2 + (a2 - b2)^2 + (a3 - b3)^2 + (a4 - b4)^2 ) 
        rowcost = np.linalg.norm(rowAL-rowY)
        costarray = np.append(costarray, rowcost)
            
    cost = (1/Y.shape[0]) * sum(costarray) 
    
    #   E.g. Turns [[17]] into 17
    cost = np.squeeze(cost)
    
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
    m = A_prev.shape[1]
        
    dW = 1./m * np.dot(A_prev.T,dZ)
    db = 1./m * np.sum(dZ, axis = 0, keepdims = True)
    
    dA_prev = np.dot(dZ, W.T)
    
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
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
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
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # number of layers in the neural network
    for l in range(L):
        parameters["W" + str(l+1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters["b" + str(l+1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def predict(X, parameters, amountModuls = None):
    """
    Predict Norms for a given Powerarray Input
    
    Arguments:      X               Example you would like to label
                    parameters      Parameters of the trained model
                    amountModuls    Amount of PV Moduls in the Example Grid if available
    
    Returns:        p               Predictions for the given dataset X
    """
    
    m = 1
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)
    
    gesamt = probas[0][0] + probas[0][1] + probas[0][2] + probas[0][3]
    
    # classes = ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV']
    if amountModuls == None:
        print('Percentage of \tVDE_AR_N_4105 \t(Rampfunc.)\t Norms in Grid:\t %f %%' %(probas[0][0]/gesamt *100))
        print('Percentage of \tVDEW_2001 \t(50.5 Hz) \t Norms in Grid:\t %f %%' %(probas[0][1]/gesamt *100))
        print('Percentage of \tDIN_V_VDE_V \t(50.2 Hz) \t Norms in Grid:\t %f %%' %(probas[0][2]/gesamt *100))
        print('Percentage of \tSysStabV \t(Threshold)\t Norms in Grid:\t %f %%' %(probas[0][3]/gesamt *100))
    else:
        print('Moduls with \tVDE_AR_N_4105 \t(Rampfunc.)\t Norm in Grid:\t %i' %(round(probas[0][0]/gesamt*amountModuls)))
        print('Moduls with \tVDEW_2001 \t(50.5 Hz)\t Norm in Grid:\t %i' %(round(probas[0][1]/gesamt*amountModuls)))
        print('Moduls with \tDIN_V_VDE_V \t(50.2 Hz)\t Norm in Grid:\t %i' %(round((probas[0][2]/gesamt*amountModuls))))
        print('Moduls with \tSysStabV \t(Threshold)\t Norm in Grid:\t %i' %(round((probas[0][3]/gesamt*amountModuls))))
    return p


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
    gesamt = probas[0][0] + probas[0][1] + probas[0][2] + probas[0][3]
    
    amountModuls = pvnorm1 + pvnorm2 + pvnorm3 + pvnorm4
    
    error1 = abs(round(probas[0][0]/gesamt*amountModuls) - pvnorm1)
    error2 = abs(round(probas[0][1]/gesamt*amountModuls) - pvnorm2)
    error3 = abs(round(probas[0][2]/gesamt*amountModuls) - pvnorm3)
    error4 = abs(round(probas[0][3]/gesamt*amountModuls) - pvnorm4)

    errorrate = (((error1 + error2 + error3 + error4)/amountModuls)*100)

    errorVDE = round(abs(probas[0][0]/gesamt*amountModuls-pvnorm1))
    errorVDEW = round(abs(probas[0][1]/gesamt*amountModuls-pvnorm2))
    errorDIN = round(abs(probas[0][2]/gesamt*amountModuls-pvnorm3))
    errorSys = round(abs(probas[0][3]/gesamt*amountModuls-pvnorm4))
    
    print('Iteration:', nr)
    print('\t\t Predicted \t Ground Truth')
    print('VDE_AR_N_4105 \t %i\t\t %i' %(round(probas[0][0]/gesamt*amountModuls), pvnorm1))
    print('VDEW_2001 \t %i\t\t %i' %(round(probas[0][1]/gesamt*amountModuls), pvnorm2))
    print('DIN_V_VDE_V \t %i\t\t %i' %(round((probas[0][2]/gesamt*amountModuls)), pvnorm3))
    print('SysStabV \t %i\t\t %i' %(round((probas[0][3]/gesamt*amountModuls)), pvnorm4))


    print('False Predicted: %i / %i' %((error1 + error2 + error3 + error4), amountModuls))
    print('Precentage False Predicted: %f %% \n' %(errorrate))
                
    nr = nr-1
    return errorrate, errorVDE, errorVDEW, errorDIN, errorSys

def read_h5():
    # test = h5py.File('datasets/test_pvnorms.h5', 'w')
    # train = h5py.File('datasets/train_pvnorms.h5', 'w')

    # classes = ['VDE_AR_N_4105', 'VDEW_2001', 'DIN_V_VDE_V', 'SysStabV']
    # d2 = np.random.random(size = (1000,200))

    # test.create_dataset('list_classes', data = classes)
    # test.create_dataset('train_set_x', data = d2)
    # test.create_dataset('train_set_y', data = d2)

    f = h5py.File('datasets/train_pvnorms.h5', 'r') 
    print (list(f))

    print (f['list_classes'].shape) # images
    print (f['train_set_x'].shape) # classes 
    print (f['train_set_y'].shape) 

    g = open('error','w')

    np.set_printoptions(threshold=sys.maxsize)

    itere = 0
    for i in f['train_set_x'][:]:
        itere = itere + 1 
        print(i, file=g)