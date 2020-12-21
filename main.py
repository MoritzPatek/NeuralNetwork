#so as an absolut beginner when it comes to neural networks i tought i just firstly write my own 
#very simple one and then work myself to the point i want to be at

#so what is a neural network actually?
#those words are really sounding more complicated then they actually are

#basically a neural network consists out of inputs, hidden layers, outputs and the weights (don't accept a fat mom joke now)
#but she kinda thicc tho 

#the inputs reseave data from what ever, that data can come from sensors or images, it just has to be readable for the machine
#so its a must to somehow convert the real data to ones and zeros

#just an import that helps us with Vectors and matrix stuff
import numpy as np

#this function returns always a value between 0 and 1
def sigmoid(x):
    #np.exp is probably the main part
    #it takes the mathematical constant Euler's number
    #and calculates like: e^x
    #the special thing about np.exp is that its taking your input array
    #and calculates the value for every row of your array
    return 1 / (1 + np.exp(-x))


#this funtction is used to calculate the adjustment for the weights
def sigmoid_derivative(x):
    return x * (1 - x)

#just inputs so the network has data to train 
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

#those are the outputs the neural network should get too
#the problem with the outputs is that regardles of the amounts of interation 
#and changes of the weights it will never REALLY reach those values
training_outputs = np.array([[0,1,1,0]]).T

#a specific seed for random numbers
np.random.seed(1)

#init of the weights with random values in the matrix
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)

#interates 10.000 times
for iteration in range(10000):

    input_layer = training_inputs
    #np.dot returns the product of two arrays 
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    #error is the difference between the training outputs and the calc. outputs
    error = training_outputs - outputs
    #the difference gets multiplicated by the derivativ of the outputs 
    adjustments = error * sigmoid_derivative(outputs)

    #the weights get their new velues by calculating the product of the 
    #input_layers and the adjustments  
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)

#and thats already it, my first neural network with all the stuff needed for an bigger model
#if you have any questions or found a mistake pls contact me :)
