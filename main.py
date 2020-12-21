#so as an absolut beginner when it comes to neural networks i tought i just firstly write my own 
#very simple one and then work myself to the point i want to be at

#so what is a neural network actually?
#those words are really sounding more complicated then they actually are

#basically a neural network consists out of inputs, hidden layers, outputs and the weights (don't accept a fat mom joke now)
#but she kinda thicc tho 

#the inputs reseave data from what ever, that data can come from sensors or images, it just has to be readable for the machine
#so its a must to somehow convert the real data to ones and zeros

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_outputs = np.array([[0,0,1],
                             [1,1,1],
                             [1,0,1],
                             [0,1,1]])

training_inputs = np.array([[0,1,1,0]]).T

np.random.seed(1)



