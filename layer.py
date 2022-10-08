import numpy as np

class Layer:
        def __init__(self, input_num, output_num, activation, activation_deriv, a):
                self.input = input_num 
                self.output = output_num
                self.alpha = a
                self.g = activation
                self.gDerivative = activation_deriv

                self.initWeights(self.input, self.output)
                self.initBias(self.output)

        def initWeights(self, xsize, ysize):
                self.W = np.random.rand(xsize, ysize)

        def initBias(self, size):
                self.b = np.random.rand(1, size)

        def forwardprop(self, x):
                self.X = x
                self.z1= np.dot(self.X, self.W) + self.b #Grabs the input prior to the activation function for backprop
                return self.g(self.z1) #Apply activation function
        
        def backprop(self, dz2):
                dz1 = self.gDerivative(self.z1) * dz2
                
                Xerror = np.dot(dz1, self.W.T) 
                Werror = np.dot(self.X.T, dz1)

		            #Gradient Descent - Update Rules
                self.W  = self.W - self.alpha * Werror
                self.b = self.b - self.alpha * dz1

                return Xerror

if __name__ == '__main__':
        print('Entered the main function of the Layer object.')