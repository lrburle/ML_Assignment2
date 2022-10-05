import numpy as np

class Layer:
        def __init__(self, input_num, output_num, activation, a):
                self.input = input_num 
                self.output = output_num
                self.alpha = a
                self.g = activation

                self.initWeights(self.input, self.output)
                self.initBias(self.output)

        def initWeights(self, xsize, ysize):
                self.W = np.random.rand(xsize, ysize)

        def initBias(self, size):
                self.b = np.random.rand(1, size)

        def forwardprop(self, x):
                self.X = x
                out = np.dot(self.X, self.W) + self.b
                return self.g(out) #Apply activation function
        
        def backprop(self, error):
                Xerror = np.dot(error, self.W.T) 
                Werror = np.dot(self.X.T, error)

                self.W -= self.alpha * Werror
                self.b -= self.alpha * error

if __name__ == '__main__':
        print('Entered the main function of the Layer object.')