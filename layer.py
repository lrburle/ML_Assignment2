import numpy as np

class Layer:
        def __init__(self, input_num, output_num, activation):
                self.input = input_num 
                self.output = output_num
                self.g = activation

                self.initWeights(self.input, self.output)
                self.initBias(self.output)

        def initWeights(self, xsize, ysize):
                self.W = np.random.rand(xsize, ysize)

        def initBias(self, size):
                self.b = np.random.rand(1, size)

        # g(b + sum(x*w))
        def hiddenLayer(self, x, w, number_of_neurons):

                y = []
                for i in range(number_of_neurons):
                        y.append(self.perceptron(x, w))
                return y

if __name__ == '__main__':
        print('Entered the main function of the Layer object.')