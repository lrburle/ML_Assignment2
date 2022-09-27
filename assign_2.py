import numpy as np
import matplotlib.pyplot as plt
import csv
class assign2:
        def __init__(self):
                self.a = 0.3 #Learning rate

        def init_theta(self, m, n):
                self.theta = np.random.random((m, n))

        # g(w0 + sum(x*w))
        def perceptron(self, x, w):
                return self.sigmoid(np.dot(x, w))
        
        def hiddenLayer(self, x, w, number_of_neurons):
                y = []
                for i in range(number_of_neurons):
                        y.append(self.perceptron(x, w))
                return y
                
        
        def backprop(self, ):


        def hypothesis(self, x, w):
                y = np.dot(x, w)
                return y

        def costFunction(self, xdata, ydata):
                [m, n] = xdata.shape

                sum = 0
                for i in range(m):
                        sum += (self.hypothesis(xdata[i])-ydata[i])**2
                costf = (1/m) * sum

                return costf

	#Used to modifify the input data appropriately
        def sigmoid(self, z):
                return 1 / (1 + np.exp(-z))

        def sigmoidDerivative(self, z):
                return self.sigmoid(z) * (1 - self.sigmoid(z))

        def gradientDescent(self, xdata, ydata):
                [m, n] = xdata.shape

                theta_new = self.theta.copy()

                for j in range(n):
                        sum = 0 
                        for i in range(m):
                                sum += (self.hypothesis(xdata[i]) - ydata[i]) * xdata[i, j]

                        theta_new[0, j] = self.theta[0, j] - self.a * (1 / m) * sum
                
                self.theta = theta_new 


        def concatOnes(self, data):
                [m,n] = data.shape
                out = np.ones((m, 1))
                return np.concatenate((out, data), axis=1)
        
if __name__ == '__main__':
        a2 = assign2()

        #Load in the data to be used for question 1
        x_test = np.load('X_test.csv')
        x_train = np.load('X_train.csv')
        y_test = np.load('Y_test.csv')
        y_train = np.load('Y_train.csv')

        #Plot the initial training data and hypothesis function.
        plt.figure(0, figsize=[15, 12])
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        plt.title(f'Q2 - Data Set - No Basis Functions, ' + r'$\alpha$ = ' + f'{a1.a}', fontsize=24)
        plt.ylim(-50, 70)
        og, hypth = plt.plot(x_train[:, 1], y_train, 'o', x, h, 'r-')

        hypth.set_label('Hypothesis')
        og.set_label('Training Data Set')
        plt.legend(loc='lower right')
        plt.grid()

        print(f'Initialzed theta array is: {a1.theta}')

        iterations = 100
        for k in range(iterations):
                print(f'Current iteration is {k} @ error = {error[-1]}')

                a2.gradientDescent(x_train, y_train)
                x, h = a2.hypothesisDataGeneration(np.min(x_train[:, 1]),np.max(x_train[:,1]), order)
                error.append(a2.costFunction(x_train, y_train))
                epsilon = np.abs(error[-1] - error[-2])
                hypth.set_ydata(h)
                plt.draw()
                if (epsilon < 10e-6):
                        print('Convergence threshold met.')
                        break
