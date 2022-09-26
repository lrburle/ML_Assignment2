import numpy as np
import matplotlib.pyplot as plt
import csv
class assign2:
        def __init__(self):
                self.a = 0.3 #Learning rate

                #Basis Function Variables
                self.mu = 0 #Mean Value
                self.phi = 1 #Variance
        
        def init_theta(self, m, n):
                self.theta = np.random.random((m, n))

        def hypothesis(self, xdata):
                y = np.dot(self.theta, xdata)
                return y

        def costFunction(self, xdata, ydata):
                [m, n] = xdata.shape

                sum = 0
                for i in range(m):
                        sum += (self.hypothesis(xdata[i])-ydata[i])**2
                costf = (1/m) * sum

                return costf

        def gradientDescent(self, xdata, ydata):
                [m, n] = xdata.shape

                theta_new = self.theta.copy()

                for j in range(n):
                        sum = 0 
                        for i in range(m):
                                sum += (self.hypothesis(xdata[i]) - ydata[i]) * xdata[i, j]

                        theta_new[0, j] = self.theta[0, j] - self.a * (1 / m) * sum
                
                self.theta = theta_new 

	#Used to modifify the input data appropriately
        def sigmoidBasis(self, data):
                [m,n] = data.shape
                out = np.zeros((m, 1))
                out[:,0] = np.exp(((data[:, 0] - self.mu) ** 2) / (2*self.phi**2)) 
                return np.concatenate((data, out), axis=1)

        def polynomialBasis(self, data, order):
                [m,n] = data.shape
                out = np.zeros((m, 1))
                out[:,0] = data[:, 0] ** order
                return np.concatenate((data, out), axis=1) 
        
        def concatOnes(self, data):
                [m,n] = data.shape
                out = np.ones((m, 1))
                return np.concatenate((out, data), axis=1)
        
        #Parses the csv data for question 2
        def csvParse(self, csvFile):
                file = open(csvFile, 'r')
                #Grabbing the header names for each column
                reader = csv.DictReader(file)
                header = reader.fieldnames
                header = header[1:-1]
                file = file.readlines()[0:]
                data = np.loadtxt(file, delimiter=',')

                [x,y] = data.shape

                x_train = data[0:(x//2), 1:(y-1)]
                y_train = data[0:(x//2), (y-1)]
                x_test = data[(x//2):x, 1:(y-1)]
                y_test = data[(x//2):x, (y-1)]
                
                return x_train, y_train, x_test, y_test, header

	#Selects a feature set from the question 2 dataset
        def featureSelection(self, data, columns):
                [m,n] = data.shape

                for i in columns:
                        if i == 0:
                                data_out = data[:, 0].reshape((m, 1)) 
                        else:
                                data_out = np.concatenate((data_out, data[:, i].reshape((m, 1))), axis=1)

                return data_out
        
	#Local weight algorithms used for Question 3
        def normalEquations(self, X, W, Y):
                self.theta = ((X.T*(W * X)).I*(X.T*W*Y.T))

        def localWeights(self, query, tao, x_train):
                [m, n] = x_train.shape
                w = np.mat(np.eye(m))
                
                for i in range(m):
                        w[i, i]  = np.exp(((x_train[i] - query) * (x_train[i] - query).T) / (-2*tao**2))

                return w 

        def costFunctionLR(self):
                [m, n] = self.w.shape

                for i in range(m):
                       self.costfLR += self.w[i] * (self.theta.T * self.x_train[i] - self.y_train[i])
                
                self.constfLR = 0.5 * self.constfLR 
        
        def hypothesisDataGeneration(self, lowerbounds, upperbounds, order):
                xdata = np.linspace(lowerbounds, upperbounds, 1000)
                data = xdata 

                data = data.reshape((1000, 1))

                for i in range(2, order+1):
                        data = self.polynomialBasis(data, i)

                data = self.concatOnes(data) #Input data with the order desired to be concatenated into the original dataset.

                h = []

                for i in range(1000):
                        h.append(self.hypothesis(data[i]))

                return xdata, h

if __name__ == '__main__':
        a2 = assign2()

        #Load in the data to be used for question 1
        x_test = np.load('x_test.csv')
        x_train = np.load('x_train.csv')
        y_test = np.load('y_test.csv')
        y_train = np.load('y_train.csv')

        #Plot the initial training data and hypothesis function.
        plt.figure(0, figsize=[15, 12])
        plt.xlabel('x', fontsize=18)
        plt.ylabel('y', fontsize=18)
        plt.title(f'Q1 - Train Data Set - No Basis Functions, ' + r'$\alpha$ = ' + f'{a1.a}', fontsize=24)
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
                a1.gradientDescent(x_train, y_train)
                x, h = a1.hypothesisDataGeneration(np.min(x_train[:, 1]),np.max(x_train[:,1]), order)
                error.append(a1.costFunction(x_train, y_train))
                epsilon = np.abs(error[-1] - error[-2])
                hypth.set_ydata(h)
                plt.draw()
                if (epsilon < 10e-6):
                        print('Convergence threshold met.')
                        break
