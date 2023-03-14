import pandas as pd
import numpy as np
from random import randint, random
import matplotlib.pyplot as plt

def Linear_Regression():
    # Read the CSV file into a pandas DataFrame
    filename = input('Enter filename: \n> ')
    data = pd.read_csv(filename)

    # Determine the number of input variables
    n = len(data.columns) - 1
    x, y = [], []
    # INPUT
    if n == 1:
        # SIMPLE LR
        x = data.iloc[:, 0].values.reshape(-1, 1)  # input variable as a column vector
        y = data.iloc[:, 1].values.reshape(-1, 1)  # output variable as a column vector
    else:
        # MULTIPLE LR
        x = data.iloc[:, :-1].values  # input variables as a matrix
        y = data.iloc[:, -1].values.reshape(-1, 1)  # output variable as a column vector

    # PROCESSING
    # Add a column of ones to X to represent the intercept term
    x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

    # Compute the optimal weights using linear regression
    weights = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    # OUTPUT
    if n > 1:
        # MILTIPLE LR
        # Print the optimal weights
        print('\nOptimal weights:')
        for i in range(n + 1):
            print(f'{data.columns[i]} = {weights[i][0]}')
    else:
        # SIMPLE LR
        print(f'\nOptimal regression line: y = {weights[1][0]} * x + {weights[0][0]}')
        # Plot the data and the optimal regression line
        plt.scatter(x[:, 1], y)
        plt.plot(x[:, 1], x.dot(weights), color='red')
        plt.xlabel('Input variable')
        plt.ylabel('Output variable')
        plt.show()

def create_testfile(n: int, rg_type: str='simple'):
    rg_type = rg_type.lower()
    with open('rg_input.csv', 'w') as f:
        if rg_type == 'simple':
            f.write('x,y\n')
            for i in range(n):
                f.write(f'{randint(19,83)},{randint(15,36) + random()}\n')
            return
        
        if rg_type == 'multiple':
            f.write('x,y,z\n')
            for i in range(n):
                f.write(f'{randint(19,83)},{randint(15,36) + random()},{randint(120,192)}\n')
            return
                
        print('error creatnig file, invalid rg_type\n')
