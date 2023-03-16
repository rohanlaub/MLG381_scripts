import pandas as pd
import numpy as np
from random import randint

def Naive_Bayes() -> None:
    # X : tuple() of n dimension -> all but final column
    # C -> Final column (aka classifier)
    # i -> unique values in C
    # n -> amnt of attributes (columns other than C)
    
    # READ DATA
    data = pd.read_csv(input('Enter filename:\n> '))
    print('reading data...\n')

    # Assign data
    Headings = list(data.keys())
    X_data = data.loc[:, Headings[:-1]]
    C = data.iloc[:, -1].values.reshape(-1, 1)

    print(data)

    # Accept input for tuple X of length n
    # example X = (youth,medium,yes,fair)
    X = tuple(input('\nEnter the classification tuple.\n'
              'Seperate values using only a comma: \',\'\n > ').split(','))
    print()
    
    # Calculate P(X|Ci) P(Ci) given X for Ci
    
    # 1. P(Ci) : prior probability of each class
        # read data > final column > count unique values and calc P(Ci)

    i,i_counts = np.unique(C, return_counts=True)
    P_Ci = []

    print(' ======= P(Ci) ======= \n')

    for Ci in range(len(i)):
        P_Ci.append(round(float(i_counts[Ci])/len(C),3))
        print(f'P({Headings[-1]} = {i[Ci]}) = '
              f'{i_counts[Ci]} / {len(C)} \n\t= {P_Ci[Ci]}')
        
    print()

    # 2. Compute P(X|Ci)
        # P(X|Ci) = P(X1|Ci)*P(X2|Ci)*P(X3|Ci)*P(X4|Ci)
        # P(X|Ci) = P(Xk|Ci)*P(Xk+1|Ci)*...*P(Xkn|Ci)
    print(' ======= P(X|Ci) for i ======= \n')
    P_X_Ci = []

    for Ci in range(len(i)):
        temp_P_XCi = 1
        for X_item in range(len(X)):
            countX_given_Ci = 0
            for idx, row in X_data.iterrows():
                if row[Headings[X_item]] == X[X_item] and C[idx] == i[Ci]:
                    countX_given_Ci += 1

            if countX_given_Ci == 0:
                P_Xk_Ci = round(countX_given_Ci+1 / i_counts[Ci]+len(np.unique(X_data[Headings[X_item]].tolist())),3)

            print(f'P({Headings[X_item]} = {X[X_item]} | {Headings[-1]} = {i[Ci]}) '
                  f'= {countX_given_Ci} / {i_counts[Ci]}\n\t= {P_Xk_Ci}')
            temp_P_XCi *= P_Xk_Ci

        print(f'\nP(X|{Headings[-1]}={i[Ci]}) = {round(temp_P_XCi,3)}\n')
        P_X_Ci.append(round(temp_P_XCi,3))

    # 3. P(X|Ci) * P(Ci)
    P_XCi_PCi = []

    print(' ======= P(X|Ci) * P(Ci) ======= \n')
    for Ci in range(len(P_Ci)):
        P_XCi_PCi.append(round( P_X_Ci[Ci] * P_Ci[Ci] ,6))
        print(f'P(X|{Headings[-1]} = {i[Ci]}) = {round(P_XCi_PCi[Ci],3)}')

    # array type problems ndarray.index???? fix it
    print('\n ======= Normalization ======= \n')
    P_X = 1.00
    for x_i in range(len(Headings[:-1])):
        temp = tuple(np.unique(X_data[Headings[x_i]].tolist(), 
                                       return_counts=True))
        
        Xi_counts = list(temp[1])
        Xi_vals = list(temp[0])
        P_X *= Xi_counts[Xi_vals.index(X[x_i])]/sum(Xi_counts)

    print(f'P(X) = {P_X}')
    P_Ci_X = []

    for item in P_XCi_PCi:
        P_Ci_X.append(round(item/P_X,4))
        print(f'P({Headings[-1]} = {i[P_XCi_PCi.index(item)]} | X) = ' 
              f'{round(item/P_X,4)}')

    allnull = True    
    for temp_i in P_Ci_X:
        if temp_i > 0:
            allnull = False

    if not allnull:
        print(f'\n\t.: {Headings[-1]} = {i[P_Ci_X.index(max(P_Ci_X))]} given X:{X}')
    else:
        print('.: ALL NULL VALUES...')

def sample_file(n: int) -> None:

    x = {
        'age': ['youth','middle-aged','senior'],
        'income': ['high','medium','low'],
        'student': ['yes','no'],
        'credit-rating': ['fair','excellent'],
        'buys-item': ['yes','no']
    }

    with open('nb_input.csv', 'w') as f:
        for i in range(n):
            line = ''
            for key in x:
                line += (f'{x[key][randint(0, len(x[key])-1)]},')
            line = line[:len(line)-1]
            f.write(f'{line}\n')