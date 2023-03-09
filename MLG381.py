import numpy as np
import pandas as pd

def Main():
    end = False
    while not end:

        print('''
 ===== Menu: =====
1. K-Means Clustering
2. Process Decision Tree \n\t*if any values are NaN, check if your input.csv file is correct
q | quit
______________________''')

        prompt = input('> ').lower()

        if prompt == 'q' or prompt == 'quit':
            end = True

        if prompt == '1':
            K_Means_Clustering()

        if prompt == '2':
            # load the data from the CSV file
            filename = input('Enter the CSV filename: ')
            data = pd.read_csv(filename)
            
            # apply the ID3 algorithm to the data and print the decision tree
            target = data.columns[-1]
            print('Applying ID3 algorithm to dataset of size', len(data))
            root = id3(data, target, indent=0)

            print_tree(root)

class Node:
    def __init__(self, value):
        self.value = value
        self.children = {}
        
    def add_child(self, value, node):
        self.children[value] = node

def id3(data, target, indent):
    # print the current subset of data being processed
    print(' ' * indent, end='')
    print('Processing subset of size', len(data))
    
    # base case: if all examples are of the same class, return a leaf node
    if len(set(data[target])) == 1:
        print(' ' * (indent + 2), end='')
        print('All examples in subset belong to class', data[target].iloc[0])
        return Node(data[target].iloc[0])
    
    # base case: if there are no features left to split on, return a leaf node with the majority class
    if len(data.columns) == 1:
        maj_class = data[target].value_counts().idxmax()
        print(' ' * (indent + 2), end='')
        print('No features left to split on. Majority class is', maj_class)
        return Node(maj_class)
    
    # calculate the information gain of each feature and select the one with the highest gain
    ent = entropy(data[target],indent)
    gain = []
    for col in data.columns[:-1]:
        tg_ent = 0
        for val in set(data[col]):
            tg_ent += entropy(data[data[col] == val][target],indent) * len(data[data[col] == val]) / len(data)
        gain.append(ent - tg_ent)
    best_feature = data.columns[:-1][np.argmax(gain)]
    
    # create a new internal node with the best feature
    root = Node(best_feature)
    print(' ' * (indent + 2), end='')
    print('Splitting on feature', best_feature)
    for val in set(data[best_feature]):
        # recursively split the data and add the child nodes
        print(' ' * (indent + 4), end='')
        print('Subset for', best_feature, '=', val, ':')
        child = id3(data[data[best_feature] == val].drop(columns=[best_feature]), target, indent=indent+6)
        root.add_child(val, child)
        
    return root

def entropy(column, indent):
    # calculate the entropy of a column
    counts = column.value_counts()
    probs = counts / len(column)
    ent = -np.sum(probs * np.log2(probs))
    print(' ' * (indent + 4), end='')
    print('Entropy of column', column.name, '=', ent)
    return ent

def print_tree(node, indent=0):
    # print the decision tree in hierarchical text-format
    print(' ' * indent, end='')
    print(node.value)
    for val, child in node.children.items():
        print(' ' * (indent + 2), end='')
        print(val, end=' ')
        if isinstance(child, Node):
            print('')
            print_tree(child, indent=indent+4)
        else:
            print(': ', end='')
            print(child)

def K_Means_Clustering():
    # k = pre-specified
    # Repeat until cluster groups from iteration x = iteration x-1
        # Determine centroid coordinates
        # Determine distance from each object to centroid (Dn-Matrix)
        # Group into clusters based on minimum distance

    k = int(input('Cluster Count\n> '))
    centroid_coords = []

    coords = []
    inp = input('Enter the data, each dimension seperated only with a comma.\nWrite "done" to stop entering data.\n' \
                'enter data > ')
    while not inp == 'done':
        coords.append(tuple(map(int, inp.split(','))))
        inp = input('enter data > ')

    print('________________________________________________________________')
    
    solved = False
    iteration_count = 0

    Previous_G_Matrix = []
    Grouping_matrix = []

    while not (solved):
        print(f'Iteration: {iteration_count}')
        # 1. Determine centroid coordinates
        if len(centroid_coords) == 0:
            for i in range(k):
                centroid_coords.append(coords[i])
        else:
            temp = []
            for row in range(k):
                temp_x = 0.00
                temp_y = 0.00
                counter = 0
                for item_index in range(len(Grouping_matrix[row])):
                    if Grouping_matrix[row][item_index] == 1:
                        temp_x += float(coords[item_index][0])
                        temp_y += float(coords[item_index][1])
                        counter += 1
                temp.append(tuple([np.round(temp_x/counter, 2), np.round(temp_y/counter, 2)]))
                    
            centroid_coords = temp.copy()
        
        # 2. Determine distance from each object to centroid (Dn-Matrix)
        Distance_matrix = []

        for x in range(k):
            Distance_matrix.append([])
            for y in range(len(coords)):
                # Euclidean distance
                temp_val = np.round(np.linalg.norm(np.asarray(centroid_coords[x]) - np.asarray(coords[y])), 2)
                Distance_matrix[x].append(temp_val)
        
        # 3. Group into clusters based on minimum distance
        Previous_G_Matrix = Grouping_matrix.copy()
        Grouping_matrix = [[],[]]

        for i in range(len(coords)):
            if Distance_matrix[0][i] < Distance_matrix[1][i]:
                Grouping_matrix[0].append(1)
                Grouping_matrix[1].append(0)
            else:
                Grouping_matrix[0].append(0)
                Grouping_matrix[1].append(1)  

        # 4. Test group similarity -> if .. same = True
        if Previous_G_Matrix == Grouping_matrix:
            solved = True

        # Output
        if solved: 
            print(f'Solved: {solved}')
            print(f'Coordinates: {coords}\n')
        print(f'Centroid Coordinates: {centroid_coords}')
        print(f'Distance Matrix: {Distance_matrix}')
        print(f'Grouping Matrix: {Grouping_matrix}\n' \
              f'________________________________________________________________')

        iteration_count += 1
    

if __name__ == "__main__":
    Main()
