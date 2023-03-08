import numpy as np

def Main():
    end = False
    while not end:
        prompt = input('> ')

        if prompt == 'q' or prompt == 'quit':
            end = True

        if prompt == 'k-means':
            K_Means_Clustering()

        if prompt == '':
            pass


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

    print('')
    
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
        print(f'Solved: {solved}')
        print(f'\nCoordinates: {coords}\n')
        print(f'Centroid Coordinates: {centroid_coords}')
        print(f'Distance Matrix: {Distance_matrix}')
        print(f'Grouping Matrix: {Grouping_matrix}\n' \
              f'________________________________________________________________')

        iteration_count += 1
    

if __name__ == "__main__":
    Main()
