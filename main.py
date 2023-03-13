import pandas as pd

# Custom scripts
import kmeans as o1
import decision_tree as o2
import regression as o3

def Main():
    end = False
    while not end:

        print('''
 ===== Menu: =====
1. K-Means Clustering
2. Process Decision Tree \n\t*if any values are NaN, check if your input.csv file is correct
3.1 Linear Regression
3.2 Logistic Regression
q | quit
______________________''')

        prompt = input('> ').lower()

        if prompt == 'q' or prompt == 'quit':
            end = True

        if prompt == '1':
            o1.K_Means_Clustering()

        if prompt == '2':
            try:
                # Load data from CSV file
                print('')
                data = pd.read_csv(input("Enter CSV file name: "))
                print()

                root = o2.build_tree(data)
                print('\n FINAL DECISION TREE:\n')
                o2.print_tree(root)
            except Exception as e:
                print(e)

        if prompt == '3.1':
            o3.Linear_Regression()

        if prompt == '3.2':
            o3.Logistic_Regression()

if __name__ == "__main__":
    Main()
