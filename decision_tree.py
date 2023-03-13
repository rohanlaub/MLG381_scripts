import numpy as np
import pandas as pd
from collections import Counter

# Node class for building the decision tree
class Node:
    def __init__(self, data: pd.DataFrame, label: str):
        self.data = data
        self.label = label
        self.children = {}

# Function to calculate entropy of a dataset
def entropy(data: pd.DataFrame) -> float:
    labels = data.iloc[:, -1]
    counts = Counter(labels)
    entropy = 0.0
    for count in counts.values():
        p = count / len(labels)
        entropy -= p * np.log2(p)
    return entropy

# Function to calculate information gain for a feature
def info_gain(data: pd.DataFrame, feature: str) -> tuple[float, str, float, float]:
    values = data[feature].unique()
    E_SX = 0.0
    for value in values:
        subset = data[data[feature] == value]
        E_SX += len(subset) / len(data) * entropy(subset)
        print(f'Sv/S: {len(subset)} / {len(data)}\tS({value}) {entropy(subset)}')

    E_S = entropy(data)
    I_SX = E_S - E_SX

    # OUTPUT CALCULATIONS
    print(f'E(S): {E_S}\nE(S,{feature}): {E_SX}\nI(S,{feature}): {I_SX}\n')
    return E_S, feature, E_SX, I_SX

# Function to select the best feature to split on
def select_feature(data: pd.DataFrame) -> tuple[float, str, float, float]:
    features = data.columns[:-1]
    best_feature = None
    best_gain = 0.0
    for feature in features:
        gain = info_gain(data, feature)[-1] 
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return info_gain(data, best_feature)

# Function to build the decision tree recursively
def build_tree(data: pd.DataFrame, indent: int=0) -> Node:
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return Node(None, labels.unique()[0])
    best_feature_info = select_feature(data)
    node = Node(data, best_feature_info[1])
    values = data[best_feature_info[1]].unique()
    for value in values:
        subset = data[data[best_feature_info[1]] == value].drop(best_feature_info[1], axis=1)
        if len(subset) == 0:
            node.children[value] = Node(None, labels.mode()[0])
        else:
            child = build_tree(subset, indent+1)
            node.children[value] = child
    return node

# Function to print the decision tree
def print_tree(node: Node, indent: int=0) -> str:

    if node.data is None:
        print(f'{"  "* indent}-> {node.label}')
    else:
        print(f'{"  "* indent}-> {node.label}')
        for value, child in node.children.items():
            print(f'{"  " * (indent+1)}-> {value}')
            print_tree(child, indent+2)
    