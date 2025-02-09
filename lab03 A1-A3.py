import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("C:/Users/Jiya Borikar/Downloads/Judgment_Embeddings_InLegalBERT.xlsx", engine="openpyxl")

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values   

# A1
def evaluate_spread_and_distances(X, y):
    
    unique_classes = np.unique(y)
    class1, class2 = unique_classes[:2]  
    X_class1 = X[y == class1]
    X_class2 = X[y == class2]

    centroid1 = np.mean(X_class1, axis=0)
    centroid2 = np.mean(X_class2, axis=0)

    spread1 = np.std(X_class1, axis=0)
    spread2 = np.std(X_class2, axis=0)

    interclass_distance = np.linalg.norm(centroid1 - centroid2)

    print("Class 1 Centroid:", centroid1)
    print("Class 2 Centroid:", centroid2)
    print("Class 1 Spread (Std Dev):", spread1)
    print("Class 2 Spread (Std Dev):", spread2)
    print("Distance between Class 1 and Class 2 Centroids:", interclass_distance)

evaluate_spread_and_distances(X, y)

# A2
def plot_histogram_for_feature(X, feature_index=0):
    feature_data = X[:, feature_index]
    feature_mean = np.mean(feature_data)
    feature_variance = np.var(feature_data)

    print(f"Feature Index {feature_index}")
    print(f"Mean: {feature_mean}")
    print(f"Variance: {feature_variance}")

    plt.hist(feature_data, bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f"Histogram for Feature {feature_index}")
    plt.xlabel(f"Feature {feature_index}")
    plt.ylabel("Frequency")
    plt.show()

plot_histogram_for_feature(X, feature_index=0)  

# A3
def calculate_minkowski_distance(X, feature_vector1_index, feature_vector2_index):
    vec1 = X[feature_vector1_index]
    vec2 = X[feature_vector2_index]

    minkowski_distances = []
    r_values = range(1, 11)

    for r in r_values:
        distance = np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)
        minkowski_distances.append(distance)

    plt.plot(r_values, minkowski_distances, marker='o', color='red', label='Minkowski Distance')
    plt.title("Minkowski Distance for r = 1 to 10")
    plt.xlabel("r (Order of the Minkowski Distance)")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid()
    plt.show()

calculate_minkowski_distance(X, feature_vector1_index=0, feature_vector2_index=1)
