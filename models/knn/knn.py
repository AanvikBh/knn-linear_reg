# from math import *
# import heapq

# from math import sqrt

# # distance metrics only defined for int/float and not on categorical values 
# # can use binary indicator for categorical values 

# def euclidean_distance(row1, row2):
#     dist = 0.0
#     for i in range(len(row1)):
#         if isinstance(row1[i], (int, float)) and isinstance(row2[i], (int, float)):
#             dist += (row1[i] - row2[i]) ** 2
#     dist = sqrt(dist)
#     return dist

# def manhattan_distance(row1, row2):
#     dist = 0.0
#     for i in range(len(row1)):
#         if isinstance(row1[i], (int, float)) and isinstance(row2[i], (int, float)):
#             dist += abs(row1[i] - row2[i])
#     return dist

# def cosine_distance(row1, row2):
#     dot_product = 0.0
#     norm_row1 = 0.0
#     norm_row2 = 0.0
    
#     for i in range(len(row1)):
#         if isinstance(row1[i], (int, float)) and isinstance(row2[i], (int, float)):
#             dot_product += row1[i] * row2[i]
#             norm_row1 += row1[i] ** 2
#             norm_row2 += row2[i] ** 2
    
#     norm_row1 = sqrt(norm_row1)
#     norm_row2 = sqrt(norm_row2)
    
#     if norm_row1 == 0 or norm_row2 == 0:
#         return 1.0  # If either norm is zero, cosine similarity is undefined; assume max distance
    
#     cosine_similarity = dot_product / (norm_row1 * norm_row2)
#     cosine_distance = 1 - cosine_similarity
    
#     return cosine_distance


# # ChatGPT prompt: https://chatgpt.com/share/2c57959a-246a-4fa0-a15e-36e477c628ee

# # def get_neighbours(k, train_dataset, test_row, metric):
# #     max_heap = []
# #     for row in train_dataset:
# #         dist = metric(row, test_row)
# #         if len(max_heap) < k:
# #             heapq.heappush(max_heap, (-dist, row))
# #         else:
# #             if -dist > max_heap[0][0]:
# #                 heapq.heapreplace(max_heap, (-dist, row))
# #     neighbours = [row for (_, row) in max_heap]
# #     return neighbours 

# # def predict_class(k, train_dataset, test_row, metric):
# #     k_nearest_neighbours = get_neighbours(k, train_dataset, test_row, metric)    
# #     # Count the occurrences of each class label among the nearest neighbors
# #     class_votes = {}
# #     for neighbour in k_nearest_neighbours:
# #         label = neighbour[-1]  # Assuming the class label is the last element in the row
# #         if label in class_votes:
# #             class_votes[label] += 1
# #         else:
# #             class_votes[label] = 1
# #     # Return the class label with the most votes
# #     predicted_label = max(class_votes, key=class_votes.get)
# #     return predicted_label


# class KNN:
#     def __init__(self, k, metric):
#         self.k = k
#         self.distance_metrices = metric
#         self.train_dataset = []
#         self.val_dataset = []
#         self.test_dataset = []
#         if self.distance_metrics == "euclidean":
#             self.dist_func = euclidean_distance
#         elif self.distance_metrics == "manhattan":
#             self.dist_func = manhattan_distance
#         elif self.distance_metrics == "cosine":
#             self.dist_func = cosine_distance

#     def fit(self, train_dataset, val_dataset=None):
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset if val_dataset is not None else []

#     def predict(self, test_row):
#         k_nearest_neighbours = self.get_neighbours(self.k, self.train_dataset, test_row)
        
#         # Count the occurrences of each class label among the nearest neighbors
#         class_votes = {}
#         for neighbour in k_nearest_neighbours:
#             label = neighbour[-1]  # Assuming the class label is the last element in the row
#             if label in class_votes:
#                 class_votes[label] += 1
#             else:
#                 class_votes[label] = 1
        
#         # Return the class label with the most votes
#         predicted_label = max(class_votes, key=class_votes.get)
#         return predicted_label

#     def predict_batch(self, test_dataset):
#         predictions = []
#         for test_row in test_dataset:
#             predictions.append(self.predict(test_row))
#         return predictions

#     def get_neighbours(self, k, train_dataset, test_row):
#         max_heap = []

#         for row in train_dataset:
#             dist = self.metric(row, test_row)
#             if len(max_heap) < k:
#                 heapq.heappush(max_heap, (-dist, row))
#             else:
#                 if -dist > max_heap[0][0]:
#                     heapq.heapreplace(max_heap, (-dist, row))

#         neighbours = [row for (_, row) in max_heap]
#         return neighbours


from joblib import Parallel, delayed
import numpy as np
import pandas as pd

def euclidean_distance(X_train, X_query):
    distances = []
    for row in X_train:
        dist = np.sqrt(np.sum((row - X_query) ** 2))
        distances.append(dist)
    return np.array(distances)

def manhattan_distance(X_train, X_query):
    distances = []
    for row in X_train:
        dist = np.sum(np.abs(row - X_query))
        distances.append(dist)
    return np.array(distances)

def cosine_distance(X_train, X_query):
    distances = []
    norm_X_query = np.linalg.norm(X_query)
    for row in X_train:
        norm_row = np.linalg.norm(row)
        dot_product = np.dot(row, X_query)
        eps = 1e-10
        cosine_similarity = dot_product / (norm_row * norm_X_query + eps)
        cosine_dist = 1 - cosine_similarity
        distances.append(cosine_dist)
    return np.array(distances)


class KNN:
    def __init__(self, k, metric, vectorized=None):
        self.k = k
        self.distance_metrics = metric
        self.X_train = list()
        self.y_train = list()
        self.vectorization = 0
        if vectorized == 'Vectorization':
            self.vectorization = 1

        if self.vectorization == 0:
            if self.distance_metrics == "euclidean":
                self.dist_func = euclidean_distance
            elif self.distance_metrics == "manhattan":
                self.dist_func = manhattan_distance
            elif self.distance_metrics == "cosine":
                self.dist_func = cosine_distance

    def fit(self, train_data_X, train_data_Y):
        self.X_train = train_data_X
        self.y_train = train_data_Y

    def predict(self, test_row):
        # Calculate the distances between the test_row and all training data
        # distances = self.dist_func(self.X_train, test_row)
        if self.vectorization:
            distances = self.distance_optim(test_row)
        else:
            distances = self.dist_func(self.X_train, test_row)
        
        # Get the indices of the k smallest distances
        sorted_indices = np.argsort(distances)
        
        # Get the labels of the k nearest neighbors
        k_nearest = [self.y_train[i] for i in sorted_indices[:self.k]]
        
        # Get the most common label among the nearest neighbors
        # most_common = Counter(k_nearest).most_common(1)[0][0]
        most_common = np.bincount(k_nearest).argmax()
        return most_common

    def predict_batch(self, test_X):
        predictions = Parallel(n_jobs=1)(delayed(self.predict)(test_X[i]) for i in tqdm(range(len(test_X))))

        return predictions

    # def get_neighbours(self, k, train_dataset, test_row):
    #     distances = [(self.dist_func(row, test_row), row) for row in train_dataset]
    #     distances.sort(key=lambda x: x[0])
    #     neighbours = [row for _, row in distances[:k]]
    #     return neighbours
    def distance_optim(self, X_query):
        if self.distance_metrics == "euclidean":
            return np.sqrt(np.sum((self.X_train - X_query)**2, axis=1))
        elif self.distance_metrics == "manhattan":
            return np.sum(np.abs(self.X_train - X_query), axis=1)
        elif self.distance_metrics == "cosine":
            norm_X_train = np.linalg.norm(self.data_X, axis=1)
            norm_X_query = np.linalg.norm(X_query)
            eps = 1e-10
            return 1 - (np.dot(self.X_train, X_query.T) / (norm_X_train * norm_X_query + eps))
        
    # def distance_normal(self, X_query):
    #     return 

