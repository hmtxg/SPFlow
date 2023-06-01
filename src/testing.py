# import numpy as np
# from sklearn.cluster import KMeans

# #Identical instances
# X = np.array([[1, 2], [1, 2], [1, 2], [1, 2],[1, 2]])

# #KMeans initialized to 2 clusters
# kmeans2 = KMeans(n_clusters=2, random_state=17)

# #Fit the model to the data
# kmeans2.fit_predict(X)
# data_slices_sum = []
# #Cluster labels per instance
# labels2 = kmeans2.labels_
# datalength = kmeans2.labels_.shape[0]
# for i in range(datalength//2):
#     kmeans2.labels_[i] = 1
#     data_slices_sum.append(X[i])
# print(data_slices_sum)

# print(kmeans2.labels_)
# #Cluster centroids
# centroids2 = kmeans2.cluster_centers_

# print(labels2)     #Out: [0 0 0 0]
# print(centroids2)  #Out: [[1. 2.], [1. 2.]]

import os
print('aaaaaaaaaaaaa')
print(os.environ['PATH'])