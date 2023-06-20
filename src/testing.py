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

import pandas as pd
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np  
csv_path = "/Users/Hanna/OneDrive/Desktop/SPFlow/src/spn/RSPMN_MDP_Datasets/FrozenLake/1StepFL.csv"
df = pd.read_csv(csv_path, sep=',')
partial_order = [['state'], ['action'], ['reward']]
decision_nodes = ['action']
utility_nodes = ['reward']
feature_names = ['state', 'action', 'reward']
meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]
train_data = df.values
from spn.algorithms.SPMN import SPMN
spmn = SPMN(partial_order , decision_nodes, utility_nodes, feature_names, meta_types, cluster_by_curr_information_set=True, util_to_bin = False)
spmn_structure = spmn.learn_spmn(train_data)    
from spn.io.Graphics import plot_spn
        #plot_spn(spmn_structure, "FLSPMNGen.pdf")
from spn.algorithms.MEU import meu
test_data = [[0, np.nan, np.nan]]
meu = meu(spmn_structure, test_data)
print(meu)