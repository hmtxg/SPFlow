# import pandas as pd
# import numpy as np

# df = pd.read_csv('/home/ht65490/Desktop/SPFlow/src/spn/RSPMN_MDP_Datasets/Taxi/Taxi.csv')
# # Get the number of columns in the original DataFrame
# num_columns = df.shape[1]
# print(num_columns)
# # Calculate the number of groups of 3 columns
# num_groups = 15

# # Create an empty DataFrame to store the reshaped data
# reshaped_df = pd.DataFrame()

# # Iterate over the groups of 3 columns
# for i in range(num_groups):
#     # Get the starting and ending column indices for the current group
#     start = i * 3
#     end = (i + 1) * 3
    
#     # Select the columns for the current group
#     group_columns = df.iloc[:, start:end]
    
#     # Reshape the group columns and append them to the reshaped DataFrame
#     group_values = group_columns.values.reshape(-1, 3)
#     reshaped_df = pd.concat([reshaped_df, pd.DataFrame(group_values)], axis=0)

# # Reset the index of the reshaped DataFrame
# reshaped_df.reset_index(drop=True, inplace=True)

# # Save the reshaped DataFrame to a new CSV file
# reshaped_df.to_csv('taxi1step.csv', index=False)

# import gym
# import numpy as np
# import csv

# env = gym.make("FrozenLake-v1", is_slippery=True)

# from functools import reduce
# out_array = [reduce(lambda x, y: x + y, [["state" + str(i), "action" + str(i), "reward" + str(i)] for i in range(8)])]

# for i in range(10000):
#     prev_state = 0
#     env.reset()
#     log = [i]
#     print("[t,action,observation,reward]:")
#     for t in range(8):
#         action = env.action_space.sample()
#         new_state, reward, done, info = env.step(action)
#         observation = prev_state
#         log = [observation, action, reward]
#         print([t, observation, action, reward])
#         out_array.append(log)
#
# file_path = '/Users/Hanna/OneDrive/Desktop/PettingZoo/FLGen.csv'
#
# with open(file_path, 'w', newline='') as myfile:
#     wr = csv.writer(myfile, delimiter='\t')
#     wr.writerows(out_array)

import pandas as pd
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np  
csv_path = "/home/ht65490/Desktop/SPFlow/src/spn/RSPMN_MDP_Datasets/Taxi/taxi1step.csv"
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
from spn.algorithms.MEU import meu
test_data = [[np.nan, np.nan, np.nan]]
meu = meu(spmn_structure, test_data)
print(meu)