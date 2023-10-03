if __name__ == '__main__':
    import pandas as pd    
    csv_path = "/home/ht65490/Desktop/SPFlow/RSPMN_MDP_Datasets/FrozenLake/FrozenLake.csv"
    df1 = pd.read_csv(csv_path, sep=',')
    partial_order = [['state'],['action'],['reward']]
    decision_nodes = ["action"]
    utility_node = ["reward"]
    feature_names = ['state','action','reward']

    from spn.structure.StatisticalTypes import MetaType
    meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]
    # Utility variable is the last variable. Other variables are of discrete type
    from spn.algorithms.SPMNDataUtil import align_data
    import numpy as np

    #df, column_titles = align_data(df1, partial_order)  # aligns data in partial order sequence
    train_data = df1.values
    from spn.algorithms.SPMN import SPMN
    spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
    spmn_structure = spmn.learn_spmn(train_data)  
    from spn.algorithms.MEU import meu
    from spn.io.Graphics import plot_spn
    plot_spn(spmn_structure, '/home/ht65490/Desktop/SPFlow/src/output/FL_Documentation', feature_labels=['S','A','R'])
    test_data = [[np.nan, np.nan, np.nan]]
    meu = meu(spmn_structure, test_data)
    print(meu)