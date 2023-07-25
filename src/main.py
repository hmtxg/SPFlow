if __name__ == '__main__':
    import pandas as pd    
    csv_path = "/home/ht65490/Desktop/SPFlow/PD_3S_Mirror.csv"
    df1 = pd.read_csv(csv_path, sep=',')
    partial_order = [['Stage0Reward'],['Stage1Action'],['Stage1Reward'],['Stage2Action'],['Stage2Reward'],['Stage3Action'],['Stage3Reward'],['AvgReward']]
    utility_node = ['AvgReward']
    decision_nodes = ['Stage1Action', 'Stage2Action', 'Stage3Action']
    feature_names = ['Stage0Reward','Stage1Action','Stage1Reward','Stage2Action','Stage2Reward','Stage3Action','Stage3Reward','AvgReward']

    from spn.structure.StatisticalTypes import MetaType
    # Utility variable is the last variable. Other variables are of discrete type
    meta_types = [MetaType.DISCRETE]*7+[MetaType.UTILITY]
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
    plot_spn(spmn_structure, '/home/ht65490/Desktop/SPFlow/src/output/PD_1S_Defect.pdf', feature_labels=['R0', 'A1', 'R1', 'A2', 'R2', 'A3', 'R3', 'AR'])
    test_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
    meu = meu(spmn_structure, test_data)
    print(meu)