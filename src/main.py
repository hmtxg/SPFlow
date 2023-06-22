if __name__ == '__main__':
    import pandas as pd    
    csv_path = "/home/ht65490/Desktop/SPFlow/src/spn/data/Export_Textiles/Export_Textiles.tsv"
    df1 = pd.read_csv(csv_path, sep='\t')
    partial_order = [['Export_Decision'], ['Economical_State'], ['Profit']]
    utility_node = ['Profit']
    decision_nodes = ['Export_Decision']
    feature_names = ['Export_Decision', 'Economical_State', 'Profit']

    from spn.structure.StatisticalTypes import MetaType
    # Utility variable is the last variable. Other variables are of discrete type
    meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]
    from spn.algorithms.SPMNDataUtil import align_data
    import numpy as np

    df, column_titles = align_data(df1, partial_order)  # aligns data in partial order sequence
    train_data = df.values
    from spn.algorithms.SPMN import SPMN
    spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
    spmn_structure = spmn.learn_spmn(train_data)  
    from spn.algorithms.MEU import meu
    test_data = [[np.nan, np.nan, np.nan]]
    meu = meu(spmn_structure, test_data)
    print(meu)