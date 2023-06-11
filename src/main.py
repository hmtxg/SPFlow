if __name__ == '__main__':
        # import numpy as np
        # import pandas as pd
        # import logging
        # from spn.algorithms.SPMN import SPMN
        # from spn.io.Graphics import plot_spn
        # from spn.structure.StatisticalTypes import MetaType
        # from spn.algorithms.SPMNDataUtil import align_data, cooper_tranformation
        # from spn.algorithms.EM import EM_optimization
        # from spn.structure.Base import get_nodes_by_type, Sum, Product, Max, Leaf
        # from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
        # from spn.structure.leaves.histogram.Histograms import Histogram
        # from spn.algorithms.MEU import meu

        # #logging.basicConfig(filename=".log", level=logging.DEBUG)

        # # load the data
        # csv_path = "/Users/Hanna/OneDrive/Desktop/SPFlow/src/spn/data/Powerplant_Airpollution/Powerplant_airpollutionOG.tsv"
        # data_frame = pd.read_csv(csv_path, sep='\t')
        # partial_order = [['Installation_Type'], ['Coal_Worker_Strike'], ['Strike_Resolution'], ['Strike_Intervention'], 
        #          ['Additional_Cost']]
        # utility_node = ['Additional_Cost']
        # decision_nodes = ['Installation_Type', 'Strike_Intervention']
        # feature_names = ['Installation_Type', 'Coal_Worker_Strike', 'Strike_Resolution', 'Strike_Intervention', 
        #          'Additional_Cost']
        # meta_types = [MetaType.DISCRETE]*4+[MetaType.UTILITY]  
        # #data = data[feature_names] # re-arrange the columns to match the feature_names
        # #logging.info(feature_names)
        # #print(data_frame['Economical_State'].unique())
        # #print(data_frame['Export_Decision'].unique())

        # # replace the values of categorical features by their indeces
        # # for i,col in enumerate(list(data_frame.columns)) :
        # #         vals = sorted(data_frame[col].unique())
        # #         if meta_types[i] == MetaType.DISCRETE :
        # #                 to_replace = {k: int(v) for v, k in enumerate(vals)}
        # #                 data_frame.replace(to_replace, inplace=True)

        # #train_data = cooper_tranformation(data_frame.to_numpy(), 2)
        # train_data = data_frame.to_numpy()
        # # print(train_data)

        # # create the SPMN
        # spmn = SPMN(
        #         partial_order=partial_order,
        #         decision_nodes=decision_nodes,
        #         utility_node=utility_node,
        #         feature_names=feature_names,
        #         meta_types=meta_types,
        #         util_to_bin=False
        # )

        # # learn the structure
        # spmn.learn_spmn(train_data)

        # EM_optimization(spn=spmn.spmn_structure, data=train_data)

        # plot_spn(spmn.spmn_structure, 'Powerplant.pdf')

        # # debug the structure
        # all_nodes = get_nodes_by_type(spmn.spmn_structure)
        # logging.info(f'There are {len(all_nodes)} nodes in the SPMN')
        # for node in all_nodes :
        #         if isinstance(node, Sum) :
        #                 print(f'{node.id}: Sum node with weights {node.weights}')
        #         elif isinstance(node, Max) :
        #                 print(f'{node.id}: Max node with decisions {node.dec_values}')
        #         elif isinstance(node, Product) :
        #                 print(f'{node.id}: Product node with {len(node.children)} children')
        #         elif isinstance(node, Utility) :
        #                 print(f'{node.id}: Utility node with scope {node.scope}')
        #                 print(f'  breaks: {node.breaks}')
        #                 print(f'  bin_repr_points: {node.bin_repr_points}')
        #                 print(f'  count: {node.count}')
        #         elif isinstance(node, Histogram) :
        #                 print(f'{node.id}: Variable node with scope {node.scope}')
        #                 print(f'  breaks: {node.breaks}')
        #                 print(f'  bin_repr_points: {node.bin_repr_points}')
        #                 print(f'  count: {node.count}')

        # input_data = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
        # res = meu(spmn.spmn_structure, input_data)
        # print(f'MEU is {res}')
        import pandas as pd    
        csv_path = "/Users/Hanna/OneDrive/Desktop/SPFlow/src/spn/data/HIV_Screening/HIV_Screening.tsv"
        df = pd.read_csv(csv_path, sep='\t')
        partial_order = [['Screen'], ['HIV_Test_Result'], ['Treat_Counsel'], 
                 ['HIV_Status', 'Compliance_Medical_Therapy', 'Reduce_Risky_Behavior'],['QALE']]
        utility_node = ['QALE']
        decision_nodes = ['Screen', 'Treat_Counsel']
        feature_names = ['Screen', 'HIV_Test_Result', 'Treat_Counsel', 
                 'HIV_Status', 'Compliance_Medical_Therapy', 'Reduce_Risky_Behavior','QALE']

        from spn.structure.StatisticalTypes import MetaType
        # Utility variable is the last variable. Other variables are of discrete type
        meta_types = [MetaType.DISCRETE]*6+[MetaType.UTILITY]  
        from spn.algorithms.SPMNDataUtil import align_data
        import numpy as np

        df1, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
        train_data = df1.values
        from spn.algorithms.SPMN import SPMN
        spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
        spmn_structure = spmn.learn_spmn(train_data)    
        from spn.io.Graphics import plot_spn
        plot_spn(spmn_structure, "HIV.pdf")
        from spn.algorithms.MEU import meu
        test_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        meu = meu(spmn_structure, test_data)
        print(meu)