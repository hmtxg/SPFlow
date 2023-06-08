if __name__ == '__main__':
#     import pandas as pd    
#     csv_path = "/Users/Hanna/OneDrive/Desktop/SPFlow/src/spn/data/Computer_Diagnostician/Computer_Diagnostician.tsv"
#     df = pd.read_csv(csv_path, sep='\t')
#     partial_order = [['System_State'], ['Rework_Decision'],
#                  ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 
#                  'Rework_Cost']]
#     utility_node = ['Rework_Cost']
#     decision_nodes = ['Rework_Decision']
#     feature_names = ['System_State', 'Rework_Decision', 'Logic_board_fail', 
#                 'IO_board_fail', 'Rework_Outcome', 'Rework_Cost']

#     from spn.structure.StatisticalTypes import MetaType
#     # Utility variable is the last variable. Other variables are of discrete type
#     meta_types = [MetaType.DISCRETE]*5+[MetaType.UTILITY]  
#     from spn.algorithms.SPMNDataUtil import align_data
#     import numpy as np

#     df, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
#     train_data = df.values
#     from spn.algorithms.SPMN import SPMN
#     spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
#             meta_types, cluster_by_curr_information_set=True,
#             util_to_bin = False)
#     spmn_structure = spmn.learn_spmn(train_data)  
#     from spn.algorithms.MEU import meu
#     test_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
#     meu = meu(spmn_structure, test_data)
#     print(meu)
        import numpy as np
        import pandas as pd
        import logging

        from spn.algorithms.SPMN import SPMN
        from spn.io.Graphics import plot_spn
        from spn.structure.StatisticalTypes import MetaType
        from spn.algorithms.SPMNDataUtil import align_data, cooper_tranformation
        from spn.algorithms.EM import EM_optimization
        from spn.structure.Base import get_nodes_by_type, Sum, Product, Max, Leaf
        from spn.structure.leaves.spmnLeaves.SPMNLeaf import Utility
        from spn.structure.leaves.histogram.Histograms import Histogram
        from spn.algorithms.MEU import meu

        logging.basicConfig(level=logging.INFO)

        # load the data
        file_name = '/Users/Hanna/OneDrive/Desktop/SPFlow/src/spn/data/Export_Textiles/Export_textiles.tsv'
        data_frame = pd.read_csv(file_name, sep='\t')

        # set the params
        feature_names = ['Economical_State', 'Export_Decision', 'U']
        partial_order = [['Economical_State'], ['Export_Decision'], ['U']]
        decision_nodes = ['Export_Decision']
        utility_nodes = ['U']
        meta_types = [MetaType.DISCRETE, MetaType.DISCRETE, MetaType.UTILITY]
        util_to_bin = False
        data_frame,_ = align_data(data_frame, partial_order)

        #data = data[feature_names] # re-arrange the columns to match the feature_names
        #logging.info(feature_names)
        #print(data_frame['Economical_State'].unique())
        #print(data_frame['Export_Decision'].unique())

        # replace the values of categorical features by their indeces
        for i,col in enumerate(list(data_frame.columns)) :
                vals = sorted(data_frame[col].unique())
                if meta_types[i] == MetaType.DISCRETE :
                        to_replace = {k: int(v) for v, k in enumerate(vals)}
                        data_frame.replace(to_replace, inplace=True)

        #train_data = cooper_tranformation(data_frame.to_numpy(), 2)
        train_data = data_frame.to_numpy()
        print(train_data)

        # create the SPMN
        spmn = SPMN(
                partial_order=partial_order,
                decision_nodes=decision_nodes,
                utility_node=utility_nodes,
                feature_names=feature_names,
                meta_types=meta_types,
                util_to_bin=util_to_bin
        )

        # learn the structure
        spmn.learn_spmn(train_data)

        EM_optimization(spn=spmn.spmn_structure, data=train_data)

        plot_spn(spmn.spmn_structure, 'test.pdf')

        # debug the structure
        all_nodes = get_nodes_by_type(spmn.spmn_structure)
        logging.info(f'There are {len(all_nodes)} nodes in the SPMN')
        for node in all_nodes :
                if isinstance(node, Sum) :
                        print(f'{node.id}: Sum node with weights {node.weights}')
                elif isinstance(node, Max) :
                        print(f'{node.id}: Max node with decisions {node.dec_values}')
                elif isinstance(node, Product) :
                        print(f'{node.id}: Product node with {len(node.children)} children')
                elif isinstance(node, Utility) :
                        print(f'{node.id}: Utility node with scope {node.scope}')
                        print(f'  breaks: {node.breaks}')
                        print(f'  bin_repr_points: {node.bin_repr_points}')
                        print(f'  count: {node.count}')
                elif isinstance(node, Histogram) :
                        print(f'{node.id}: Variable node with scope {node.scope}')
                        print(f'  breaks: {node.breaks}')
                        print(f'  bin_repr_points: {node.bin_repr_points}')
                        print(f'  count: {node.count}')

        input_data = np.array([np.nan, np.nan, np.nan]).reshape(1,3)
        res = meu(spmn.spmn_structure, input_data)
        print(f'MEU is {res}')