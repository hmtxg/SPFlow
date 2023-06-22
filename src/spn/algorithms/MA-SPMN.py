from spn.structure.Base import Sum, Product, Max, Context
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric, learn_mspn_for_spmn
from spn.algorithms.SPMNHelper import get_ds_context, column_slice_data_by_scope, \
                                      split_on_decision_node, get_split_rows_KMeans, \
                                      get_row_indices_of_cluster, row_slice_data_by_indices
import logging
import numpy as np
from spn.algorithms.SPMN import SPMN
# from spn.algorithms.TransformStructure import Prune
from spn.algorithms.MEU import meu, meu_prod
from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.StatisticalTypes import MetaType


class MASPMN:
    def __init__(self, schema, partial_order, decision_nodes, number_agents, utility_node, feature_names, meta_types, cluster_by_curr_information_set=False, util_to_bin=False):
            
        self.params = MASPMNParams(
                schema,
                partial_order,
                decision_nodes,
                number_agents,
                utility_node,
                feature_names,
                meta_types,
                util_to_bin
            )
        self.cluster_by_curr_information_set = cluster_by_curr_information_set
        self.maspmn_structure = None
    
    def buildCTCE(self, Agents_data):
        partial_order = self.partial_order
        decision_nodes = self.decision_nodes
        utility_node = self.utility_node
        feature_names = self.feature_names
        meta_types = self.meta_types
        util_to_bin = self.util_to_bin
        rawSPMN = SPMN(partial_order, decision_nodes, utility_node, feature_names, meta_types, util_to_bin, cluster_by_curr_information_set=True)
        trainedSPMN = rawSPMN.learn_spmn(Agents_data)
        return trainedSPMN
    
    def getAgentData(agentIndex, Agents_data):
        Agent_data = Agents_data[agentIndex]
        return Agent_data
    
    def buildCTDE(self, Agents_data):
        #Set Parameters
        numAgents = self.number_agents
        partial_order = self.partial_order
        decision_nodes = self.decision_nodes
        utility_node = self.utility_node
        feature_names = self.feature_names
        meta_types = self.meta_types
        util_to_bin = self.util_to_bin

        #Build Agent Layer
        agentNodes = []
        for i in range(numAgents):
            #Create & Train Agent SPMN
            rawAgentSPMN = SPMN(partial_order, decision_nodes, utility_node, feature_names, meta_types, util_to_bin, cluster_by_curr_information_set=True)
            Agent_data = self.getAgentData(i, Agents_data)
            agentNodes.append(rawAgentSPMN.learn_spmn(Agent_data))

        root = Product(agentNodes)
        return root
   
    def buildDTDE(self, Agents_data):
        # perceptSPN = self.buildPerceptSPN(self, Agents_data)
        # A perceptSPN can be built and evaluated to provide an RV, but the partial ordering of the input data may also include Percept RVs
        # I am implementing the simplest way, which is considering RVs in the input data to the SPMN. 
        partial_order = self.partial_order
        decision_nodes = self.decision_nodes
        utility_node = self.utility_node
        feature_names = self.feature_names
        meta_types = self.meta_types
        util_to_bin = self.util_to_bin
        rawSPMN = SPMN(partial_order, decision_nodes, utility_node, feature_names, meta_types, util_to_bin, cluster_by_curr_information_set=True)
        trainedSPMN = rawSPMN.learn_spmn(Agents_data)
        return trainedSPMN

    def learn_MASPMN(self, Agents_data):
        if(self.schema=='CTCE'):
            spmn = self.buildCTCE(self, Agents_data)
            return spmn
        elif(self.schema=='CTDE'):
            agentSPMNS = self.buildCTDE(self, Agents_data)
            return agentSPMNS
        elif(self.schema=='DTDE'):
            agentSPMN= self.buildDTDE(self, Agents_data)
            return agentSPMN
        
    def meu_MASPMN(maspmn_structure, test_data):
        meu = meu(maspmn_structure, test_data)
        return meu
    
    # def meu_MASPMN_CTDE(self, agentSPMNs, test_data_agents, test_data_top):
        # agentSPMNs = agentSPMNs
        # numAgents = self.number_agents
        # #Create top network agentLeafs
        # agentLeafs = []
        # for i in range(numAgents):
        #     agentNet = agentSPMNs[i]
        #     agentData = test_data_agents[i]
        #     #Evaluate agentSPMN
        #     agentLeafs[i] = meu(agentNet, agentData) #parametric data to learn top net

        # root = Product()
        # ds_context = Context(meta_types=[MetaType.DISCRETE*numAgents])
        # ds_context.add_domains(agentLeafs)
        # mspn = learn_mspn(agentLeafs, ds_context, min_instances_slice=1)
        # meu_prod(mspn, test_data_top)


class MASPMNParams:

    def __init__(self, schema, partial_order, decision_nodes, percept_nodes_data, number_agents, utility_node, feature_names, meta_types, util_to_bin):

        self.schema = schema
        self.partial_order = partial_order
        self.decision_nodes = decision_nodes
        self.percept_nodes_data = percept_nodes_data
        self.number_agents = number_agents
        self.utility_node = utility_node
        self.feature_names = feature_names
        self.meta_types = meta_types
        self.util_to_bin = util_to_bin