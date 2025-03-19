from scipy.spatial import distance
import numpy as np

from simulator.node.node_info import Node_Type
from simulator.node.node import Node
from simulator import parameters as para

class ConnectorNode(Node):
    def find_receiver(self, net):
        """
        find receiver node
        :param node: node send this package
        :param net: network
        :return: find node nearest base from neighbor of the node and return id of it
        """
        if not self.is_active:
            return Node(id = -1)

        if self.candidate is not None and self.candidate.is_active == True:
            return self.candidate
        
        centroid = np.array(net.listClusters[self.cluster_id].centroid)

        neighbors = np.array(self.neighbor)
        if not neighbors.size:
            return Node(id=-1)
    
        active_mask = np.array([n.is_active for n in neighbors])
        type_mask = np.isin([n.type_node for n in neighbors], [Node_Type.IN_NODE, Node_Type.OUT_NODE, Node_Type.CONNECTOR_NODE])
        cluster_mask = np.array([n.cluster_id == self.cluster_id for n in neighbors])
        level_mask = np.array([self.level > n.level for n in neighbors])
    
        valid_mask = active_mask & type_mask & cluster_mask & level_mask
            
        valid_neighbors = neighbors[valid_mask]
        if not valid_neighbors.size:
            return Node(id=-1)
    
        # Vector hóa tính khoảng cách
        neighbor_locs = np.array([n.location for n in valid_neighbors])
        distances = distance.cdist(neighbor_locs, [centroid], metric='euclidean').flatten()
    
        # Tìm node gần nhất
        min_idx = np.argmin(distances)
        node_min = valid_neighbors[min_idx]
        self.candidate = node_min
        
        return node_min
    
    def probe_neighbors(self, network):
        self.neighbor.clear()
        self.potentialSender.clear()

        nodes = np.array(network.node)
        if not nodes.size:
            return
        
        node_locs = np.array([n.location for n in nodes])
        active_mask = np.array([n.is_active for n in nodes])
        id_mask = np.array([n.id != self.id for n in nodes])

        # Tính khoảng cách vector hóa
        distances = distance.cdist(node_locs, [self.location], metric='euclidean').flatten()
        range_mask = distances <= self.com_ran
        
        # Lọc neighbor
        neighbor_mask = active_mask & id_mask & range_mask
        self.neighbor = nodes[neighbor_mask].tolist()
        
        # Lọc potentialSender
        type_mask = np.isin([n.type_node for n in self.neighbor], 
                            [Node_Type.SENSOR_NODE, Node_Type.CONNECTOR_NODE])
        cluster_mask = np.array([n.cluster_id == self.cluster_id for n in self.neighbor])
        potential_mask = type_mask & cluster_mask
        
        self.potentialSender = np.array(self.neighbor)[potential_mask].tolist()