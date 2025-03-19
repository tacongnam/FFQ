import math
from scipy.spatial import distance
import numpy as np

from simulator.node.node_info import Node_Type
from simulator.node.node import Node
from simulator import parameters as para

class RelayNode(Node):
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0, 
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None, type_node=-1, cluster_id = -1, centroid=None,
                 send_cluster_id=None,receive_cluster_id=None):
        
        super().__init__(location, com_ran, sen_ran, energy, prob, avg_energy, len_cp, id, is_active, energy_max, 
                         energy_thresh, type_node, cluster_id, centroid)
        self.send_cluster_id = send_cluster_id # id cluster gửi
        self.receive_cluster_id = receive_cluster_id # id cluster nhận
    
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
    
        for node in self.neighbor:
            distance_min = 10000007.0
            node_min = Node(id = -1)
            
            if node.is_active == True and (node.type_node == Node_Type.RELAY_NODE and self.send_cluster_id.id == node.send_cluster_id.id 
               and self.receive_cluster_id.id == node.receive_cluster_id.id):
                
                location_end = para.base
                if self.receive_cluster_id.id == -1:
                       location_end = para.base
                else: location_end = self.receive_cluster_id.centroid
                
                distance_1 = distance.euclidean(node.location, location_end)
                distance_2 = distance.euclidean(self.location, location_end)
                
                if distance_1 < distance_2 and distance.euclidean(node.location, self.location) < distance_min:
                    node_min = node
                    self.candidate = node
                    distance_min = distance.euclidean(node.location, self.location)

                if node_min.id != -1:
                    return node_min
              
        for node in self.neighbor:
            if  node.is_active == True and node.type_node == Node_Type.IN_NODE and node.cluster_id == self.receive_cluster_id.id:
            # if(node.__class__.__name__ == "InNode") and self.level > node.level:
                self.candidate = node
                return node

        return Node(id = -1)
    
    def probe_neighbors(self, network):
        self.neighbor.clear()
        self.potentialSender.clear()

        for node in network.node:
            if  node.is_active == True and self != node and distance.euclidean(node.location, self.location) <= self.com_ran:
                self.neighbor.append(node)
                if(node.type_node == Node_Type.RELAY_NODE and self.send_cluster_id.id == node.send_cluster_id.id 
                   and self.receive_cluster_id.id == node.receive_cluster_id.id):
                        self.potentialSender.append(node)
                        
                if(node.type_node == Node_Type.OUT_NODE and node.cluster_id == self.send_cluster_id.id):
                        self.potentialSender.append(node)