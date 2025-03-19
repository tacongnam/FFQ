import math
from scipy.spatial import distance
import numpy as np
import random

from simulator.node.node_info import Node_Type
from simulator.node.node import Node
from simulator import parameters as para

class InNode(Node):
    def init_inNode(self):
        self.out_node_list = []
        self.out_node_number = 1

        self.chosen_out_node_index = self.chosen_random_index()


        self.rr_current_unit = 0 # số lần liên tiếp gửi gói tin cho outnode trước khi chuyển sang node khác
        self.package_index = 0

        self.rr_max_unit = 2
        self.rr_max_cycle = 5
        self.max_package_index = self.out_node_number * self.rr_max_unit * self.rr_max_cycle

    def find_receiver(self, net):
        """
        find receiver node
        :param node: node send this package
        :param net: network
        :return: find node nearest base from neighbor of the node and return id of it
        """
        if not self.is_active:
            return Node(id = -1)
        
        self.get_out_node_list()

        if(self.package_index == self.max_package_index):
            self.package_index = 0
            self.rr_current_unit = 0 
            self.chosen_out_node_index = self.chosen_random_index()

        if(self.rr_current_unit == self.rr_max_unit):
            self.rr_current_unit = 0 
            self.chosen_out_node_index = (self.chosen_out_node_index + 1) % self.out_node_number

        self.rr_current_unit = self.rr_current_unit + 1
        self.package_index = self.package_index + 1

        return self.out_node_list[self.chosen_out_node_index]
    
    def probe_neighbors(self, network):
        self.neighbor.clear()
        self.potentialSender.clear()

        for node in network.node:
            if node.is_active == True and self != node and distance.euclidean(node.location, self.location) <= self.com_ran:
                self.neighbor.append(node)
                if node.type_node == Node_Type.RELAY_NODE and self.cluster_id == node.receive_cluster_id.id:
                    self.potentialSender.append(node)
                    
                if(node.type_node in [Node_Type.SENSOR_NODE, Node_Type.CONNECTOR_NODE] and self.cluster_id == node.cluster_id):
                    self.potentialSender.append(node)

    def chosen_random_index(self):
        if(self.out_node_number == 1):
            return 0
        index = random.randint(0, self.out_node_number - 1)
        return index
    
    def get_out_node_list(self):
        for node in self.neighbor:
            if node.is_active == True and (node.type_node == Node_Type.OUT_NODE and node.is_active == True and self.cluster_id == node.cluster_id):
                self.out_node_list.append(node)
        self.out_node_number = len(self.out_node_list)
        self.max_package_index = self.out_node_number * self.rr_max_unit * self.rr_max_cycle