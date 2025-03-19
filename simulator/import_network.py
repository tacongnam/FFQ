import csv, time, random, os, sys, copy, yaml
import numpy as np
from scipy.stats import sem, t
from tabulate import tabulate

sys.path.append(os.path.dirname(__file__))

from simulator.mobilecharger.info import MobileCharger
from simulator.network.info import Network
from simulator.node.cluster import Cluster
from simulator.node.target import Target
from optimizer.qlearning import Qlearning

from simulator.node.node_info import Node_Type
from simulator.node.in_node import InNode
from simulator.node.out_node import OutNode
from simulator.node.sensor_node import SensorNode
from simulator.node.relay_node import RelayNode
from simulator.node.connector_node import ConnectorNode

from simulator import parameters as para

class Simulation:
    '''
        Store simulation data
    '''

    def __init__(self, data_location):
        with open(data_location, 'r') as file:
            self.net_argc = yaml.safe_load(file)
        self.net_argc = copy.deepcopy(self.net_argc)

    def network_init(self, q_alpha, q_gamma):
        self.com_range = self.net_argc['node_phy_spe']['com_range']
        self.sen_range = self.net_argc['node_phy_spe']['sen_range']
        self.prob = self.net_argc['node_phy_spe']['prob_gp']
        self.nb_mc = 3
        self.package_size = self.net_argc['node_phy_spe']['package_size']
        self.theta = 0.1
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.energy = self.net_argc['node_phy_spe']['capacity']
        self.energy_max = self.net_argc['node_phy_spe']['capacity']
        self.node_pos = self.net_argc['nodes']
        self.energy_thresh = 0.4 * self.energy

    def targets_init(self):
        list_clusters = {}
        target_pos = []
        clusters_data = self.net_argc['Clusters']

        list_clusters[-1] = Cluster(-1, para.base)
        target_id = 0

        for cluster in clusters_data:
            list_clusters[int(cluster['cluster_id'])] = Cluster(int(cluster['cluster_id']),  cluster['centroid'])
            
            for target in cluster['list_targets']:
                new_target = Target(target_id, target, int(cluster['cluster_id']))
                target_pos.append(new_target)
                target_id += 1

        print('[Simulator] Build targets: Done')

        return list_clusters, target_pos
    
    def sensors_init(self, list_clusters):
        list_node = []

        # Build connector node
        connector_node_data = self.net_argc['ConnectorNode']

        for node in connector_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = ConnectorNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.CONNECTOR_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build connector node: Done')

        # Build in node
        in_node_data = self.net_argc['InNode']

        for node in in_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = InNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.IN_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            gen_node.init_inNode()
            list_node.append(gen_node)
        
        # print('Build Sensors - Build in node: Done')
        
        # Build out node
        out_node_data = self.net_argc['OutNode']

        for node in out_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = OutNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.OUT_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build out node: Done')
        
        # Build sensor node
        sensor_node_data = self.net_argc['SensorNode']

        for node in sensor_node_data:
            cluster = int(node['cluster_id'])
            location = node['location']
            gen_node = SensorNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.SENSOR_NODE, cluster_id=cluster, centroid=list_clusters[cluster].centroid)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build sensor node: Done')
        
        # Build relay node
        relay_node_data = self.net_argc['RelayNode']

        for node in relay_node_data:
            receive_cluster_id = list_clusters[int(node['receive_cluster_id'])]
            send_cluster_id = list_clusters[int(node['send_cluster_id'])]
            location = node['location']
            gen_node = RelayNode(location=location, com_ran=self.com_range, sen_ran=self.sen_range, energy=self.energy, 
                                     energy_max=self.energy_max, id=len(list_node), energy_thresh=self.energy_thresh, prob=self.prob, 
                                     type_node=Node_Type.RELAY_NODE, cluster_id=-1, centroid=None, receive_cluster_id=receive_cluster_id, send_cluster_id=send_cluster_id)
            
            list_node.append(gen_node)

        # print('Build Sensors - Build relay node: Done')

        print('[Simulator] Build sensors: Done')
        
        return list_node
    
    def run_simulator(self, run_times, E_mc, num_test):
        try:
            os.makedirs('log')
        except FileExistsError:
            pass
        try:
            os.makedirs('fig')
        except FileExistsError:
            pass

        output_file = open("log/q_learning_Kmeans.csv", "w")
        result = csv.DictWriter(output_file, fieldnames=["nb_run", "lifetime", "dead_node"])
        result.writeheader()
            
        life_time = []

        # Initialize Test case
        test_begin = 0
        test_end = num_test
            
        for nb_run in range(run_times):
            random.seed(nb_run)

            print("[Simulator] Repeat ", nb_run, ":")

            # Initialize Sensor Nodes and Targets
            list_clusters, target_pos = self.targets_init()
            list_node = self.sensors_init(list_clusters)

            # Initialize Mobile Chargers with Q / Double Q
            mc_list = []
            for id in range(self.nb_mc):
                if nb_run < test_begin + 1:
                    mc = MobileCharger(id, energy=E_mc, capacity=E_mc, e_move=1, e_self_charge=540, velocity=5, depot_state = para.n_clusters, double_q=False)
                    mc_list.append(mc)
                else:
                    mc = MobileCharger(id, energy=E_mc, capacity=E_mc, e_move=1, e_self_charge=540, velocity=5, depot_state = para.n_clusters, double_q=False)
                    mc_list.append(mc)


            # Construct Network
            net_log_file = "log/network_log_new_network_{}.csv".format(nb_run)
            MC_log_file = "log/MC_log_new_network_{}.csv".format(nb_run)
            experiment = "{}_eweight_{}".format(nb_run, para.e_weight)

            # Record the start time
            start_time = time.time()

            net = Network(list_node=list_node, mc_list=mc_list, target=target_pos, experiment=experiment, com_range=self.com_range, list_clusters=list_clusters)
                
            # Initialize Q-learning Optimizer
            q_learning = Qlearning(net=net, nb_action=para.n_clusters, theta=self.theta, q_alpha=self.q_alpha, q_gamma=self.q_gamma)

            if nb_run == test_end:
                para.e_weight += 1
                test_begin = test_end + 1
                test_end = test_begin + num_test
            
            print("[Simulator] Initializing experiment, repetition {}:\n".format(nb_run))
            print("[Simulator] Network:")
            print(tabulate([['Sensors', len(net.node)], ['Targets', len(net.target)], ['Package Size', self.package_size], ['Sending Freq', self.prob], ['MC', self.nb_mc]], headers=['Parameters', 'Value']), '\n')
            print("[Simulator] Optimizer:")
            print(tabulate([['Alpha', q_learning.q_alpha], ['Gamma', q_learning.q_gamma], ['Theta', q_learning.theta]], headers=['Parameters', 'Value']), '\n')

            # Define log file
            file_name = "log/q_learning_Kmeans_new_network_{}.csv".format(nb_run)
            with open(file_name, "w") as information_log:
                writer = csv.DictWriter(information_log, fieldnames=["time", "nb_dead_node", "nb_package"])
                writer.writeheader()
            
            sim = net.simulate(optimizer=q_learning, t=0, dead_time=0)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")

            life_time.append(sim[0])
            result.writerow({"nb_run": nb_run, "lifetime": sim[0], "dead_node": sim[1]})

        confidence = 0.95
        h = sem(life_time) * t.ppf((1 + confidence) / 2, len(life_time) - 1)
        result.writerow({"nb_run": np.mean(life_time), "lifetime": h, "dead_node": 0})

        return net