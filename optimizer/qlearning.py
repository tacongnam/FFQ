import numpy as np
from optimizer.rewards import reward_function, additional_reward_function
from optimizer.utils import init_function, FLCDS_model, get_all_path, q_max_function
from simulator import parameters as para

class Qlearning:
    def __init__(self, net, init_func=init_function, nb_action=para.n_clusters, theta=0, q_alpha=0.5, q_gamma=0.5):
        self.action_list = []
        self.nb_action = nb_action
        self.q_table = init_func(nb_action=nb_action)
        self.q1 = init_func(nb_action=nb_action)
        self.q2 = init_func(nb_action=nb_action)
        
        self.charging_time = [0.0 for _ in range(nb_action + 1)]
        self.reward = np.asarray([0.0 for _ in range(nb_action + 1)])
        self.reward_max = [0.0 for _ in range(nb_action + 1)]
        self.list_request = []
        
        self.theta = para.theta
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma

        self.FLCDS = FLCDS_model(network=net)
        self.all_path = get_all_path(net=net)

    def update_all_path(self, net):
        self.all_path = get_all_path(net=net)

    def update(self, mc, network, time_stem, q_max_func=q_max_function, reward_func=reward_function, doubleq=True):
        if not len(self.list_request):
            return self.action_list[mc.state], -1.0
        
        if mc.state < para.n_clusters:
            network.clusters.last_charging_time[mc.state] = network.t

        if doubleq == True:
            if np.random.rand() < 0.5:
                self.set_reward(q_table=self.q1, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
                self.q1[mc.state] =  (1 - self.q_alpha) * self.q1[mc.state] + self.q_alpha * (self.reward + self.q_gamma * self.q_max(mc, self.q2, q_max_func))
            else:
                self.set_reward(q_table=self.q2, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
                self.q2[mc.state] =  (1 - self.q_alpha) * self.q2[mc.state] + self.q_alpha * (self.reward + self.q_gamma * self.q_max(mc, self.q1, q_max_func))
            self.q_table[mc.state] = (self.q1[mc.state] + self.q2[mc.state]) / 2
        else:
            self.set_reward(q_table=self.q_table, mc=mc,time_stem=time_stem, reward_func=reward_func, network=network)
            self.q_table[mc.state] =  (1 - self.q_alpha) * self.q_table[mc.state] + self.q_alpha * (self.reward + self.q_gamma * self.q_max(mc, self.q_table, q_max_func))

        self.choose_next_state(mc, self.q_table)

        if mc.state == len(self.action_list) - 1:
            charging_time = (mc.capacity - mc.energy) / mc.e_self_charge
        else:
            charging_time = self.charging_time[mc.state]
        
        if charging_time > 1:
            print("[Optimizer] MC #{} is sent to point {} (id={}) and charge for {:.2f}s".format(mc.id, self.action_list[mc.state], mc.state, charging_time))

        # print(self.charging_time)
        return self.action_list[mc.state], charging_time
    
    def q_max(self, mc, table, q_max_func=q_max_function):
        return q_max_func(q_table=table)
    
    def set_reward(self, q_table, mc = None, time_stem=0, reward_func=reward_function, network=None):   
        energy = np.asarray([0.0 for _ in self.action_list], dtype=float)
        connect = np.asarray([0.0 for _ in self.action_list], dtype=float)
        cover = np.asarray([0.0 for _ in self.action_list], dtype=float)
        history = np.asarray([0.0 for _ in self.action_list], dtype=float)
        penalty = np.asarray([0.0 for _ in self.action_list], dtype=float)
        
        for index, row in enumerate(q_table):
            reward = reward_func(network=network, mc=mc, q_learning=self, state=index, time_stem=time_stem)
            energy[index] = reward[0]
            connect[index] = reward[1]
            cover[index] = reward[2]
            self.charging_time[index] = reward[3]

        energy = energy / np.sum(energy)
        connect = connect / np.sum(connect)
        cover = cover / np.sum(cover)

        history, penalty = additional_reward_function(network=network, mc=mc, q_learning=self)

        # Tính reward vector hóa
        self.reward = (para.energy_q * energy + para.connect_q * connect + 
                    para.cover_q * cover + para.history_q * history - 
                    para.penalty_q * penalty)
        
        # Lưu reward_max nếu cần
        self.reward_max = list(zip(energy, connect, cover, history, penalty))

    def choose_next_state(self, mc, table):
        # next_state = np.argmax(self.q_table[mc.state])
        if mc.energy < para.E_mc_thresh:
            mc.state = len(table) - 1
            print('[Optimizer] MC #{} energy is running low ({:.2f}), and needs to rest!'.format(mc.id, mc.energy))
        else:
            mc.state = np.argmax(table[mc.state])
            # print(self.reward_max[mc.state])
            # print(self.action_list[mc.state])