# Libraries
import numpy as np
from scipy.spatial import distance

from simulator.network.info import Network
from simulator.mobilecharger.info import MobileCharger

from simulator import parameters as para
from optimizer.utils import get_charging_time, get_charge_per_sec, penalty_reward

BASE = -1

def q_max_function(q_table):
    temp = [max(row) for index, row in enumerate(q_table)]
    return np.asarray(temp)
    
def reward_function(network: Network, mc: MobileCharger, q_learning, state, time_stem):
    theta = q_learning.theta
    charging_time = get_charging_time(network, mc, q_learning, time_stem=time_stem, state=state, theta=theta)
    w, nb_target_alive = get_weight(network, mc, q_learning, state, charging_time)

    requests = q_learning.list_request
    node_ids = [request["id"] for request in requests]
    E = np.array([network.node[nid].energy for nid in node_ids])
    e = np.array([request["avg_energy"] for request in requests])

    p = get_charge_per_sec(network, q_learning, state)
    p_hat = p / np.sum(p)

    second = nb_target_alive / len(network.target)
    third = np.sum(w * p_hat)
    first = np.sum(e * p / E)

    return first, second, third, charging_time

def additional_reward_function(network: Network, mc: MobileCharger, q_learning):
    fourth = np.append(network.clusters.charging_history_reward(network), [0])      # No history for depot
    fifth = penalty_reward(network, mc, q_learning)

    return fourth, fifth

def init_function(nb_action=para.n_clusters):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def get_weight(net, mc, q_learning, action_id, charging_time):
    p = get_charge_per_sec(net, q_learning, action_id)
    all_path = q_learning.all_path
    time_move = distance.euclidean(q_learning.action_list[mc.state], q_learning.action_list[action_id]) / mc.velocity
    list_dead = []
    w = [0 for _ in q_learning.list_request]

    for request_id, request in enumerate(q_learning.list_request):
        temp = (net.node[request["id"]].energy - time_move * request["avg_energy"]) + (p[request_id] - request["avg_energy"]) * charging_time
        if temp < 0:
            list_dead.append(net.node[request["id"]].id)

    for request_id, request in enumerate(q_learning.list_request):
        nb_path = 0
        for path in all_path:
            if net.node[request["id"]].id in path:
                nb_path += 1
        w[request_id] = nb_path

    total_weight = sum(w) + len(w) * 10 ** -3
    w = np.asarray([(item + 10 ** -3) / total_weight for item in w])
    nb_target_alive = 0
    
    for path in all_path:
        if BASE in path and not (set(list_dead) & set(path)):
            nb_target_alive += 1
    return w, nb_target_alive