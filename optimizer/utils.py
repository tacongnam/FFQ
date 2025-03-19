# Libraries
import numpy as np
from scipy.spatial import distance
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import defaultdict

# Modules
from simulator import parameters as para
from simulator.network.info import Network
from simulator.mobilecharger.info import MobileCharger

def init_function(nb_action):
    return np.zeros((nb_action + 1, nb_action + 1), dtype=float)

def FLCDS_model(network=None):
    max_energy = network.node[0].energy_max
    
    E_min = ctrl.Antecedent(np.linspace(0, max_energy, num = 10001), 'E_min')
    L_r = ctrl.Antecedent(np.arange(0, len(network.node) + 1), 'L_r')
    Theta = ctrl.Consequent(np.linspace(0, 1, num = 101), 'Theta')

    L_r['L'] = fuzz.trapmf(L_r.universe, [0, 0, 2, 6])
    L_r['M'] = fuzz.trimf(L_r.universe, [2, 6, 10])
    L_r['H'] = fuzz.trapmf(L_r.universe, [6, 10, len(network.node), len(network.node)])

    E_min['L'] = fuzz.trapmf(E_min.universe, [0, 0, 0.25 * max_energy, 0.5 * max_energy])
    E_min['M'] = fuzz.trimf(E_min.universe, [0.25 * max_energy, 0.5 * max_energy, 0.75 * max_energy])
    E_min['H'] = fuzz.trapmf(E_min.universe, [0.5 * max_energy, 0.75 * max_energy, max_energy, max_energy])

    Theta['VL'] = fuzz.trimf(Theta.universe, [0, 0, 1/3])
    Theta['L'] = fuzz.trimf(Theta.universe, [0, 1/3, 2/3])
    Theta['M'] = fuzz.trimf(Theta.universe, [1/3, 2/3, 1])
    Theta['H'] = fuzz.trimf(Theta.universe, [2/3, 1, 1])

    R1 = ctrl.Rule(L_r['L'] & E_min['L'], Theta['H'])
    R2 = ctrl.Rule(L_r['L'] & E_min['M'], Theta['M'])
    R3 = ctrl.Rule(L_r['L'] & E_min['H'], Theta['L'])
    R4 = ctrl.Rule(L_r['M'] & E_min['L'], Theta['M'])
    R5 = ctrl.Rule(L_r['M'] & E_min['M'], Theta['L'])
    R6 = ctrl.Rule(L_r['M'] & E_min['H'], Theta['VL'])
    R7 = ctrl.Rule(L_r['H'] & E_min['L'], Theta['L'])
    R8 = ctrl.Rule(L_r['H'] & E_min['M'], Theta['VL'])
    R9 = ctrl.Rule(L_r['H'] & E_min['H'], Theta['VL'])

    FLCDS_ctrl = ctrl.ControlSystem([R1, R2, R3, R4, R5, R6, R7, R8, R9])
    FLCDS = ctrl.ControlSystemSimulation(FLCDS_ctrl)

    return FLCDS

BASE = -1

def build_graph(net):
    graph = defaultdict(list)
    base_sensors = set()

    # Lấy tất cả sensor từ net.target một lần
    sensors = [(sensor, target) for target in net.target for sensor, _ in target.listSensors]
    sensor_locations = np.array([sensor.location for sensor, _ in sensors])
    sensor_ids = np.array([sensor.id for sensor, _ in sensors])
    
    # Vector hóa tính khoảng cách tới base
    base_distances = distance.cdist(sensor_locations, [para.base], metric='euclidean').flatten()
    com_ran = np.array([sensor.com_ran for sensor, _ in sensors])
    base_connected = base_distances <= com_ran
    
    # Thêm sensor kết nối với base
    for idx in np.where(base_connected)[0]:
        sensor_id = sensor_ids[idx]
        base_sensors.add(sensor_id)
        graph[sensor_id].append(-1)
    
    # Vector hóa tìm receiver
    for idx, (sensor, _) in enumerate(sensors):
        receiver = sensor.find_receiver(net=net)
        if receiver.id != -1 and receiver.id != sensor.id:
            graph[sensor.id].append(receiver.id)
    
    return graph, base_sensors

def get_path(graph, base_sensors, sensor_id, memo=None):
    if memo is None:
        memo = {}
    
    if sensor_id in memo:
        return memo[sensor_id]
    if sensor_id in base_sensors:
        return [sensor_id, -1]
    if sensor_id not in graph:
        return []
    
    # Sử dụng stack thay vì đệ quy
    stack = [(sensor_id, [sensor_id])]  # (node, path)
    visited = {sensor_id}
    
    while stack:
        current_id, path = stack.pop()
        
        for next_id in graph[current_id]:
            if next_id == -1:
                full_path = path + [-1]
                memo[sensor_id] = full_path
                return full_path
            
            if next_id not in visited:
                visited.add(next_id)
                stack.append((next_id, path + [next_id]))
    
    memo[sensor_id] = []
    return []

def get_all_path(net):
    graph, base_sensors = build_graph(net)
    list_path = []
    
    # Lấy tất cả sensor từ các target
    target_sensors = [(target, sensor) for target in net.target for sensor, _ in target.listSensors]
    
    # Tìm path cho từng target
    memo = {}  # Chia sẻ memo giữa các lần gọi get_path
    for target, sensor in target_sensors:
        path = get_path(graph, base_sensors, sensor.id, memo)
        if path and path[-1] == -1:
            list_path.append(path)
            break
    else:
        list_path.append([])  # Nếu không tìm thấy path cho target
    
    return list_path

def get_charge_per_sec(net, q_learning, state):
    # Lấy vị trí của các node trong list_request
    node_locations = np.array([net.node[request["id"]].location for request in q_learning.list_request])
    
    action_location = q_learning.action_list[state]
    
    # Vector hóa tính khoảng cách
    distances = distance.cdist(node_locations, [action_location], metric='euclidean').flatten()
    return para.theta / (distances + para.beta) ** 2

import numpy as np
from scipy.spatial import distance

def get_charging_time(network=None, mc=None, q_learning=None, time_stem=0, state=None, theta=0.1):
    time_move = distance.euclidean(mc.current, q_learning.action_list[state]) / mc.velocity

    # request_id = [request["id"] for request in network.mc.list_request]
    FLCDS = q_learning.FLCDS

    energy_threshold = 0.4 * network.node[0].energy_max

    L_r_crisp = len(q_learning.list_request)
    
    E_min_crisp = network.node[network.find_min_node()].energy

    FLCDS.input['L_r'] = L_r_crisp
    FLCDS.input['E_min'] = E_min_crisp
    FLCDS.compute()
    alpha = FLCDS.output['Theta']
    q_learning.alpha = alpha
    
    energy_min = np.max([energy_threshold + alpha * (network.node[0].energy_max - energy_threshold),
                         E_min_crisp + alpha * (network.node[0].energy_max - E_min_crisp)])

    #energy_min = para.q_theta * network.node[0].energy_max
    
    s1 = []  # list of node in request list which has positive charge
    s2 = []  # list of node not in request list which has negative charge
    for node in network.node:
        d = distance.euclidean(q_learning.action_list[state], node.location)
        p = para.alpha / (d + para.beta) ** 2
        p1 = 0
        for other_mc in network.mc_list:
            if other_mc.id != mc.id and other_mc.get_status() == "charging":
                d = distance.euclidean(other_mc.current, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - time_stem)
            elif other_mc.id != mc.id and other_mc.get_status() == "moving" and other_mc.state != len(q_learning.q_table) - 1:
                d = distance.euclidean(other_mc.end, node.location)
                p1 += (para.alpha / (d + para.beta) ** 2)*(other_mc.end_time - other_mc.arrival_time)
        
        if node.energy - time_move * node.avg_energy + p1 < energy_min and p - node.avg_energy > 0:
            s1.append((node, p, p1))
        if node.energy - time_move * node.avg_energy + p1 > energy_min and p - node.avg_energy < 0:
            s2.append((node, p, p1))
    
    t = []
    for node, p, p1 in s1:
        t.append((energy_min - node.energy + time_move * node.avg_energy - p1) / (p - node.avg_energy))
    for node, p, p1 in s2:
        t.append((energy_min - node.energy + time_move * node.avg_energy - p1) / (p - node.avg_energy))
    
    dead_list = [] 
    for item in t:
        nb_dead = 0
        for node, p, p1 in s1:
            temp = node.energy - time_move * node.avg_energy + p1 + (p - node.avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        for node, p, p1 in s2:
            temp = node.energy - time_move * node.avg_energy + p1 + (p - node.avg_energy) * item
            if temp < energy_min:
                nb_dead += 1
        dead_list.append(nb_dead)
    if dead_list:
        arg_min = np.argmin(dead_list)
        return t[arg_min]
    return 0

def q_max_function(q_table):
    temp = [max(row) for index, row in enumerate(q_table)]
    return np.asarray(temp)

def penalty_reward(network: Network, current_mc: MobileCharger, optimizer):
    action_locs = np.array(optimizer.action_list)
    mc_locs = np.array([optimizer.action_list[mc.state] for mc in network.mc_list])

    distances = distance.cdist(action_locs, mc_locs, metric='euclidean')
    
    mask = distances < 2 * para.cha_ran
    penalty_values = np.where(mask, 1 / np.maximum(1, distances), 0)
    penalty = np.sum(penalty_values, axis=1)
    
    return penalty