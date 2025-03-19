import csv
import numpy as np
from simulator.network.package import Package
from simulator import parameters as para

def uniform_com_func(net):
    targets = net.target
    if not targets:
        return True
    
    send_mask = np.random.random(len(targets)) <= para.send_probability
    active_targets = np.array(targets)[send_mask]

    for target in active_targets:
        # Lấy danh sách sensor active
        sensors = [n[0] for n in target.listSensors if n[0].is_active]
        if not sensors:
            continue
        
        # Chọn sensor đầu tiên active và gửi
        sensor = sensors[0]
        package = Package(is_energy_info=False)
        sensor.send(net, package, receiver=sensor.find_receiver(net))
    
    return True

def to_string(net):
    min_energy = 10 ** 10
    min_node = -1
    for node in net.node:
        if node.energy < min_energy:
            min_energy = node.energy
            min_node = node
    min_node.print_node()

def count_package_function(net):
    targets = net.target
    if not targets:
        return 0
    
    count = 0
    for target in targets:
        # Lấy danh sách sensor active
        sensors = [n[0] for n in target.listSensors if n[0].is_active]
        if not sensors:
            continue
        
        # Gửi package từ sensor đầu tiên active
        sensor = sensors[0]
        temp_package = Package(is_energy_info=True)
        sensor.send(net, temp_package, receiver=sensor.find_receiver(net))
        
        if temp_package.path[-1] == -1:
            count += 1
    
    return count

def show_info(network=None, mi=0, avg=0, cha=0, past_dead=0, past_package=0, optimizer=None):
    if network != None:
        print("\n[Network] Simulating time: {}s, lowest energy node: {:.4f}, used: {:.4f}, charged: {:.4f} at {} (id = {})".format(network.t, network.node[mi].energy, network.node[mi].actual_used, network.node[mi].charged, network.node[mi].location, mi))
        print('\t\t-----------------------')
        print('\t\tAverage used of each node: {:.6f}, average each node per second: {:.6f}'.format(avg, avg / network.t))
        print('\t\tAverage charged of each node this second: {:.6f}'.format(cha))
        print('\t\tNumber of dead nodes: {}'.format(past_dead))
        print('\t\tNumber of packages: {}'.format(past_package))
        print('\t\t-----------------------\n')

        for mc in network.mc_list:
            print("\t\tMC #{} is {} at {} with energy {}".format(mc.id, mc.get_status(), mc.current, mc.energy))
                
        network_info = {
            'time_stamp' : network.t,
            'number_of_dead_nodes' : past_dead,
            'number_of_monitored_target' : past_package,
            'lowest_node_energy': round(network.node[mi].energy, 3),
            'lowest_node_location': network.node[mi].location,
            'theta': optimizer.theta,     
            'avg_energy': network.get_average_energy(),
            'average_used_of_each_node': avg,
            'average_used_of_each_node_this_second': avg / network.t,
            'average_charged_of_each_node_per_time': cha,
                
            'MC_0_status' : network.mc_list[0].get_status(),
            'MC_1_status' : network.mc_list[1].get_status(),
            'MC_2_status' : network.mc_list[2].get_status(),
            'MC_0_location' : network.mc_list[0].current,
            'MC_1_location' : network.mc_list[1].current,
            'MC_2_location' : network.mc_list[2].current,
        }
            
        with open(network.net_log_file, 'a') as information_log:
            node_writer = csv.DictWriter(information_log, fieldnames=['time_stamp', 'number_of_dead_nodes', 'number_of_monitored_target', 'lowest_node_energy', 'lowest_node_location', 'theta', 'avg_energy', 'average_used_of_each_node', 'average_used_of_each_node_this_second', 'average_charged_of_each_node_per_time', 'MC_0_status', 'MC_1_status', 'MC_2_status', 'MC_0_location', 'MC_1_location', 'MC_2_location'])
            node_writer.writerow(network_info)