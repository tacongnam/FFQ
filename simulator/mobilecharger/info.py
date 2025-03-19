import csv
from scipy.spatial import distance
import numpy as np
from simulator.mobilecharger.utils import get_location, charging
from simulator import parameters as para


class MobileCharger:
    def __init__(self, id, energy=None, e_move=None, start=para.depot, end=para.depot, velocity=None,
                 e_self_charge=None, capacity=None, depot_state=para.n_clusters, double_q=True):
        self.id = id

        self.is_stand = False  # is true if mc stand and charge
        self.is_self_charge = False  # is true if mc is charged
        self.is_active = False

        self.start = start  # from location
        self.end = end  # to location
        self.current = start  # location now
        self.end_time = -1
        self.moving_time = 0
        self.arrival_time = 0

        self.energy = energy  # energy now
        self.capacity = capacity  # capacity of mc
        self.e_move = e_move  # energy for moving
        self.e_self_charge = e_self_charge  # energy receive per second
        self.velocity = velocity  # velocity of mc
        self.state = depot_state # Current state in Q_table

        self.double_q = double_q
        
        if self.double_q == True:
            print("MC", self.id, "enable double q-learning")
        else:
            print("MC", self.id, "enable single q-learning")

        self.log_buffer = []

    def get_status(self):
        if not self.is_active:
            return "deactivated"
        if not self.is_stand:
            return "moving"
        if not self.is_self_charge:
            return "charging"
        return "self_charging" 

    def update_location(self, func=get_location):
        self.current = func(self)
        self.energy -= self.e_move

    def charge(self, net=None, func=charging):
        func(self, net)

    def self_charge(self):
        self.energy = min(self.energy + self.e_self_charge, self.capacity)

    def check_state(self):
        if distance.euclidean(self.start, self.end) > 1 and distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        elif distance.euclidean(self.current, self.end) >= 1:
            self.is_stand = False

        if distance.euclidean(para.depot, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False

    def get_next_location(self, network, time_stem, optimizer=None, update_path=False):
        if update_path == True:
            optimizer.update_all_path(network)

        next_location, charging_time = optimizer.update(self, network, time_stem, doubleq=self.double_q)
        if charging_time == -1:
            return 
        
        self.start = self.current
        self.end = next_location
        self.moving_time = distance.euclidean(self.start, self.end) / self.velocity
        self.end_time = time_stem + self.moving_time + charging_time
        self.arrival_time = time_stem + self.moving_time

        # if self.end != [0.0, 0.0] and self.moving_time != 0:
        #    print("[Mobile Charger] MC #{} moves to {} in {}s and charges for {}s".format(self.id, self.end, self.moving_time, charging_time))
        # elif self.end == [0.0, 0.0]:
        #    print("[Mobile Charger] MC #{} is self-charge for {}s".format(self.id, self.moving_time + charging_time))

        mc_info = {
            'time_stamp': time_stem,
            'id': self.id,
            'starting_point': self.start,
            'destination_point': self.end,
            'decision_id': self.state,
            'charging_time': charging_time,
            'moving_time': self.moving_time
        }
        self.log_buffer.append(mc_info)

        # Ghi buffer khi đầy (ví dụ: 100 bản ghi)
        if len(self.log_buffer) >= 100:
            with open(network.mc_log_file, "a") as mc_log_file:
                writer = csv.DictWriter(mc_log_file, fieldnames=['time_stamp', 'id', 'starting_point', 
                                                                'destination_point', 'decision_id', 
                                                                'charging_time', 'moving_time'])
                writer.writerows(self.log_buffer)
            self.log_buffer.clear()

    def flush_buffer(self, log_file):
        if self.log_buffer:
            with open(log_file, "a") as f:
                writer = csv.DictWriter(f, fieldnames=['time_stamp', 'id', 'starting_point', 
                                                    'destination_point', 'decision_id', 
                                                    'charging_time', 'moving_time'])
                writer.writerows(self.log_buffer)
            self.log_buffer.clear()

    def run(self, time_stem, net=None, optimizer=None, update_path=False):
        if ((not self.is_active) and optimizer.list_request) or abs(time_stem - self.end_time) < 1:
            self.is_active = True
            
            request_ids = np.array([request["id"] for request in optimizer.list_request])
            energies = np.array([net.node[rid].energy for rid in request_ids])
            thresh = np.array([net.node[rid].energy_thresh for rid in request_ids])

            keep_mask = energies < thresh
            optimizer.list_request = [optimizer.list_request[i] for i in np.where(keep_mask)[0]]
            
            for rid in request_ids[~keep_mask]:
                net.node[rid].is_request = False

            if not optimizer.list_request:
                self.is_active = False

            self.get_next_location(network=net, time_stem=time_stem, optimizer=optimizer, update_path=update_path)
        else:
            if self.is_active:
                if not self.is_stand:
                    self.update_location()
                elif not self.is_self_charge:
                    self.charge(net)
                else:
                    self.self_charge()

        if self.energy < para.E_mc_thresh and not self.is_self_charge and self.end != para.depot:
            self.start = self.current
            self.end = para.depot
            self.is_stand = False
            charging_time = self.capacity / self.e_self_charge
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()
