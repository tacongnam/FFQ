import math
import numpy as np
from scipy.spatial import distance

from simulator.node.utils import to_string, request_function
from simulator import parameters as para
from simulator.node.node_info import Node_Type

class Node:
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0,
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None, type_node=-1, cluster_id=-1, centroid=None):
        self.location = location  # location of sensor
        self.com_ran = com_ran  # communication range
        self.sen_ran = sen_ran  # sensing range
        self.cha_ran = para.cha_ran    # charging range

        self.energy = energy  # energy of sensor
        self.energy_max = energy_max  # capacity of sensor
        self.energy_thresh = energy_thresh  # threshold to sensor send request for mc
        self.used_energy = 0.0  # energy was used from last check point to now
        self.actual_used = 0.0
        self.avg_energy = avg_energy  # average energy of sensor
        self.prev = 0

        self.prob = prob  # probability of sending data
        self.len_cp = len_cp  # length of check point list
        self.id = id  # identify of sensor
        self.is_active = is_active  # statement of sensor. If sensor dead, state is False
        self.is_request = False
        
        self.level = 0
        self.cluster_id = cluster_id
        self.centroid = centroid

        self.type_node = Node_Type.UNSET if type_node == -1 else type_node
        self.neighbor = []  # neighborhood of sensor
        self.potentialSender = [] # là danh sách con của neighbor nhưng có khả năng gửi gói tin cho self
        self.listTargets = [] # danh sách các targets trong phạm vi có thể theo dõi, không tính tới việc sẽ theo dõi các targets này hay không

        self.sent_through = 0   # số package gửi qua
        self.charged = 0        # năng lượng đã sạc
        self.charged_added = 0  # năng lượng đã sạc trong giây
        self.charged_count = 0  # số giây đã sạc

        self.candidate = None

    def update_average_energy(self):
        """
        calculate average energy of sensor
        :param func: function to calculate
        :return: set value for average energy with estimate function is func
        """
        self.avg_energy = (self.energy - self.prev) / para.update_time
        self.prev = self.energy

    def charge(self, mc):
        """
        charging to sensor
        :param mc: mobile charger
        :return: the amount of energy mc charges to this sensor
        """
        if not (self.energy <= self.energy_max - 1e-5 and mc.is_stand and self.is_active):
            return 0

        d = distance.euclidean(self.location, mc.current)
        p_theory = para.alpha / (d + para.beta) ** 2
        p_actual = np.minimum(self.energy_max - self.energy, p_theory)

        self.energy += p_actual
        self.charged += p_actual
        self.charged_added += p_actual
        return p_actual

    def send(self, net=None, package=None, receiver=None):
        """
        send package
        :param package:
        :param net: the network
        :param receiver: the receiver node
        :return: send package to the next node and reduce energy of this node
        """
        d0 = math.sqrt(para.EFS / para.EMP)
        package.update_path(self.id)
        dist = distance.euclidean(self.location, para.base)
        pkg_size = package.size

        if dist > self.com_ran and receiver.id != -1:
            d = distance.euclidean(self.location, receiver.location)
            e_send = para.ET + (para.EFS * d ** 2 if d <= d0 else para.EMP * d ** 4)
            energy_used = e_send * pkg_size
        
            # Cập nhật năng lượng và thống kê
            self.energy -= energy_used
            self.used_energy += energy_used
            self.actual_used += energy_used
            if pkg_size > 0:
                self.sent_through += 1
        
            # Gửi tiếp
            receiver.receive(package)
            receiver.send(net, package, receiver=receiver.find_receiver(net=net))
        elif dist <= self.com_ran:
            package.is_success = True
            d = dist
            e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4

            self.energy -= e_send * package.size
            self.used_energy += e_send * package.size
            self.actual_used += e_send * package.size
            if package.size > 0:
                self.sent_through += 1
            package.update_path(-1)

        self.check_active(net)

    def receive(self, package):
        """
        receive package from other node
        :param package: size of package
        :return: reduce energy of this node
        """
        energy_used = para.ER * package.size
        self.energy -= energy_used
        self.used_energy += energy_used
        self.actual_used += energy_used

    def check_active(self, net):
        """
        check if the node is alive
        :param net: the network
        :return: None
        """
        if self.energy < 0 or not self.neighbor:
            self.is_active = False
        else:
            # Vector hóa kiểm tra neighbor active
            self.is_active = np.any([n.is_active for n in self.neighbor])

    def request(self, index, optimizer, t, request_func=request_function):
        """
        send a message to mc if the energy is below a threshold
        :param mc: mobile charger
        :param t: time to send request
        :param request_func: structure of message
        :return: None
        """
        # print(self.check_point)
        if not self.is_request:
            request_func(self, index, optimizer, t)
            self.is_request = True

    def print_node(self, func=to_string):
        """
        print node information
        :param func: print function
        :return:
        """
        func(self)
