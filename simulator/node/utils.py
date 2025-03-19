import numpy as np
from scipy.spatial import distance
from simulator import parameters as para

def to_string(node):
    """
    print information of a node
    :param node: sensor node
    :return: None
    """
    print("Id =", node.id, "Location =", node.location, "Energy =", node.energy, "ave_e =", node.avg_energy,
          "Neighbor =", node.neighbor)

def find_receiver(node):
    """
    find receiver node
    :param node: node send this package
    :param net: network
    :return: find node nearest base from neighbor of the node and return id of it
    """
    if not node.is_active:
        return -1
        
    candidate = [neighbor for neighbor in node.neighbor if
                 neighbor.level < node.level and neighbor.is_active]
    if candidate:
        d = [distance.euclidean(node.location, para.base) for node in candidate]
        id_min = np.argmin(d)
        return candidate[id_min]
    else:
        return -1


def request_function(node, index, optimizer, t):
    """
    add a message to request list of mc.
    :param node: the node request
    :param mc: mobile charger
    :param t: time get request
    :return: None
    """
    optimizer.list_request.append(
        {"id": index, "energy": node.energy, "avg_energy": node.avg_energy, "energy_estimate": node.energy,
         "time": t})