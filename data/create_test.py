import yaml
import pandas as pd
from ast import literal_eval

df = pd.read_csv("data/node.csv")
node_pos = list(literal_eval(df.node_pos[0]))

list_node = []

list_target = []
connector_node = []

target = [int(item) for item in df.target[0].split(',')]

for target_id in target:
    location = list(node_pos[target_id])
    list_target.append(location)


for node_p in node_pos:
    node = {
        'cluster_id': 0, 
        'location': list(node_p)
    }
    connector_node.append(node)


data = {
    'Clusters': [
        {
            "centroid": [500.0, 500.0],
            "cluster_id": 0,
            "list_targets": list_target
        }
    ],
    'ConnectorNode': connector_node,
    'InNode': [],
    'OutNode': [],
    'RelayNode': [],
    'SensorNode': [],
    'nodes': []
}

data['node_phy_spe'] = {
  'capacity': 10,
  'com_range': 80.1,
  'sen_range': 40.1,
  'prob_gp': 1,
  'package_size': 400.0,
}

with open('data/test_750.yaml', 'w') as file:
    yaml.dump(data, file)
