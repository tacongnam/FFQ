import numpy as np
from sklearn.cluster import KMeans  # For initial centroids (optional)

from simulator.node.node import Node
from simulator import parameters as para

class Clusters:
    def __init__(self):
        self.num_clusters = para.n_clusters
        self.clusters = None
        self.centroids = None
        self.membership = None
        
        self.last_charging_time = None

    def fuzzy_c_means(self, network, m=2, max_iter=100, error=1e-5):
        n = len(network.node)
        c = self.num_clusters

        self.membership = np.random.dirichlet(np.ones(c), size=n)

        X = np.array([s.location for s in network.node])
        Y = np.array([s.avg_energy**0.05 for s in network.node])
        d = np.linalg.norm(Y)
        Y = Y/d
        
        kmeans = KMeans(n_clusters=c, random_state=0).fit(X, sample_weight=Y)
        self.centroids = kmeans.cluster_centers_

        for _ in range(max_iter):
            old_membership = self.membership.copy()
                
            # Update centroids 
            um = self.membership ** m
            numerator = np.sum(um[:, :, np.newaxis] * X[:, np.newaxis, :], axis=0)
            denominator = np.sum(um, axis=0)
            self.centroids = numerator / denominator[:, np.newaxis]

            diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
                
            dist = np.maximum(dist, 1e-10)
            power = 2/(m-1)
            dist_power = dist ** (-power)
            self.membership = dist_power / np.sum(dist_power, axis=1)[:, np.newaxis]

            if np.linalg.norm(self.membership - old_membership) < error:
                break

        self.clusters = [[] for _ in range(c)]
        self.sensor_cluster_membership = {}

        for i, sensor in enumerate(network.node):
            self.sensor_cluster_membership[sensor.id] = []
            mask = self.membership[i] >= para.membership_threshold
            for j in np.where(mask)[0]:
                self.clusters[j].append(sensor)
                self.sensor_cluster_membership[sensor.id].append((j, self.membership[i, j]))

        self.last_charging_time = np.zeros(self.num_clusters, dtype=np.float32)

        print(f"Clustering successfully! - {len(self.clusters)} clusters")

    def energy_depletion_issue(self, sensor: Node, centroid):
        """Calculate energy depletion issue for a sensor."""
        # Factors:
        # 1. Residual energy (lower = worse)
        # 2. Energy consumption rate (higher = worse)
        # 3. Distance to centroid (closer = worse, if in range)
        
        dist = np.linalg.norm(sensor.location - centroid)
        if dist > sensor.cha_ran:
            return 0  # Not considered if out of charging range
        
        # Normalize values (assuming max values for scaling)
        max_energy = sensor.energy_max  # Example max residual energy (Joules)
        max_rate = 1      # Example max consumption rate (Joules/second)
        
        # Energy depletion metric (higher = bigger issue)
        # Weight factors can be adjusted
        w1, w2, w3 = 0.4, 0.4, 0.2  # Weights sum to 1
        residual_factor = 1 - (sensor.energy / max_energy)  # 0 to 1 (lower energy = higher value)
        rate_factor = sensor.avg_energy / max_rate             # 0 to 1 (higher rate = higher value)
        dist_factor = 1 - (dist / sensor.cha_ran)            # 0 to 1 (closer = higher value)
        
        return w1 * residual_factor + w2 * rate_factor + w3 * dist_factor

    def update_centers(self):
        """Update cluster centers based on energy depletion."""
        for i in range(self.num_clusters):
            centroid = self.centroids[i]
            # Find sensor with max energy depletion issue in charging range
            max_issue = -1
            new_center = centroid
            
            for sensor in self.clusters[i]:
                issue = self.energy_depletion_issue(sensor, centroid)
                if issue > max_issue:
                    max_issue = issue
                    new_center = sensor.location
            
            self.centroids[i] = new_center

    def get_charging_pos(self):
        charging_pos = []
        for centroid in self.centroids:
            charging_pos.append([int(centroid[0]), int(centroid[1])])
        charging_pos.append(para.depot)

        return charging_pos
    
    def charging_history_reward(self, network):
        time_aspect = np.array([(network.t - last) / network.t for last in self.last_charging_time])
        
        critical_aspect = []
        charging_aspect = []
        total_charged_amount = 0

        for sensor in network.node:
            total_charged_amount += sensor.charged

        for cluster in self.clusters:
            n_critical = 0
            charged_amount = 0

            for sensor in cluster:
                if sensor.energy < sensor.energy_thresh:
                    n_critical += 1
                charged_amount += sensor.charged
            
            critical_aspect.append(n_critical / len(network.node))
            
            if total_charged_amount == 0:   # Haven't charged
                charging_aspect.append(0)
            else:
                charging_aspect.append(charged_amount / total_charged_amount)

        return time_aspect + critical_aspect - charging_aspect