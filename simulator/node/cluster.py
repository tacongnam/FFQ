class Cluster:
    def __init__(self, id, centroid):
        self.centroid = centroid
        self.id = id
    
    def __repr__(self):
        return "Cluster {}: {}".format(self.id, self.centroid)