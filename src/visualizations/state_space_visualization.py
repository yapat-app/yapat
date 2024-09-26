from visualizations import BaseVisualization


class StateSpaceVis(BaseVisualization):

    def __init__(self, result_id):


        super().__init__(result_id)

    def plot(self):
        embeddings, cluster_labels = self.load_data()
        pass


