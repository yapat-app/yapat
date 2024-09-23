
from npeet.entropy_estimators import entropy as calculate_entropy

from evaluations import BaseEvaluation

class EntropyEvaluation(BaseEvaluation):
    def __init__(self):

        super().__init__()

    def calculate_result(self, dataset_id: int, embedding_id: int):

        data = self.load_data(dataset_id, embedding_id)
        self.scaled_data = self.scale_data(data)
        self.evaluation_result = calculate_entropy(self.scaled_data)
        #self.save_result(self.evaluation_result)
        return self.evaluation_result

entropy = EntropyEvaluation()
entropy.calculate_result(1, 2)