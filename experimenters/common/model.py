from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score

from experimenters.common.dataset import Dataset


def evaluate(model, dataset, positive_label='0'):
    predicts = model.predict(dataset.images)
    return {
        "f1": f1_score(dataset.labels, predicts, pos_label=positive_label),
        "recall": recall_score(dataset.labels, predicts, pos_label=positive_label),
        "precision": precision_score(dataset.labels, predicts, pos_label=positive_label),
        "accuracy": accuracy_score(dataset.labels, predicts)
    }


class Model:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.best_model = None
        self.best_result = {"metrics": {"f1": 0.0}}

    def train(self, train_dataset: Dataset, test_dataset: Dataset):
        pass