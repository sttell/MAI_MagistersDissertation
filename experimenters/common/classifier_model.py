from experimenters.common.model import Model, evaluate
from experimenters.common.dataset import Dataset


class ClassifierModel(Model):
    def __init__(self, hyperparams, model_class):
        Model.__init__(self, hyperparams)
        self.model_class = model_class

    def _get_params_variants(self):
        return somelib.combine(self.hyperparams) # (NDA lib)

    def train(self, train_dataset: Dataset, test_dataset: Dataset):
        param_variants = self._get_params_variants()
        for params in param_variants:
            clf = self.model_class(**params)
            clf.fit(train_dataset.images, train_dataset.labels)

            metrics = evaluate(clf, test_dataset)

            if metrics['f1'] > self.best_result["metrics"]["f1"]:
                self.best_result["metrics"] = metrics
                self.best_result["params"] = params
                self.best_model = clf