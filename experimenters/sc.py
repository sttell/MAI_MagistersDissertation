from experimenters.common.operations import apply_bilateral_filter, apply_sobel, apply_grayscale, apply_resize, apply_flatten
from experimenters.common.classifier_model import ClassifierModel
from experimenters.common.experiment_base import ExperimentBase
from experimenters.common.dataset import Dataset
from experimenters.context import Context

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
import numpy as np

import json
import os


class SCExperiment(ExperimentBase):
    def __init__(self, ctx: Context):
        ExperimentBase.__init__(self, ctx)
        with open(os.path.join(ctx.config_path, "sc_experiment_config.json"), 'r') as file:
            self.config = json.load(file)

    def _preprocess_image(self, image: np.ndarray, hyperparams):
        image = apply_bilateral_filter(image, hyperparams["bilateral_filter"])
        image = apply_grayscale(image)
        image = apply_resize(image, hyperparams["resize"])
        image = apply_sobel(image, hyperparams["sobel_filter"])
        image = apply_flatten(image)
        return (image - np.mean(image)) / np.std(image)

    def _preprocess_data(self, dataset):
        hyperparams = self.config['hyperparams']
        dataset.images = [self._preprocess_image(image, hyperparams) for image in dataset.images]

    def run(self):
        dataset = Dataset(self.ctx.dataset_path)
        self._preprocess_data(dataset)
        train_dataset, test_dataset, validation_dataset = dataset.split(self.config["dataset_split"])
        models = {
            "SVC": ClassifierModel(
                hyperparams=self.config["model_params_grid"]["svc"],
                model_class=svm.SVC
            ),
            "Random Forest": ClassifierModel(
                hyperparams=self.config["model_params_grid"]["random_forest"],
                model_class=RandomForestClassifier
            ),
            "Gradient Boosting on Decision Trees": ClassifierModel(
                hyperparams=self.config["model_params_grid"]["gbdt"],
                model_class=GradientBoostingClassifier
            )
        }
        best_models = self._train_models(models, train_dataset, test_dataset)
        self._evaluate_models(best_models, validation_dataset)

    @staticmethod
    def get_name() -> str:
        return "SC"
