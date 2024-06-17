from experimenters.common.operations import apply_bilateral_filter, apply_sobel, apply_grayscale, apply_resize
from experimenters.common.classifier_model import ClassifierModel
from experimenters.common.experiment_base import ExperimentBase
from experimenters.common.dataset import Dataset
from experimenters.context import Context

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as svm
import numpy as np
import cv2

import json
import os


class CMCExperiment(ExperimentBase):
    def __init__(self, ctx: Context):
        ExperimentBase.__init__(self, ctx)
        with open(os.path.join(ctx.config_path, "cmc_experiment_config.json"), 'r') as file:
            self.config = json.load(file)

    def _generate_filters(self):
        def draw_circle_bold(image, center_pt, radius, width):
            result = cv2.circle(image, center_pt, radius, (255, 255, 255), -1)
            result = cv2.circle(image, center_pt, radius - width, (0, 0, 0), -1)
            return result
        hyperparams = self.config["hyperparams"]["pattern"]

        image_size = (
            self.config["hyperparams"]["resize"]["major_size"],
            self.config["hyperparams"]["resize"]["major_size"]
        )
        filters = []

        for radius in hyperparams['radius']:
            for x_shift in hyperparams['x_shift']:
                for y_shift in hyperparams['y_shift']:
                    for width in hyperparams['width']:
                        zeros_img = np.zeros(image_size, dtype=np.uint8)
                        filter_img = draw_circle_bold(
                            zeros_img,
                            (image_size[0] // 2 + x_shift, image_size[0] // 2 + y_shift),
                            radius,
                            width
                        )
                        filter_img = cv2.GaussianBlur(filter_img, (3, 3), 50.0, 50.0)
                        filters.append(filter_img)

        return np.array(filters)

    def _get_corr_map(self, image, filters):
        def corr(filter_, image_):
            x = filter_.astype(np.float32)
            y = image_.astype(np.float32)
            x_mean = np.average(x)
            y_mean = np.average(y)
            return np.sum((x - x_mean) * (y - y_mean)) / np.sqrt(
                np.sum(np.power(x - x_mean, 2)) * np.sum(np.power(y - y_mean, 2)))

        return [corr(image, fil) for fil in filters]

    def _preprocess_image(self, image, hyperparams):
        image = apply_bilateral_filter(image, hyperparams["bilateral_filter"])
        image = apply_resize(image, hyperparams["resize"])
        image = apply_grayscale(image)
        return image.astype(np.uint8)

    def _preprocess_data(self, dataset):
        filters = self._generate_filters()
        dataset.images = [ self._preprocess_image(image, self.config["hyperparams"]) for image in dataset.images]
        dataset.images = [ self._get_corr_map(image, filters) for image in dataset.images ]

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
        return "CMC"