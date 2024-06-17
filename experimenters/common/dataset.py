from sklearn.model_selection import train_test_split
from typing import Set, Tuple
import cv2
import os


class Dataset:
    def __init__(self, path: os.PathLike, is_auto_upload=True):
        self.images_extensions = ["png", "bmp", "jpg", "jpeg"]
        self.path = os.path.abspath(path)

        self.images = []
        self.labels = []
        self.possible_labels: Set[str] = set()

        if is_auto_upload:
            self.upload()

    def _load_label_directories(self):
        # Add possible labels
        subdirectories = os.listdir(self.path)
        for subdirectory in subdirectories:
            subdirectory_name = os.path.basename(subdirectory)
            if os.path.isdir(subdirectory) and (not subdirectory_name.startswith('.')):
                self.possible_labels.add(subdirectory_name)

    def _load_data(self):
        for label in self.possible_labels:
            files = os.listdir(os.path.join(self.path, label))
            for file in files:
                if not file.split('.')[-1] in self.images_extensions:
                    continue
                self.images.append(cv2.imread(file))
                self.labels.append(label)

    def upload(self):
        self._load_label_directories()
        self._load_data()

    def split(self, configuration) -> Tuple['Dataset', 'Dataset', 'Dataset']:
        # Split to Train + (Validation & Test)
        X_train, X_test_valid, Y_train, Y_test_valid = train_test_split(
            self.images,
            self.labels,
            shuffle=configuration["shuffle"],
            train_size=(configuration["train"]) / 100,
            test_size=(configuration["test"] + configuration["validation"]) / 100
        )
        train_dataset = Dataset(self.path)
        train_dataset.images = X_train
        train_dataset.labels = Y_train
        train_dataset.possible_labels = set(train_dataset.labels)

        # Recalculate ratio for Test and Validation sets
        test_ratio = configuration["test"] / (configuration["test"] + configuration["valid"])
        valid_ratio = configuration["valid"] / (configuration["test"] + configuration["valid"])

        # Split Validation & Test
        X_test, X_valid, Y_test, Y_valid = train_test_split(
            X_test_valid,
            Y_test_valid,
            shuffle=configuration["shuffle"],
            train_size=test_ratio,
            test_size=valid_ratio
        )
        test_dataset = Dataset(self.path)
        test_dataset.images = X_test
        test_dataset.labels = Y_test
        test_dataset.possible_labels = set(Y_test)

        valid_dataset = Dataset(self.path)
        valid_dataset.images = X_valid
        valid_dataset.labels = Y_valid
        valid_dataset.possible_labels = set(Y_valid)

        return train_dataset, test_dataset, valid_dataset