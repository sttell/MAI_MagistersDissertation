import os


class Context:
    def __init__(self, args):
        self._args = args
        self._is_valid = False

        # Parse arguments
        self.dataset_path: os.PathLike = args.dataset_path
        self.config_path: os.PathLike = args.config_path

    def validate(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"Directory with dataset doesn't exists by path: {self.dataset_path}")
        if not os.path.isdir(self.dataset_path):
            raise ValueError(f"Path with dataset must be path to directory.")
        if not os.path.exists(self.config_path):
            raise ValueError(f"Directory with configuration files doesn't exists by path: {self.config_path}")
        if not os.path.isdir(self.config_path):
            raise ValueError(f"Path with configuration files must be path to directory.")
        self._is_valid = True

    def is_valid(self):
        return self._is_valid
