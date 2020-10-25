'''
'''
import os
import pickle
import abc
try:
    from settings import Settings
except ModuleNotFoundError:
    from arch_model.settings import Settings


class Dataset(abc.ABC):
    def __init__(self):
        self.dataset_path = os.path.join(os.getcwd(), "dataset/data/")
        self.input_shape = Settings().input_shape
        self.dataset_train = {}
        self.dataset_eval = {}
        self.split = 0.9
        self.dataset_name = None
        self.show_images = False
        self.save_in_disk = True

    @abc.abstractmethod
    def raw_load(self):
        '''
        Load raw dataset in a dictionary tree structure:
            "Class1":
                [List of images]
            "Class2":
                [List of images]
                ...
        '''
        pass

    def save_dataset(self):
        data_train_path_name = os.path.join(self.dataset_path, self.dataset_name + str(self.split) + "_train.pickle")
        with open(data_train_path_name, "wb") as f:
            pickle.dump(self.dataset_train, f)
        print("Training dataset pickled in disk!")
        data_eval_path_name = os.path.join(self.dataset_path, self.dataset_name + str(self.split) + "_eval.pickle")
        with open(data_eval_path_name, "wb") as f:
            pickle.dump(self.dataset_eval, f)
        print("Evaluation dataset pickled in disk!")

    def load(self, path=None, split=None):
        if split is not None:
            self.split = split
        print(" ====== LOADING DATASET ======")
        if path is None:
            path = self.dataset_path
        with open(path + self.dataset_name + str(self.split) + "_train.pickle", "rb") as f:
            self.dataset_train = pickle.load(f)
        with open(path + self.dataset_name + str(self.split) + "_eval.pickle", "rb") as f:
            self.dataset_eval = pickle.load(f)
        print("Done!")

    def get(self, t=None, split=None):
        if split is not None:
            self.split = split
        if not self.dataset_eval or not self.dataset_train:
            self.load()
        if t == "train":
            return self.dataset_train
        if t == "eval":
            return self.dataset_eval
        if t is None:
            return (self.dataset_train, self.dataset_eval)
