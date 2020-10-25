import abc
import datetime
import os


class Trainer(abc.ABC):
    def __init__(self, model=None):
        self.model = model
        self.__train = True
        self.__loop_count = 0
        self.epochs = 1
        self.dataset = None
        self.checkpoint = 50
        self.weights_path = "training_weights"
        self.saving_path = None
        self.add_path = None

    @property
    def training_enabled(self):
        return self.__train

    @property
    def cur_loop(self):
        return self.__loop_count

    @abc.abstractmethod
    def loop_train(self, dataset, loops=None, checkpoint=None):
        pass

    def enable_training(self):
        self.__train = True

    def disable_training(self):
        self.__train = False

    def count(self):
        self.__loop_count += 1

    def save(self, mid_path=None, tag=None):
        self.last_saved_weights = f'weights_{tag}_{datetime.datetime.now()}.h5'
        self.saving_path = self.weights_path
        if mid_path is not None:
            self.saving_path = os.path.join(self.weights_path, mid_path)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        self.model.save_weights(os.path.join(self.saving_path, self.last_saved_weights))
        print("Model weights saved in disk!")

    def load_model(self, weights=None):
        if weights is None:
            weights = self.last_saved_weights
        self.model.load_weights(weights)
        print("Model weights loaded in model!")

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def set_path_to_save(self, path):
        self.add_path = path
