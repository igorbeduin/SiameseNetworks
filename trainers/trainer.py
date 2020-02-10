import abc


class Trainer(abc.ABC):
    def __init__(self, model=None):
        self.model = model
        self.__train = False
        self.__loop_count = 0
        self.epochs = 1000
        self.dataset = None

    @property
    def training_enabled(self):
        return self.__train

    @property
    def cur_loop(self):
        return self.__loop_count

    @abc.abstractmethod
    def loop_train(self, dataset):
        pass

    def enable_training(self):
        self.__train = True
    
    def disable_training(self):
        self.__train = False

    def count(self):
        self.__loop_count += 1

