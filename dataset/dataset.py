'''
'''
import abc
try:
    from settings import Settings
except ModuleNotFoundError:
    from arch_model.settings import Settings


class Dataset(abc.ABC):
    def __init__(self):
        self.input_shape = (105, 105)

    @abc.abstractmethod
    def load(self):
        pass
