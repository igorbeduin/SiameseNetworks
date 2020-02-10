import numpy as np
try:
    from trainer import Trainer
except ImportError:
    from trainers.trainer import Trainer


class BatchTrainer(Trainer):
    def __init__(self, model=None):
        super().__init__(model)

    def __apply_transformations(self):
        # TODO: reshape image to right dimensions
        pass

    def get_batch(self):
        # TODO: randomly construct de list of pairs for batch training
        sub1_0 = np.array([self.dataset['subject01'][0]])
        _, w, h = sub1_0.shape
        sub1_0 = sub1_0.reshape(w, h, 1)

        sub1_1 = np.array([self.dataset['subject01'][1]])
        _, w, h = sub1_1.shape
        sub1_1 = sub1_1.reshape(w, h, 1)
        self.__apply_transformations()
        # print(sub1_0.shape, sub1_1.shape)

        sub2_0 = np.array([self.dataset['subject02'][0]])
        _, w, h = sub2_0.shape
        sub2_0 = sub2_0.reshape(w, h, 1)

        sub2_1 = np.array([self.dataset['subject02'][1]])
        _, w, h = sub2_1.shape
        sub2_1 = sub2_1.reshape(w, h, 1)
        # print(sub2_0.shape, sub2_1.shape)
        
        pairs = [np.array([sub1_0, sub1_1]), np.array([sub2_0, sub2_1])]
        targets = np.array([0, 0])
        return (pairs, targets)

    def train_on_batch(self):
        (pairs, targets) = self.get_batch()
        oss = self.model.train_on_batch(pairs, targets)
        print(oss)

    def loop_train(self, dataset):
        self.epochs = 1
        self.dataset = dataset
        while self.training_enabled:
            self.count()
            print(f"Training... ({self.cur_loop})")
            self.train_on_batch()
            if self.cur_loop == self.epochs:
                self.disable_training()
