import numpy as np
import random
from time import time
try:
    from trainer import Trainer
except ImportError:
    from trainers.trainer import Trainer


class BatchTrainer(Trainer):
    def __init__(self, model=None):
        super().__init__(model)

    def __apply_transformations(self, image):
        h, w = image.shape
        image = image.reshape(h, w, 1)
        return image

    def get_batch(self, batch_size=32, dataset=None, transform=True):
        if dataset is None:
            _dataset = self.dataset
        else:
            _dataset = dataset
        targets = np.zeros((batch_size, ))
        targets[:batch_size//2] = 1
        random.shuffle(targets)
        input_1, input_2 = [], []
        for target in targets:
            if target == 1:
                key = random.choice(list(_dataset.keys()))
                while len(_dataset[key]) == 1:
                    key = random.choice(list(_dataset.keys()))
                image_1 = random.choice(_dataset[key])
                image_2 = random.choice(_dataset[key])
                while (image_2 == image_1).all():
                    image_2 = random.choice(_dataset[key])
            else:
                key_1 = random.choice(list(_dataset.keys()))
                key_2 = random.choice(list(_dataset.keys()))
                while (key_2 == key_1):
                    key_2 = random.choice(list(_dataset.keys()))
                image_1 = random.choice(_dataset[key_1])
                image_2 = random.choice(_dataset[key_2])

            if transform is True:
                image_1 = self.__apply_transformations(image_1)
                image_2 = self.__apply_transformations(image_2)
            input_1.append(image_1)
            input_2.append(image_2)

        pairs = [np.array(input_1), np.array(input_2)]
        return (pairs, targets)

    def train_on_batch(self):
        (pairs, targets) = self.get_batch()
        loss = self.model.train_on_batch(pairs, targets)
        print(f"Current loss: {loss}")
        self.loss = round(loss, 2)

    def loop_train(self, dataset, loops=None, checkpoint=None):
        if checkpoint is not None:
            self.checkpoint = checkpoint
        if loops is not None:
            self.epochs = loops
        print(" ====== STARTING TRAINING ======")
        start_time = time()
        self.dataset = dataset
        while self.training_enabled:
            self.count()
            print(f"Current loop: ({self.cur_loop}/{self.epochs})")
            self.train_on_batch()
            if self.cur_loop == self.epochs:
                self.disable_training()
            cur_time = round((time() - start_time)/60, 2)
            print(f"Time since training started: {cur_time} mins")
            if self.save_weights:
                self.save(self.add_path, tag=self.loss)
            print("------------------------------------------------")

    @property
    def save_weights(self):
        if self.cur_loop % self.checkpoint == 0:
            return True
        else:
            return False
