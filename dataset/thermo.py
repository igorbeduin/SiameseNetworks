'''
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random

try:
    from dataset import Dataset
except ImportError:
    from dataset.dataset import Dataset


class Thermo(Dataset):
    def __init__(self, raw_data_path="/content/drive/My Drive/datasets/EBD_THF"):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.dataset_name = "thermo"
        self.ext = ".jpg"

    def raw_load(self, split=None):
        # self.show_images = True
        full_files_list = []
        if split is not None:
            self.split = split
        print(" ====== (RAW) LOADING IMAGES ======")
        i = 1
        train_no = 0
        eval_no = 0
        for (dirs, sub_dirs, files_list) in os.walk(self.raw_data_path, topdown=True):
            for file in files_list:
                full_files_list.append(file)

        split_index = int(self.split * len(full_files_list))
        shuffled_files_list = full_files_list.copy()
        random.shuffle(shuffled_files_list)
        for file in shuffled_files_list:
            if self.ext in file:
                index = full_files_list.index(file)
                print(f"({i},{len(full_files_list)})")
                subject = file[:4]
                file_raw_data_path = os.path.join(self.raw_data_path, subject, file)
                image = np.array(Image.open(file_raw_data_path))
                image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                (h, w, ch) = image.shape
                if ch > 1:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if self.show_images:
                    plt.imshow(image)
                    plt.show()
                if (index <= split_index):
                    train_no += 1
                    if subject not in self.dataset_train:
                        self.dataset_train[subject] = [image]
                    else:
                        self.dataset_train[subject].append(image)
                else:
                    eval_no += 1
                    if subject not in self.dataset_eval:
                        self.dataset_eval[subject] = [image]
                    else:
                        self.dataset_eval[subject].append(image)
                i += 1
        print("Done!")
        if self.save_in_disk:
            self.save_dataset()
