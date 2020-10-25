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


class Yalefaces(Dataset):
    def __init__(self, raw_data_path="/Users/igorbeduin/Desktop/FIT/Databases/yalefaces"):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.dataset_name = "yalefaces"
        self.ext = ".png"

    def raw_load(self, split=None):
        if split is not None:
            self.split = split
        print(" ====== (RAW) LOADING IMAGES ======")
        i = 1
        train_no = 0
        eval_no = 0
        for (_, _, files_list) in os.walk(self.raw_data_path, topdown=True):
            split_index = int(self.split * len(files_list))
            shuffled_files_list = files_list.copy()
            random.shuffle(shuffled_files_list)
            for file in shuffled_files_list:
                if self.ext in file:
                    index = files_list.index(file)
                    print(f"({i},{len(files_list)})")
                    # file is in format subjectNNxx...xx.gif
                    # so we grab the 0 to 9 characters to get the subj number
                    subject = file[:9]
                    file_raw_data_path = os.path.join(self.raw_data_path, file)
                    image = np.array(Image.open(file_raw_data_path))
                    image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
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



if __name__ == "__main__":
    dataset = Yalefaces()
    dataset.load()
    print(type(dataset.get()))

