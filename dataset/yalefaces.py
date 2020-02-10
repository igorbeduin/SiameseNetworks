'''
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import shutil
from PIL import Image
import pandas as pd

try:
    from dataset import Dataset
except ImportError:
    from dataset.dataset import Dataset


class Yalefaces(Dataset):
    def __init__(self, path="/Users/igorbeduin/Desktop/FIT/Databases/yalefaces"):
        super().__init__()
        self.path = path
        self.database = {}

    def load(self):
        for (_, _, filesList) in os.walk(self.path, topdown=True):
            for file in filesList:
                if (".png") in file:
                    # file is in format subjectNNxx...xx.gif
                    # so we grab the 0 to 9 characters to get the subject_number
                    subject = file[:9]
                    file_path = os.path.join(self.path, file)
                    image = np.array(Image.open(file_path))
                    image = cv2.resize(image, self.input_shape)
                    # plt.imshow(image)
                    # plt.show()
                    if subject not in self.database:
                        self.database[subject] = [image]
                    else:
                        self.database[subject].append(image)
        # plt.imshow(self.database['subject08'][0])
        # plt.show()

    def get(self):
        return self.database

if __name__ == "__main__":
    dataset = Yalefaces()
    dataset.load()
    print(type(dataset.get()))
    #df = pd.DataFrame(dataset.get())
    #print(df.head())
