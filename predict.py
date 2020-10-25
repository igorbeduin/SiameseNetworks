import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from dataset.yalefaces import Yalefaces
from dataset.thermo import Thermo
from arch_model.arch_model import ArchModel
from testers import testers
from trainers.batch_trainer import BatchTrainer

def FPR(table):
    try:
        fpr = table["fp"]/(table["fp"] + table["tn"])
        return fpr
    except ZeroDivisionError:
        pass

def TPR(table):
    try:
        tpr = table["tp"]/(table["tp"] + table["fn"])
        return tpr
    except ZeroDivisionError:
        pass

if __name__ == "__main__":
    accuracies = []
    confusion_tables = []
    table_model = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    batch_size = 1000
    split = 0.8
    dataset_name = Thermo().dataset_name
    weights_path = f"training_weights/thermo/{split}"
    model_weigths = "weights2020-02-18 21.15.05.410953.h5"
    cuts = 1000
    cut_perc = np.linspace(0, 1, cuts)
    ROC = []

    (dataset_train, dataset_eval) = Thermo().get(split=split)
    model = ArchModel().get_arch('siam')
    model.load_weights(os.path.join(weights_path, model_weigths))

    datasets = [dataset_eval, dataset_train]

    for dataset in datasets:
        if dataset is dataset_train:
            label = "TRAINING"
        elif dataset is dataset_eval:
            label = "EVALUATION"

        print("Getting batch of images...")
        (pairs, targets) = BatchTrainer().get_batch(batch_size=batch_size, dataset=dataset)
        probs = model.predict(pairs)
        targets = targets.tolist()
        probs = probs.tolist()


        for cut in cut_perc:
            print("---------------------------------")
            print(f"Cut-off: {cut}")
            _table = table_model.copy()
            (tp, fp, tn, fn) = testers.predictions(probs, targets, cut)
            _table["tp"] = tp
            _table["fp"] = fp
            _table["tn"] = tn
            _table["fn"] = fn
            acc = (_table["tp"] + _table["tn"])/(_table["tp"] + _table["tn"] + _table["fn"] + _table["fp"])
            accuracies.append(acc)
            print(_table)
            print(f"FPR: {FPR(_table)}, TPR: {TPR(_table)}")
            print(f"Accuracy: {acc}")
            confusion_tables.append(_table)

        with open(os.path.join(os.getcwd(), "confusion_tables.pickle"), "wb") as f:
            pickle.dump(confusion_tables, f)

        ROC_x = np.array(list(FPR(table) for table in confusion_tables))
        ROC_y = np.array(list(TPR(table) for table in confusion_tables))
        ROC = np.array([ROC_x, ROC_y])

        plt.subplot(2, 1, 1)
        plt.plot(ROC_x, ROC_y, 'r')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.title(f'ROC Curve - {split} split | {label} {dataset_name} dataset')

        plt.subplot(2, 1, 2)
        a = plt.plot(cut_perc[1:], accuracies[1:])
        plt.ylabel('Accuracy')
        plt.xlabel('Cut-off')
        plt.show()

        accuracies.clear()
        confusion_tables.clear()

