from trainers.batch_trainer import BatchTrainer
import numpy as np
import matplotlib.pyplot as plt


def predictions(probs, targets, cut, v=0):
    tp, fp = 0, 0
    tn, fn = 0, 0

    for prob in probs:
        if prob[0] >= cut and targets[probs.index(prob)] == 1:
            tp += 1
        elif prob[0] >= cut and targets[probs.index(prob)] == 0:
            fp += 1
        elif prob[0] < cut and targets[probs.index(prob)] == 0:
            tn += 1
        elif prob[0] < cut and targets[probs.index(prob)] == 1:
            fn += 1
    return (tp, fp, tn, fn)


