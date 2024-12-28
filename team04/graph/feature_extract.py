import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os 
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import csv
import sys
import numpy as np
import itertools

def plt_roc(y_pred, y_label):
    """
    plot the roc curve for the multi-classification 
    plot 3 classes result from the same model in the same image
    """
    # y_pred (0,1) prob for 14 classes, list
    # y_lable 0,1,2,..,13, list
    classes = range(0,14)
    needed_classes = [1, 2, 3]   ###### 
    class_name = ["Pneumonia", "Lung Opacity", "Lung Lesion"]  ####
    figure_file = "./roc_model1.jpg"

    fpr = []
    tpr = []
    roc_auc = []

    for c in needed_classes:
        now_class = [1 if y == c else 0 for y in y_label]
        now_prob = y_pred[:, c]

        tfpr, ttpr, _ = roc_curve(now_class, now_prob)
        t_roc_auc = auc(tfpr, ttpr)
        fpr.append(tfpr)
        tpr.append(ttpr)
        roc_auc.append(t_roc_auc)

    # Plot all ROC curves
    plt.figure()
    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(len(needed_classes)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of {0} (area = {1:0.2f})".format(class_name[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.savefig(figure_file)
    plt.show()


if __name__=="__main__":
    
    num_sample = 100#100
    num_model = 3
    y_preds = []
    y_labels = []
    ######### 14classes 100sampless

    for j in range(num_model):
        y_pred = []
        y_label = []
        for i in range(num_sample):
            a = np.random.dirichlet(np.ones(14),size=1)
            #print(a)
            y_pred.append(a[0])

            # generate lable
            x = np.random.uniform(0,1,1)
            if x < 0.60:
                y_label.append(np.argmax(a[0]))
            else:
                y_label.append(np.random.randint(0,13))
        
        y_pred = np.array(y_pred)
        y_label = np.array(y_label)
        y_preds.append(y_pred)
        y_labels.append(y_label)


    #draw_averroc(y_preds, y_labels)
    #draw_oneclass_roc_confi(y_preds[0], y_labels[0])
    draw_pr_curve(y_preds, y_labels)
