import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import csv
import sys
import numpy as np
import itertools

def draw_pr_curve(y_preds, y_labels):
    """
    draw different models curve
    """
    needed_class = 2   ###### 
    class_name = "Pneumonia"  ####
    figure_file = "./pr_model1.jpg"
    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])###line color
    plt.figure()
    i = 0
    for y_pred, y_label, color in zip(y_preds,y_labels,colors):
        now_class = [1 if y == needed_class else 0 for y in y_label]
        now_prob = y_pred[:, needed_class]
        tp,tr,_ = precision_recall_curve(now_class, now_prob)
        t_aver_prec = average_precision_score(now_class, now_prob)
        plt.plot(
                tr,
                tp,
                color=color,
                lw=2,
                label="PR curve of model{0} (area = {1:0.2f})".format(i,t_aver_prec),
            )
        i=i+1

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for class{0}'.format(class_name))
    plt.legend(loc=(0, -.38))
    plt.savefig(figure_file)
    plt.show()


def confid_inter(c_pred,c_label):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    tprs = []
    fprs = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(c_pred), len(c_pred))
        # print(indices.shape)
        # print(indices)
        # print(type(c_label[indices]), type(c_pred[indices]))
        # print(c_label[indices][0:10])
        # print(c_pred[indices][0:10])
        tfpr,ttpr,_ = roc_curve(c_label[indices].tolist(), c_pred[indices].tolist())
        fprs.append(tfpr)
        #print(len(ttpr))
        tprs.append(ttpr)

    # First aggregate all false positive rates  # x 
    all_fpr = np.unique(np.concatenate([fprs[i] for i in range(n_bootstraps)]))

    # Then interpolate all ROC curves at this points
    all_samelength_tpr = []
    for i in range(n_bootstraps):
        all_samelength_tpr.append(np.interp(all_fpr, fprs[i], tprs[i]))

    std_tpr = np.std(all_samelength_tpr, axis=0)

    return std_tpr, all_fpr

def get_average_roc(y_pred, y_label):
    """
    get prediction and label for all classes
    return the average fpr,tpr,auc
    """
    classes = range(0,14)
    n_classes = len(classes)
    needed_classes = classes
    #class_name = ["Pneumonia", "Lung Opacity", "Lung Lesion"]  ####
    figure_file = "./average_roc_model1.jpg"

    fpr = []
    tpr = []

    for c in needed_classes:
        now_class = [1 if y == c else 0 for y in y_label]
        now_prob = y_pred[:, c]

        tfpr, ttpr, _ = roc_curve(now_class, now_prob)
        t_roc_auc = auc(tfpr, ttpr)
        fpr.append(tfpr)
        tpr.append(ttpr)
    
    fpr = np.array(fpr, dtype=object)
    tpr = np.array(tpr, dtype=object)
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)



def draw_oneclass_roc_confi(y_pred, y_label):
    """
    draw the roc curve with confidence interval for just one model

    """
    classes = range(0,14)
    needed_classes = 2  ######  index of the class 
    class_name = "Pneumonia" ####   the name of the class
    model_name = "baseline" ##### the model name
    figure_file = "./roc_confi_model1.jpg"    

    now_class = [1 if y == needed_classes else 0 for y in y_label]
    now_prob = y_pred[:, needed_classes]

    tfpr, ttpr, _ = roc_curve(now_class, now_prob)
    t_roc_auc = auc(tfpr, ttpr)

    #######get confid_inter
    now_class = np.array(now_class)
    now_prob = np.array(now_prob)
    # get confidence interval parameters
    std_tpr, all_fpr = confid_inter(now_prob,now_class)

    # shorter std
    new_std_tpr = np.interp(tfpr, all_fpr, std_tpr)
    tprs_upper = np.minimum(ttpr + new_std_tpr, 1)
    tprs_lower = np.maximum(ttpr - new_std_tpr, 0)

    # Plot all ROC curves
    #plt.figure()
    fig, ax = plt.subplots()
    #colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])
    ax.plot(
        tfpr,
        ttpr,
        color="aqua",
        lw=2,
        label="ROC curve of {0} {1} (area = {2:0.2f})".format(model_name, class_name, t_roc_auc),
    )

    ax.fill_between(
        tfpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
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



def draw_averroc(y_preds, y_labels):
    """
    draw average roc from different models on one imge

    """
    figure_file = "./aver_roc.jpg"    ### figure file name
    colors = itertools.cycle(["aqua", "darkorange", "cornflowerblue"])###line color
    plt.figure()
    i=0
    for y_pred, y_label, color in zip(y_preds,y_labels,colors):
        aver_fpr,aver_tpr,aver_auc = get_average_roc(y_pred, y_label)
        plt.plot(
                aver_fpr,
                aver_tpr,
                color=color,
                lw=2,
                label="Average ROC curve of model{0} (area = {1:0.2f})".format(i,aver_auc),
            )
        i=i+1

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Average ROC curve")
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










    
    

    
    
    
