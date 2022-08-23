import torch 
import os
import numpy as np
import tqdm
import pandas as pd

from fairness_metrics import compute_fairness_metrics_batch_multiple_class
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_ROC(preds, labels, save_dir):

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = metrics.roc_curve(labels, preds.detach().numpy())
    roc_auc[0] =  metrics.auc(fpr[0], tpr[0])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ =  metrics.roc_curve(labels.detach().numpy().ravel(), preds.detach().numpy().ravel())
    roc_auc["micro"] =  metrics.auc(fpr["micro"], tpr["micro"])
    lw=2
    plt.figure()
    plt.plot(
        fpr[0],
        tpr[0],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[0],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig(os.path.join(save_dir,'ROC_adv.png'))
    
def plot_class_ROC(preds, labels, n_classes, save_dir, logits=False):
    
         
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(labels[:, i], preds.detach().numpy()[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(labels.detach().numpy().ravel(), preds.detach().numpy().ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    lw = 2

    #colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir,'ROC_class.png'))


def inference(gan_model, val_set, batch_size, logits):
    gan_model.eval()
    
    preds_cla = []
    preds_adv = []
    with torch.no_grad():
        x = val_set[0]
        y = val_set[1]
        s = val_set[2]
        val_index = np.arange(x.shape[0])
        val_batches = [(i * batch_size, min(x.shape[0], (i + 1) * batch_size))
                                 for i in range((x.shape[0] + batch_size - 1) // batch_size)]
        for iter, (batch_start, batch_end) in tqdm.tqdm(enumerate(val_batches)):
            batch_ids = val_index[batch_start:batch_end]
            x_batch = x[batch_start:batch_end]
            y_batch = y[batch_start:batch_end]
            s_batch = s[batch_start:batch_end]

            pred = gan_model(x_batch)
            pred_cla = pred[0]
            pred_adv = pred[1]
            preds_cla.extend(list(pred_cla))
            preds_adv.extend(list(pred_adv))
            
    preds_cla = torch.stack(preds_cla)
    preds_adv = torch.stack(preds_adv)
    
    if logits:
        preds_cla = torch.sigmoid(preds_cla)
        
    data = [preds_cla, preds_adv]
    
    return data
    
def eval_class(pred_cla, class_indexs, val_set, batch_size, logits=False, verbose=False):
    val_log = [[] for i in range(len(class_indexs))]
    
    X, Y, S = val_set
    val_index = np.arange(len(val_set[1]))
    val_batches = [(i * batch_size, min(X.shape[0], (i + 1) * batch_size))
                                 for i in range((X.shape[0] + batch_size - 1) // batch_size)]

    pred_cla = pred_cla.round()
    
    for class_index in class_indexs:
         for iter, (batch_start, batch_end) in tqdm.tqdm(enumerate(val_batches)):
            batch_ids = val_index[batch_start:batch_end]
            x_batch = pred_cla[batch_start:batch_end]
            y_batch = Y[batch_start:batch_end]
            s_batch = S[batch_start:batch_end]

            mask_pos_class = y_batch[:, class_index] == 1
            mask_neg_class = y_batch[:, class_index] == 0

            pred_pos = x_batch[mask_pos_class]
            y_pos = y_batch[mask_pos_class]
            pred_neg = x_batch[mask_neg_class]
            y_neg = y_batch[mask_neg_class]

            acc_cla = (sum(x_batch[:, class_index]==y_batch[:, class_index]).float()/len(x_batch)).cpu().detach().numpy()
                
            if len(pred_pos) == 0:
                acc_cla_pos_class = np.NAN
            else:
                acc_cla_pos_class = (sum(pred_pos[:, class_index]==y_pos[:, class_index]).float()/len(pred_pos)).cpu().detach().numpy()
                  
            acc_cla_neg_class = (sum(pred_neg[:, class_index]==y_neg[:, class_index]).float()/len(pred_neg)).cpu().detach().numpy()

            val_log[class_index].append((acc_cla, acc_cla_pos_class, acc_cla_neg_class))
        

    result_df = pd.DataFrame(columns=['Class', 'Acc', 'Acc Pos', 'Acc Neg'])
    result_df['Class'] = class_indexs
    acc = []
    acc_pos = []
    acc_neg = []
    
    for class_index in class_indexs:             
        val_log_class = np.nanmean(val_log[class_index], axis=0)
        
        acc.append(100*val_log_class[0])
        acc_pos.append(100*val_log_class[1])
        acc_neg.append(100*val_log_class[2])
        
        if verbose:
            print("Class - [%d] Validation - [acc: %.2f%%] - [acc_pos: %.2f%%] - [acc_neg: %.2f%%]"
                          % (class_index, 100*val_log_class[0], 100*val_log_class[1], 100*val_log_class[2]))
    
    result_df['Acc'] = acc
    result_df['Acc Pos'] = acc_pos
    result_df['Acc Neg'] = acc_neg
    
    return result_df
    


def eval_model(gan_model, val_set, dict_labels, save_dir, batch_size, logits):
    gan_model.cpu()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preds = inference(gan_model, val_set, batch_size, logits)
    
    fairness_result = compute_fairness_metrics_batch_multiple_class(preds[0], val_set, class_names = dict_labels.keys(), batch_size=batch_size)
    val_log_class = eval_class(preds[0], list(range(len(dict_labels))), val_set, batch_size=batch_size, logits=logits) 

    plot_class_ROC(preds[0], val_set[1], n_classes=len(dict_labels), save_dir=save_dir, logits=logits)
    plot_ROC(preds[1], val_set[2], save_dir=save_dir) 
    
    fairness_result.to_csv(os.path.join(save_dir, 'fairness_metrics.csv'))
    val_log_class.to_csv(os.path.join(save_dir, 'eval_class.csv'))
