import os
import pandas as pd
import numpy as np
from FairGAN import FairGAN
import torch
from evaluation import eval_model
import matplotlib.pyplot as plt
import re

        
def compare_fairness_classes(paths):
    results_parity = pd.DataFrame(columns=['Class'])
    results_opportunity = pd.DataFrame(columns=['Class'])
    results_ber = pd.DataFrame(columns=['Class'])
    results_odds = pd.DataFrame(columns=['Class'])
    
    #read eval files
    results = []
    col_names = []
    for i in range(len(paths)):
        results.append(pd.read_csv(os.path.join(paths[i], 'fairness_metrics.csv')))
        col_names.append(paths[i].split('/')[-2])
    
    results_parity['Class'] = results[0]['Class']
    results_opportunity['Class'] = results[0]['Class']
    results_ber['Class'] = results[0]['Class']
    results_odds['Class'] = results[0]['Class']
    
    for i, result in enumerate(results):
        results_parity[col_names[i]] = result['Statistical Parity']
        results_opportunity[col_names[i]] = result['Equality Opportunity']
        results_odds[col_names[i]] = result['Equal Odds']
        results_ber[col_names[i]] = result['BER']
    
    return results_parity, results_opportunity, results_odds, results_ber

def compare_accuracy_classes(paths):
    results_acc = pd.DataFrame(columns=['Class'])
    results_pos_class = pd.DataFrame(columns=['Class'])
    results_neg_class = pd.DataFrame(columns=['Class'])
    
    #read eval files
    results = []
    col_names = []
    for i in range(len(paths)):
        results.append(pd.read_csv(os.path.join(paths[i],'eval_class.csv')))
        col_names.append(paths[i].split('/')[-2])
    
    results_acc['Class'] = results[0]['Class']
    results_pos_class['Class'] = results[0]['Class']
    results_neg_class['Class'] = results[0]['Class']
    
    for i, result in enumerate(results):
        results_acc[col_names[i]] = result['Acc']
        results_pos_class[col_names[i]] = result['Acc Pos']
        results_neg_class[col_names[i]] = result['Acc Neg']
    
    return results_acc, results_pos_class, results_neg_class

def compute_average_fairness(paths, weights=None):
    average_results = pd.DataFrame(columns=['Model', 'Statistical Parity', 'Equality Opportunity', 'Equality Odds'])
    
    results = []
    col_names = []
    for i in range(len(paths)):
        results.append(pd.read_csv(os.path.join(paths[i],'fairness_metrics.csv')))
        #col_names.append(paths[i].split('/')[-3])
        col_names.append(paths[i].split('/')[-3] + paths[i].split('/')[-2])
    average_results['Model'] = col_names
    
    parity = []
    opportunity = []
    ber = []
    odds = []
    for result in results:
            parity.append(np.average(abs(result['Statistical Parity']), weights=weights))
            opportunity.append(np.average(abs(result['Equality Opportunity']), weights=weights))
            odds.append(np.average(abs(result['Equality Odds']), weights=weights))
            
    average_results['Statistical Parity'] = parity
    average_results['Equality Opportunity'] = opportunity
    average_results['Equality Odds'] = odds
    
    return average_results

def compute_average_acc(paths, weights=None):
    average_results_acc = pd.DataFrame(columns=['Model', 'Acc', 'Acc Pos', 'Acc Neg'])
    
    results = []
    col_names = []
    for i in range(len(paths)):
        results.append(pd.read_csv(os.path.join(paths[i],'eval_class.csv')))
        #col_names.append(paths[i].split('/')[-3])
       
        col_names.append(paths[i].split('/')[-3] + paths[i].split('/')[-2])
        
    average_results_acc['Model'] = col_names
    
    acc = []
    acc_pos = []
    acc_neg = []
    
    for result in results:
        acc.append(np.average(result['Acc'], weights=weights))
        acc_pos.append(np.average(result['Acc Pos'], weights=weights))
        acc_neg.append(np.average(result['Acc Neg'], weights=weights))
            
    average_results_acc['Acc'] = acc
    average_results_acc['Acc Pos'] = acc_pos
    average_results_acc['Acc Neg'] = acc_neg
    
    return average_results_acc

def compute_accuracy(gan_model):
    gan_model.eval()
    gan_model.cpu()
    
    pred_cla = gan_model(val_set[0])[0]
    
    if gan_model.logits:
        pred_cla = torch.sigmoid(pred_cla)

    accuracy = sum(pred_cla.round()==val_set[1]).float()/len(pred_cla)
    print("Accuracy for classifier: ", accuracy)

    mask_pos = val_set[1] == 1
    pred_cla_pos = pred_cla[mask_pos]
    label_pos = val_set[1][mask_pos]

    accuracy = sum(pred_cla_pos.round()==label_pos).float()/len(pred_cla_pos)
    print("Accuracy for classifier on pos class: ", accuracy)

    mask_neg = val_set[1] == 0
    pred_cla_neg = pred_cla[mask_neg]
    label_neg = val_set[1][mask_neg]
    accuracy = sum(pred_cla_neg.round()==label_neg).float()/len(pred_cla_neg)
    print("Accuracy for classifier on neg class: ", accuracy)

    print()

    pred_adv = gan_model(val_set[0])[1]

    accuracy = sum(pred_adv.round()==val_set[2]).float()/len(pred_adv)
    print("Accuracy for adversary: ", accuracy)

    mask_pos = val_set[2] == 1
    pred_adv_pos = pred_adv[mask_pos]
    label_pos = val_set[2][mask_pos]
    accuracy = sum(pred_adv_pos.round()==label_pos).float()/len(pred_adv_pos)
    print("Accuracy for adversary on male: ", accuracy)


    mask_neg = val_set[2] == 0
    pred_adv_neg = pred_adv[mask_neg]
    label_neg = val_set[2][mask_neg]
    accuracy = sum(pred_adv_neg.round()==label_neg).float()/len(pred_adv_neg)
    print("Accuracy for adversary on female: ", accuracy)
    
def compute_fair_metrics(gan_model):
    gan_model.eval()
    gan_model.cpu()
    result = compute_statistical_parity(gan_model, val_set)
    print("Statistical Parity:", result)

    result = equal_opportunity(gan_model, val_set, positive_target=1)
    print("Equal Opportunity:", result)

    result = equal_odds(gan_model, val_set)
    print("Equal Odds:", result)

    
    

def eval_epoch(root_dir, chans, n_output, experiments, epoch, val_set, dict_labels, overwrite=False):
    paths = []
    for exp in experiments:
        save_dir = os.path.join(root_dir, exp, "eval-epoch" + str(epoch))
        
        if overwrite or not os.path.exists(save_dir):
            if 'weight_loss' in exp:
                    logits = True
            else: logits = False

            gan_model = FairGAN(chans, n_output, 1, logits) 
            if not epoch == 0:
                gan_model.load_state_dict(torch.load(os.path.join(root_dir, exp,'model-'+str(epoch)+'.pt')))
            #gan_model.load_state_dict(torch.load(os.path.join(root_dir, exp,'model-'+str(epoch)+'.pt')))

            eval_model(gan_model, val_set, dict_labels, save_dir, batch_size=100, logits=gan_model.logits)
        
        paths.append(os.path.join(root_dir, exp, "eval-epoch" + str(epoch)))
        
    return paths


def get_results_epochs(root_dir, chans, n_output, experiments, val_set, dict_labels, max_epoch=80, step=10, overwrite=False, weights=None):
    epochs = list(range(0, max_epoch, step))
    results_acc = []
    results_fair = []
    for epoch in epochs:
        paths = eval_epoch(root_dir, chans, n_output, experiments, epoch, val_set, dict_labels, overwrite)
        results_acc.append(compute_average_acc(paths, weights))
        results_fair.append(compute_average_fairness(paths, weights))
    return results_acc, results_fair
    
    
def plot_accuracy_result(results_acc, epochs, experiments):
    plot(results_acc, 'Acc Pos', epochs, experiments)
    plot(results_acc, 'Acc Neg', epochs, experiments)
    plot(results_acc, 'Acc', epochs, experiments)
    
def plot_results(results, column, epochs, experiments, labels, y_label):
    data_all = []
    
    for exp in experiments:
        data = []
        for result in results:
            exp_name = exp.replace('/','') #exp.split('/')[0]
            mask = result['Model'] == exp_name
            data.append(result[mask][column].values[0])
        data_all.append(data)
    
    plt.rcParams["figure.figsize"] = (10,6)

    for index, data in enumerate(data_all):
        plt.plot(epochs, data, label = labels[index])
        
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    
def plot_results_multiple_runs(experiments, column, epochs, labels, y_label):
    data_all_mean = []
    data_all_std = []
    for exp in experiments:
        data_mean = []
        data_std = []
        for result in exp:
            data_mean.append(np.mean(result[column]))
            data_std.append(np.std(result[column]))
        data_all_mean.append(data_mean)
        data_all_std.append(data_std)
    
    plt.rcParams["figure.figsize"] = (10,6)
    
    colors = ["#008000", "#0000FF", "#FF0000", "#F08C00", "#A52A2A"]
    for index, data in enumerate(data_all_mean):
        plt.plot(epochs, data, label = labels[index], c=colors[index])
        #plt.errorbar(epochs, data, data_all_std[index], fmt='-o', c=colors[index],  capsize=10)
        plt.fill_between(epochs, np.array(data)-np.array(data_all_std[index]), np.array(data)+np.array(data_all_std[index]),
    alpha=0.5, edgecolor=colors[index], facecolor=colors[index])
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
    
        
def read_log(content, component):
    
    regex = component + " loss: [0-9.]*, acc: [0-9.%]*"
    result = re.findall(regex, content)
    
    loss = []
    acc = []
    for r in result:
        loss_float = float(r.strip(component + " loss: ").split(',')[0])
        loss.append(loss_float)
                        
        acc_float = float(r.split(',')[1].strip(" acc: ").strip("%"))
        acc.append(acc_float)
    return np.array(loss), np.array(acc)

def plot_log_cla_adv(root_dir, experiments, component, log):
        avg_loss_train = []
        avg_loss_val = []
        avg_acc_train = []
        avg_acc_val = []
        for experiment in experiments:
            train_log_path = os.path.join(root_dir, experiment, 'train' + '_log.txt')
            val_log_path = os.path.join(root_dir, experiment, 'val' + '_log.txt')

            with open(train_log_path, 'r') as f:
                content_train = f.read()

            with open(val_log_path, 'r') as f:
                content_val = f.read()
        

            loss_train, acc_train = read_log(content_train, component)
            loss_val, acc_val = read_log(content_val, component)
            
            avg_loss_train.append(loss_train)
            avg_loss_val.append(loss_val)
            
            avg_acc_train.append(acc_train)
            avg_acc_val.append(acc_val)
        
        

        acc_train = list(np.mean(avg_acc_train, axis=0))
        acc_val = list(np.mean(avg_acc_val, axis=0))
        
        loss_val = list(np.mean(avg_loss_val, axis=0))
        loss_train = list(np.mean(avg_loss_train, axis=0))
        if log == 'acc':
            plot_train = acc_train
            plot_val = acc_val
        elif log == 'loss':
            plot_train = loss_train
            plot_val = loss_val
        
       
        epochs = list(range(0, len(plot_train), 1))
        plt.plot(epochs, plot_train, label = component + ' ' + log +' train')
        plt.plot(epochs, plot_val, label = component + ' ' + log + ' val')
        plt.xlabel("Epochs")
        plt.ylabel(log)
        plt.legend()
        plt.show()

def compute_weights_average(val_set, classes):
    weights = []
    for industry in range(classes):
        w = sum(val_set[:, industry])
        w = w/len(val_set)
        weights.append(w)
    return weights
    