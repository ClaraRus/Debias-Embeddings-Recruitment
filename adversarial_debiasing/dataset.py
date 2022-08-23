import numpy as np
import torch 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import os
    
def save_split(data_set, save_path):
    lst = [['emb', data_set[0]], ['label', data_set[1]],
           ['gender', data_set[2]]]

    df = pd.DataFrame(lst, columns =['Data', 'Value'])
    df.to_pickle(save_path)
    
class Dataset():
    def __init__(self, data_path):
        self.name = 'dataset'
        self.chans = 1
        self.n_output = 1
        self.n_nuisance = 1
        self.data_path = data_path
    
    def create_train_val_split(self):
        train_path = os.path.join(self.data_path, 'train_val_split')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_set, val_set, dict_labels = self.prepare_data()
        weights = self.compute_loss_weights(train_set[1], len(dict_labels.keys()))
        
        dict_labels_df = pd.DataFrame(list(dict_labels))
        save_path = os.path.join(train_path, 'dict_labels.pkl')
        dict_labels_df.to_pickle(save_path)
        
        weights_df = pd.DataFrame(weights)
        save_path = os.path.join(train_path, 'weights.pkl')
        weights_df.to_pickle(save_path)
        
        save_split(train_set, save_path=os.path.join(train_path, 'train_set.pkl'))
        save_split(val_set, save_path=os.path.join(train_path, 'val_set.pkl'))
        
    def compute_loss_weights(self, Y_label, n_classes):
        # for positive class
        weights = []
        for class_index in range(n_classes):
            mask_pos_class = Y_label[:, class_index] == 1
            mask_neg_class = Y_label[:, class_index] == 0
            neg = sum(mask_neg_class)
            pos = sum(mask_pos_class)
            weights.append(neg/pos)

        weights = torch.tensor(weights)
        return weights
    
    def load_data(self):
        def convert(lst):
            res_dct = {lst[i][0]: i for i in range(0, len(lst))}
            return res_dct
        
        root_path = os.path.join(self.data_path, 'train_val_split')
        dict_labels_pd = pd.read_pickle(os.path.join(root_path, 'dict_labels.pkl')) 
        train_set_df = pd.read_pickle(os.path.join(root_path, 'train_set.pkl')) 
        val_set_df = pd.read_pickle(os.path.join(root_path, 'val_set.pkl')) 
        weights_df = pd.read_pickle(os.path.join(root_path, 'weights.pkl')) 

        dict_labels = convert(list(dict_labels_pd.values))
        weights = torch.FloatTensor([w[0] for w in list(weights_df.values)])
        train_set = [train_set_df['Value'][0], train_set_df['Value'][1], train_set_df['Value'][2]]
        val_set = [val_set_df['Value'][0], val_set_df['Value'][1], val_set_df['Value'][2]]

        return train_set, val_set, dict_labels, weights
    
    
    def train_val_split(self, X, Y, gender, split=0.30):
       
        print("Data:", X.shape, "Gender:", gender.shape, "Label:", Y.shape)

        labels = list(zip(gender, Y))
        X_index = np.arange(0, X.shape[0])

        X_train, X_val, Y_train, Y_val = train_test_split(X_index, labels, test_size=split)

        Y_val_gender, Y_val_label = list(zip(*Y_val))
        Y_train_gender, Y_train_label = list(zip(*Y_train))

        X_train = X[X_train]
        X_val = X[X_val]

        X_train_tensor = torch.tensor(X_train).float()
        gender_train_tensor = torch.tensor(Y_train_gender).float()
        y_train_tensor = torch.tensor(Y_train_label).float()

        X_val_tensor = torch.tensor(X_val).float()
        gender_val_tensor = torch.tensor(Y_val_gender).float()
        y_val_tensor = torch.tensor(Y_val_label).float()

        train_set = [X_train_tensor, y_train_tensor, gender_train_tensor]
        val_set = [X_val_tensor, y_val_tensor, gender_val_tensor]

        print("Train Data:", train_set[0].shape, "Val Data:", val_set[0].shape)
        
        return train_set, val_set
    
    def prepare_data(self):
        pass
        
class AdultDataset(Dataset):
    def __init__(self, data_path):
        super(AdultDataset, self).__init__(data_path)  
        
        self.name = 'adult_dataset'
        self.chans = 9
        self.n_output = 1
        self.n_nuisance = 1   
        
    def encode_data(self, X, data):
        encoders = {}

        X_num = X.copy()

        for col in X_num.columns.tolist():
            if X_num[col].dtype == object:
                encoders[col] = preprocessing.LabelEncoder().fit(X_num[col])
                X_num[col] = encoders[col].transform(X_num[col])

        X_num = X_num.drop(['sex'], axis=1)

        y = data['income'].copy()
        y = y.replace(" <=50K", 0)
        y = y.replace(" >50K", 1)
        gender = encoders['sex'].transform(X['sex'])
        
        return X_num, y, gender
        
    def prepare_data(self):
        data = pd.read_csv(os.path.join(self.data_path, 'adult.data'))
        data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship','race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        X = data.drop(['income', 'fnlwgt', 'capital-loss', 'capital-gain', 'race'], axis=1)
        
        X, Y, gender= self.encode_data(X, data)
        
        X = X.values
        Y = Y.values[:, None]
        gender = gender[:, None]
        
        train_set, val_set = self.train_val_split(X, Y, gender)
        
        dict_labels = {">50K": 1}
        return train_set, val_set, dict_labels        