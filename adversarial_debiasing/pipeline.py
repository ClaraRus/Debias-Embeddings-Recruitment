import os
import json
import torch

from FairGAN import FairGAN
from train import Trainer
from evaluation import eval_model
from dataset import AdultDataset

class Pipeline:
    def __init__(self, config_path, run=None):
        
        config = self.read_config(config_path)
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        
        if 'n_save' in config.keys():
            self.n_save = config['n_save']
        else: self.n_save = 10
        
        self.logits = config['logits']
        self.alternate = config['alternate_batch']
        self.penalize = config['penalize']
        
        self.train_adv = config['train_adv']
        self.train_cla = config['train_cla']
        self.train_dec = config['train_dec']
        self.lam = config['lam']
        
        self.lr = config['lr']
        self.decay = config['decay']
        
        self.dataset = config['dataset']
        self.data_path = config['root_path']
        
        self.save_dir = "../models/" + self.dataset +"/FairGAN"
        if self.train_cla:
            self.save_dir = self.save_dir + "-cla"
        if self.train_adv:
            self.save_dir = self.save_dir + "-adv"
        print(self.logits)
        if self.logits:
            self.save_dir = self.save_dir + "-weight_loss"
         
        self.save_dir = self.save_dir + "-EXP-" + str(config['EXP'])
        if not run is None:
            self.save_dir = os.path.join(self.save_dir, str(run))
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        json_string = json.dumps(config)
        print(self.save_dir)
        with open(os.path.join(self.save_dir,'config.json'), 'w') as outfile:
            json.dump(json_string, outfile)
            
        
    def read_config(self, config_path):
        with open(config_path) as json_file:
            config = json.load(json_file)
        return config
    
    def read_data(self):
        if self.dataset == 'adult':
            self.dataset = AdultDataset(self.data_path)
        else:
            raise NotImplementedError("No dataset with this name!")

        
        if not 'train_val_split' in self.data_path:
             self.dataset.create_train_val_split()
        train_set, val_set, dict_labels, weights = self.dataset.load_data()
        return train_set, val_set, dict_labels, weights
    
    def start(self, verbose):
        print("Read data...")
        train_set, val_set, dict_labels, weights = self.read_data()
        
        print("Init model...")
        gan_model = FairGAN(chans=self.dataset.chans, n_output=self.dataset.n_output, n_nuisance=1, logits=self.logits)
        
        print("Start train...")
        trainer = Trainer(train_adv=self.train_adv, train_cla=self.train_cla, train_dec= self.train_dec, lam=self.lam, lr=self.lr, decay=self.decay, epochs=self.epochs, batch_size=self.batch_size, logits=self.logits, weights=weights, alternate=self.alternate, penalize=self.penalize, save_log_dir=self.save_dir, n_save=self.n_save)
    
        trainer.train(gan_model, train_set, val_set, verbose)
        
        print("Save model...")
        torch.save(gan_model.state_dict(), os.path.join(self.save_dir, "model.pt"))
        
        print("Eval model...")
        eval_model(gan_model, val_set, dict_labels, os.path.join(self.save_dir, "eval"), self.batch_size, self.logits)