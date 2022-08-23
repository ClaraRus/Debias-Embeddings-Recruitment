import numpy as np
import tqdm
import torch 
import os
import pandas as pd
import torch.optim as optim    
    
class Trainer:
    def __init__(self, train_cla, train_adv, train_dec, lam, lr=1e-5, decay=1e-4, epochs=500, batch_size=40, logits=False, weights=None, alternate=False, penalize=False, fair_loss=False, save_log_dir=None, n_save=10, n_log=1):
        self.train_cla = train_cla
        self.train_adv = train_adv
        self.train_dec = train_dec
        self.lam = lam
        
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.logits = logits
        self.weights = weights
        
        self.alternate = alternate
        self.penalize=penalize
        self.fair_loss = fair_loss
        
        self.save_log_dir = save_log_dir
        self.n_save = n_save
        self.n_log = n_log
        
        self.loss_hyperparam = [0., 0., 0.]
        if self.train_cla:
            self.loss_hyperparam[0] = 1 * self.lam[0]
        else: self.lam[0] = 0
            
        if self.train_adv:
            self.loss_hyperparam[1] = -1 * self.lam[1]
        else: self.lam[1] = 0
            
        if self.train_dec:
            self.loss_hyperparam[2] = 1 * self.lam[2]
        else: self.lam[2] = 0
    
        if logits:
            self.weights = self.weghts.cuda()
            self.criterion_cla = torch.nn.BCEWithLogitsLoss(pos_weight=self.weights)
        else: 
            self.criterion_cla = torch.nn.BCELoss()

        if fair_loss == 'parity':
            self.criterion_adv = torch.nn.L1Loss()
        else:
            self.criterion_adv = torch.nn.BCELoss()

        self.criterion_dec = torch.nn.MSELoss()


    def update_loss_hyperparam(self, iter, acc_adv):
        if self.alternate:
            if iter % 2 == 0:
                self.loss_hyperparam[0] = 0
                self.loss_hyperparam[1] = 1 * self.lam[1]
                self.loss_hyperparam[2] = 0
            else: 
                self.loss_hyperparam[0] = 1 * self.lam[0]
                self.loss_hyperparam[1] = -1 * self.lam[1]
                self.loss_hyperparam[2] = 1 * self.lam[2]

        if self.penalize:
            if acc_adv < self.penalize[0]:
                self.loss_hyperparam[0] = 0
                self.loss_hyperparam[1] = 1 * self.lam[1]
                self.loss_hyperparam[2] = 0
            elif acc_adv > self.penalize[1]:
                self.loss_hyperparam[0] = 1 * self.lam[0]
                self.loss_hyperparam[1] = 0
                self.loss_hyperparam[2] = 1 * self.lam[2]
    
    def compute_adv_loss(self, pred_adv, s):
        if self.fair_loss == 'parity':
            mask_0 = s == 0
            mask_1 = s == 1 

            loss_0 = self.criterion_adv(pred_adv[mask_0], s[mask_0])
            loss_1 = self.criterion_adv(pred_adv[mask_1], s[mask_1])

            loss = 1 - (loss_0 + loss_1)

            return loss
        else: 
            return self.criterion_adv(pred_adv, s)

    def print_train_params(self):
        print("Train Classifier: ", self.train_cla)
        print("Train Adversary: ", self.train_adv)
        print("Train Decoder: ", self.train_dec)
        print("Params components: ", self.lam)
        print("Train with weighted loss: ", self.logits)
        print("Alternate Batch: ", self.alternate)
        print("Penalize: ", self.penalize)
        print("Fair Loss: ", self.fair_loss)

    def save_logs(self, epoch, train_log, val_log, verbose=True):
        if epoch == 1:
             try:
                os.remove(os.path.join(self.save_log_dir, 'train_log.txt'))
                os.remove(os.path.join(self.save_log_dir, 'val_log.txt'))
             except OSError:
                   pass
             with open(os.path.join(self.save_log_dir, 'train_log.txt'), 'w') as f:
                    f.write("[Epoch : %d] - Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                          % (epoch, train_log[0], train_log[1], 100*train_log[2], train_log[3], 100*train_log[4], train_log[5]))
             with open(os.path.join(self.save_log_dir, 'val_log.txt'), 'w') as f:
                    f.write("[Epoch : %d] - Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                          % (epoch, val_log[0], val_log[1], 100*np.mean(val_log[2]), val_log[3], 100*val_log[4], val_log[5]))   
        elif epoch % self.n_log == 0:
            if self.save_log_dir:
                with open(os.path.join(self.save_log_dir, 'train_log.txt'), 'a') as f:
                    f.write("[Epoch : %d] - Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                          % (epoch, train_log[0], train_log[1], 100*train_log[2], train_log[3], 100*train_log[4], train_log[5]))
                with open(os.path.join(self.save_log_dir, 'val_log.txt'), 'a') as f:
                    f.write("[Epoch : %d] - Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                          % (epoch, val_log[0], val_log[1], 100*np.mean(val_log[2]), val_log[3], 100*val_log[4], val_log[5]))
            if verbose:
                print("Train - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                              % (train_log[0], train_log[1], 100*train_log[2], train_log[3], 100*train_log[4], train_log[5]))
                print("Validation - [Loss: %f] - [CLA loss: %f, acc: %.2f%%] - [ADV loss: %f, acc: %.2f%%] - [DEC loss: %f]"
                              % (val_log[0], val_log[1], 100*np.mean(val_log[2]), val_log[3], 100*val_log[4], val_log[5]))

    def train(self, gan_model, train_set, val_set, verbose):
        
        optimizer = optim.Adam(gan_model.parameters(), lr=self.lr)

        if verbose:
            self.print_train_params()

        x_train, y_train, s_train = train_set
        x_test, y_test, s_test = val_set

        train_index = np.arange(y_train.shape[0])
        train_batches = [(i * self.batch_size, min(y_train.shape[0], (i + 1) * self.batch_size))
                             for i in range((y_train.shape[0] + self.batch_size - 1) // self.batch_size)]

        # Early stopping variables
        es_wait = 0
        es_best = np.Inf
        es_best_weights = None

        if torch.cuda.is_available():
            gan_model.cuda()
        for epoch in range(1, self.epochs + 1):
            if verbose:
                print('Epoch {}/{}'.format(epoch, self.epochs))
            np.random.shuffle(train_index)
            train_log = []
            
            gan_model.train()
            for iter, (batch_start, batch_end) in tqdm.tqdm(enumerate(train_batches)):

                #alternate batch
                x_train_batch = x_train[batch_start:batch_end]
                y_train_batch = y_train[batch_start:batch_end]
                s_train_batch = s_train[batch_start:batch_end]

                if torch.cuda.is_available():
                    x_train_batch = x_train_batch.cuda()
                    y_train_batch = y_train_batch.cuda()
                    s_train_batch = s_train_batch.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = gan_model(x_train_batch)
                pred_cla = outputs[0]
                pred_adv = outputs[1]
                pred_dec = outputs[2]

                if self.logits:
                    pred_cla = torch.sigmoid(pred_cla)

                loss_cla = self.criterion_cla(pred_cla, y_train_batch)
                loss_dec = self.criterion_dec(pred_dec, x_train_batch)
                loss_adv = self.compute_adv_loss(pred_adv, s_train_batch) 

                acc_cla = sum(pred_cla.round()==y_train_batch).float()/len(pred_cla)
                acc_adv = sum(pred_adv.round()==s_train_batch).float()/len(pred_adv)

                self.update_loss_hyperparam(iter, acc_adv)

                loss = (self.loss_hyperparam[0] * loss_cla) + (self.loss_hyperparam[1] * loss_adv) + (self.loss_hyperparam[2] * loss_dec)

                loss.backward()
                optimizer.step()

                train_log.append((loss.item(), loss_cla.item(), np.mean(acc_cla.cpu().detach().numpy()), loss_adv.item(), np.mean(acc_adv.cpu().detach().numpy()), loss_dec.item()))

            train_log = np.mean(np.array(train_log), axis=0)
            val_log = self.eval_epoch(gan_model, val_set)

            self.save_logs(epoch, train_log, val_log, verbose=True)
            if epoch % self.n_save == 0 or epoch == 1:
                print("Save model...")
                torch.save(gan_model.state_dict(), os.path.join(self.save_log_dir, "model-"+str(epoch)+".pt"))
                
            
    def eval_epoch(self, gan_model, val_set):
        gan_model.eval()
        
        val_log = []
        with torch.no_grad():
            x = val_set[0]
            y = val_set[1]
            s = val_set[2]
            val_index = np.arange(x.shape[0])
            val_batches = [(i * self.batch_size, min(x.shape[0], (i + 1) * self.batch_size))
                                     for i in range((x.shape[0] + self.batch_size - 1) // self.batch_size)]

            for iter, (batch_start, batch_end) in tqdm.tqdm(enumerate(val_batches)):
                x_batch = x[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]
                s_batch = s[batch_start:batch_end]

                if torch.cuda.is_available():
                    x_train_batch = x_train_batch.cuda()
                    y_train_batch = y_train_batch.cuda()
                    s_train_batch = s_train_batch.cuda()

                pred = gan_model(x_batch)
                pred_cla = pred[0]
                pred_adv = pred[1]
                pred_dec = pred[2]

                if self.logits:
                    pred_cla = torch.sigmoid(pred_cla)

                acc_cla = sum(pred_cla.round()==y_batch).float()/len(pred_cla)
                acc_adv = sum(pred_adv.round()==s_batch).float()/len(pred_adv)

                loss_cla = self.criterion_cla(pred_cla, y_batch)
                loss_adv = self.compute_adv_loss(pred_adv, s_batch)
                loss_dec = self.criterion_dec(pred_dec, x_batch)

                loss = (self.loss_hyperparam[0] * loss_cla) + (self.loss_hyperparam[1] * loss_adv) + (self.loss_hyperparam[2] * loss_dec)
                
                val_log.append((loss.item(), loss_cla.item(), acc_cla.cpu().detach().numpy(), loss_adv.item(), acc_adv.cpu().detach().numpy(), loss_dec.item()))

            val_log = np.mean(val_log, axis=0)

        return val_log

