import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch 
import numpy as np
import os

class MyModel(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MyModel, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.layer_1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.layer_2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x):
            hidden = self.layer_1(x)
            relu = self.relu(hidden)
            output = self.layer_2(relu)
            output = self.sigmoid(output)
            return output
        
class Trainer():
    def __init__(self, input_size, hidden_size=20, lr=0.01, epochs=100):
        self.epochs = epochs
        
        self.model = MyModel(input_size=input_size, hidden_size=hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.criterion = torch.nn.BCELoss()
   

    def train(self, X_train, Y_train):
        self.losses = []
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(X_train)
            # Compute Loss
            loss = self.criterion(y_pred, Y_train)

            if epoch % 5 == 0:
                self.losses.append(loss)
                print("epoch {}\tloss : {}\t".format(epoch,loss))
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
    def plot_loss(self):
        plt.plot(self.losses)
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
    
    def plot_ROC(self, X, Y):
        preds = self.model(X)
        fpr, tpr, threshold = metrics.roc_curve(Y, preds.detach().numpy())
        roc_auc = metrics.auc(fpr, tpr)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        ax.set_aspect('equal', adjustable='box')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
    def evaluation(self, X, Y):
        self.model.eval()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, Y) 
        print('Loss:' , loss.item())
        accuracy = sum(y_pred.round() == Y)/len(Y)
        print('Accuracy: ', accuracy)