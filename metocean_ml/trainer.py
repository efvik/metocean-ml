import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Trainer():
    """
    A general trainer for pytorch models. You can set various parameters
    in initialization trainer = Trainer(...), then use 
    logs = trainer.fit(model, train_dataset, val_dataset) to train a model.
    """
    def __init__(
            self,
            train_lossfunc:nn.Module = nn.L1Loss(),
            val_lossfunc = nn.L1Loss(),
            early_stopping_patience = 10,
            reduce_lr_patience = 5,
            minimum_lr = 1e-8,
            improvement_threshold = 1e-7,
            max_epochs = 100,
            learning_rate = 0.0001,
            batch_size = 2**10,
            verbose=True,
            ):
        self.train_lossfunc = train_lossfunc
        self.val_lossfunc = val_lossfunc
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.minimum_lr = minimum_lr
        self.improvement_threshold = improvement_threshold
        self.max_epochs = max_epochs
        self.learning_rate_init = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
    
    def fit(self,
            model:nn.Module,
            train_data:DataLoader|Dataset,
            val_data:DataLoader|Dataset):
        '''
        Arguments:
        ---------
        model : nn.Module
            Pytorch model.
        train_dataset : DataLoader or Dataset
            Training dataset.
        val_dataset DataLoader or Dataset
            Validation dataset.
        
        Returns:
        --------
        logs : dict
            Training loss and validation loss per epoch.
        '''
        train_loss_log = []
        hs_train_loss_log = []
        val_loss_log = []
        lr_log = []
        best_loss = float("inf")
        patience_counter = 0
        optimizer = torch.optim.Adam(model.parameters())
        learning_rate = self.learning_rate_init
        
        if not isinstance(train_data,DataLoader):
            train_data = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        if not isinstance(val_data,DataLoader):
            val_data = DataLoader(val_data,batch_size=self.batch_size,shuffle=False)
        
        pbar = tqdm(range(self.max_epochs),smoothing=0)
        for epoch in pbar:
            lr_log.append(learning_rate)

            # Training loop
            model.train()
            train_loss = 0.0
            hs_train_loss = 0.0
            for inputs, targets in train_data:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = self.train_lossfunc(outputs, targets)
                #hs_loss = torch.abs(outputs.mean() - targets.mean())*2
                #loss += hs_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_data.dataset)
            train_loss_log.append(train_loss)
            #print(loss,hs_loss)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_data:
                    outputs = model(inputs).squeeze()
                    loss = self.val_lossfunc(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_data.dataset)
            val_loss_log.append(val_loss)
            # Info
            pbar.set_description('Epoch {}, Training loss: {}, validation loss: {}.'.format(epoch+1,round(train_loss,5),round(val_loss,5)))

            # Check if performance increased
            if val_loss + self.improvement_threshold < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Reduce lr or stop
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print("Early stopping triggered")
                break
            if patience_counter >= self.reduce_lr_patience:
                learning_rate *= 0.1
                if learning_rate <= self.minimum_lr:
                    break
                patience_counter = 0
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                if self.verbose:
                    print(f"No improvement for {self.reduce_lr_patience} epochs. Lr reduced to {self.learning_rate}")

        return {"val_loss":val_loss_log,
                "train_loss":train_loss_log,
                "learning_rate":lr_log}

    def test(self,
             model:nn.Module,
             test_data:Dataset|DataLoader):
        
        if isinstance(test_data,Dataset):
            test_data = DataLoader(test_data,batch_size=self.batch_size,shuffle=False)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs,targets in test_data:
                outputs = model(inputs).squeeze()
                loss = self.val_lossfunc(outputs,targets)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_data.dataset)