import os
import numpy as np 
import pickle
import torch 
from torch.utils.data import DataLoader, Dataset


class ADdataset(Dataset):
    def __init__(self, data_path, mode="train", test_file=None):
        self.mode = mode
        self.data_path = data_path
        self.test_file = test_file

        if self.mode=="train" or self.mode=="val":
            with open(data_path+"/train_data.pkl", "rb") as f:
                self.x, self.label, self.x_tilde = pickle.load(f)
                val_size=int(len(self.x)*0.2)
                self.train_x=self.x[:-val_size]
                self.train_label=self.label[:-val_size]
                self.train_x_tilde=self.x_tilde[:-val_size]
                self.val_x=self.x[-val_size:]
                self.val_label=self.label[-val_size:]
                self.val_x_tilde=self.x_tilde[-val_size:]
        elif self.mode=="test":
            if self.test_file=="hardiron":
                with open(data_path+"/hardiron_test.pkl", "rb") as f:
                    self.test_x, self.test_label, self.test_x_tilde = pickle.load(f)
            elif self.test_file=="softiron":
                with open(data_path+"/softiron_test.pkl", "rb") as f:
                    self.test_x, self.test_label, self.test_x_tilde = pickle.load(f)
            else:
                with open(data_path+"/test_data.pkl", "rb") as f:
                    self.test_x, self.test_label, self.test_x_tilde = pickle.load(f)
            
        
    def __len__(self,):
        if self.mode=="train":
            return len(self.train_x)
        elif self.mode=="val":
            return len(self.val_x)
        elif self.mode=="test":
            return len(self.test_x)
        else:
            raise ValueError("mode should be train, val or test")
        
    def __getitem__(self, index):
        if self.mode=="train":
            x=self.train_x[index]
            label=self.train_label[index]
            x_tilde=self.train_x_tilde[index]
        elif self.mode=="val":
            x=self.val_x[index]
            label=self.val_label[index]
            x_tilde=self.val_x_tilde[index]
        elif self.mode=="test":
            x=self.test_x[index]
            label=self.test_label[index]
            x_tilde=self.test_x_tilde[index]
        else:
            raise ValueError("mode should be train, val or test")
        
        return x, label, x_tilde


def get_dataloader(data_path, mode="train", test_file=None, batch_size=32):
    dataset = ADdataset(data_path, mode=mode, test_file=test_file)
    
    shuffle = True if mode=="train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
