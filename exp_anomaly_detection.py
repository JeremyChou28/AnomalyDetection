import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from model.Transformer import Transformer,ConvTransformer
from model.baseline import UNet
from dataloader.data_provider import *
from datetime import datetime


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader= get_dataloader(self.data_path, mode="train", batch_size=self.batch_size)
        self.val_loader = get_dataloader(self.data_path, mode="val", batch_size=self.batch_size)
        self.test_loader = get_dataloader(self.data_path, mode="test", batch_size=self.batch_size)
        print("train_loader len:", len(self.train_loader))
        print("val_loader len:", len(self.val_loader))
        print("test_loader len:", len(self.test_loader))

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        if self.model_name=="UNet":
            self.model = UNet(n_channels=1, n_classes=1)
        elif self.model_name=="Transformer":
            self.model = Transformer(enc_in=self.input_c, c_out=self.output_c, e_layers=self.e_layers, d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout)
        elif self.model_name=="AnomalyTransformer":
            self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=self.e_layers, d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout)
        else:
            self.model = ConvTransformer(enc_in=self.input_c, c_out=self.output_c, e_layers=self.e_layers, d_channel=4, d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff, dropout=self.dropout)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler_recon = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            
            for i, (input_data, _, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output = self.model(input)
                # rec_loss = self.criterion(output, input)
                pred = output.detach().cpu()
                true = input.detach().cpu()

                loss = self.criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        
        print("======================TRAIN MODE======================")

        time_now = time.time()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        path = self.model_save_path+'/{}/'.format(current_time)
        self.model_save_path = path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=7, verbose=True)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels, input_groundtruth) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                input_groundtruth = input_groundtruth.float().to(self.device)

                output= self.model(input)

                rec_loss = self.criterion(output, input_groundtruth)
                train_loss.append(rec_loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                rec_loss.backward()
                self.optimizer.step()

            self.scheduler_recon.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(self.val_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self,):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), 'checkpoint.pth')))
        self.model.eval()
        
        prediction=[]
        groundtruth=[]
        for i, (input_data, labels,input_groundtruth) in enumerate(self.test_loader):
            input = input_data.float().to(self.device)
            input_groundtruth = input_groundtruth.float().to(self.device)
            output = self.model(input)

            prediction.append(output.detach().cpu().numpy())
            groundtruth.append(input_groundtruth.detach().cpu().numpy())
        prediction=np.concatenate(prediction,axis=0)
        groundtruth=np.concatenate(groundtruth,axis=0)
        
        print("prediction: ", prediction.shape)
        print("groundtruth: ", groundtruth.shape)
        # 计算prediction和groundtruth之间的MSE和MAE
        mse = np.mean((prediction - groundtruth) ** 2)
        mae = np.mean(np.abs(prediction - groundtruth))
        print(f'MSE: {mse.item()}, MAE: {mae.item()}')
