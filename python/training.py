import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import time
import math
import csv
import numpy as np
import os

class PresPredTrainer:
    def __init__(self, model, train_dataset, test_dataset=None, optimizer=optim.Adam, lr=0.001):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dataloader = None
        self.lr = lr
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr)

    def train(self, batch_size=32, epochs=10):
        self.dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
        losses = []
        losses_val = []
        for current_epoch in range(epochs):
            self.dataloader.dataset.train = 1
            for current_batch, (x, p) in enumerate(tqdm(self.dataloader, desc='Epoch {}'.format(current_epoch+1))):
                x = Variable(x.type(torch.FloatTensor))
                p = Variable(p.type(torch.FloatTensor))
                x = x.cuda()
                p = p.cuda()

                o = self.model(x)
                
                loss = F.binary_cross_entropy_with_logits(o, torch.squeeze(p))
                    
                self.optimizer.zero_grad()
                loss.backward()
                loss = loss.data
                tqdm.write('Loss is {:.4f}'.format(loss))
                losses.append(loss.cpu().numpy())

                self.optimizer.step()
            self.dataloader.dataset.train = 0
            loss_val = 0
            for current_batch, (x, p) in enumerate(tqdm(self.dataloader, desc='Epoch {}'.format(current_epoch+1))):
                x = Variable(x.type(torch.FloatTensor))
                p = Variable(p.type(torch.FloatTensor))
                x = x.cuda()
                p = p.cuda()

                o = self.model(x)

                loss = F.binary_cross_entropy_with_logits(o, torch.squeeze(p))
                    
                loss_val = loss_val + loss.data
            loss_val = loss_val/len(self.dataloader)
            losses_val.append(loss_val.cpu().numpy())
            print(" => Validation loss at epoch {} is {:.4f}".format(current_epoch, loss_val))
        if epochs>0:
            time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
            if not os.path.exists('save/'):
                os.makedirs('save')
            
            torch.save(self.model, os.path.join('save', 'model_' + time_string + '.pt'))
            with open('loss.txt', 'a') as lf:
                writer = csv.writer(lf)
                for l in losses:
                    writer.writerow([np.round(l*10000)/10000])
            with open('loss_val.txt', 'w') as lf:
                writer = csv.writer(lf)
                for l in losses_val:
                    writer.writerow([np.round(l*10000)/10000])

    def test(self, batch_size=32):
        self.dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
        self.dataloader.dataset.train = 1
        
        mse_t_pres_val = 0
        # All sources
        pres_acc = float(0)
        n_ex = float(0)
        tp = float(0)
        tn = float(0)
        fp = float(0)
        fn = float(0)
        n_p = float(0)
        n_n = float(0)
        # Source specific
        pres_acc_s = float(0)
        n_ex_s = float(0)
        tp_s = float(0)
        tn_s = float(0)
        fp_s = float(0)
        fn_s = float(0)
        n_p_s = float(0)
        n_n_s = float(0)
        
        for current_batch, (x, p) in enumerate(tqdm(self.dataloader, desc='Test')):
            x = Variable(x.type(torch.FloatTensor))
            p = p.type(torch.FloatTensor)
            x = x.cuda()

            o = self.model(x)
            o = F.sigmoid(o)
            
            pres_pred = o.round().cpu().data
            
            pres_target = torch.squeeze(p)
            
            t_pres_pred = torch.mean(pres_pred, dim=0)
            t_pres_target = torch.mean(pres_target, dim=0)
            mse_t_pres = (t_pres_pred-t_pres_target)**2
            mse_t_pres_val = mse_t_pres_val + mse_t_pres
            
            # All sources
            tp = tp + torch.sum((pres_pred==1) & (pres_target==1))
            tn = tn + torch.sum((pres_pred==0) & (pres_target==0))
            fp = fp + torch.sum((pres_pred==1) & (pres_target==0))
            fn = fn + torch.sum((pres_pred==0) & (pres_target==1))
            pres_acc = pres_acc + torch.sum(pres_pred == pres_target)
            n_p = n_p + torch.sum(pres_target==1)
            n_n = n_n + torch.sum(pres_target==0)
            n_ex = n_ex + torch.numel(pres_target)
            
            # Source specific
            tp_s = tp_s + torch.sum((pres_pred==1) & (pres_target==1), dim=0).type(torch.FloatTensor)
            tn_s = tn_s + torch.sum((pres_pred==0) & (pres_target==0), dim=0).type(torch.FloatTensor)
            fp_s = fp_s + torch.sum((pres_pred==1) & (pres_target==0), dim=0).type(torch.FloatTensor)
            fn_s = fn_s + torch.sum((pres_pred==0) & (pres_target==1), dim=0).type(torch.FloatTensor)
            pres_acc_s = pres_acc_s + torch.sum(pres_pred == pres_target, dim=0).type(torch.FloatTensor)
            n_p_s = n_p_s + torch.sum(pres_target==1, dim=0).type(torch.FloatTensor)
            n_n_s = n_n_s + torch.sum(pres_target==0, dim=0).type(torch.FloatTensor)
            n_ex_s = n_ex_s + pres_target.size(0)
            
            
        mse_t_pres_val = mse_t_pres_val/len(self.dataloader)
        print(" => Traffic estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(mse_t_pres_val[0], math.sqrt(mse_t_pres_val[0])))
        print(" => Voices estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(mse_t_pres_val[1], math.sqrt(mse_t_pres_val[1])))
        print(" => Birds estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(mse_t_pres_val[2], math.sqrt(mse_t_pres_val[2])))
        print(" => All sources estimated PTP MSE is {:.4f} (RMSE is {:.4f})".format(torch.mean(mse_t_pres_val), math.sqrt(torch.mean(mse_t_pres_val))))
        # All sources
        pres_acc = pres_acc/n_ex
        print(tp)
        tp = tp/n_p
        tn = tn/n_n
        fp = fp/n_n
        fn = fn/n_p
        print(" => All sources presence accuracy is {:.2f}%".format(100*pres_acc))
        print(" => All sources tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(100*tp, 100*tn, 100*fp, 100*fn))
        # Source specific
        pres_acc_s = pres_acc_s/n_ex_s
        tp_s = tp_s/n_p_s
        tn_s = tn_s/n_n_s
        fp_s = fp_s/n_n_s
        fn_s = fn_s/n_p_s
        print(" => Traffic presence accuracy is {:.2f}%".format(100*pres_acc_s[0]))
        print(" => Traffic tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(100*tp_s[0], 100*tn_s[0], 100*fp_s[0], 100*fn_s[0]))
        print(" => Voices presence accuracy is {:.2f}%".format(100*pres_acc_s[1]))
        print(" => Voices tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(100*tp_s[1], 100*tn_s[1], 100*fp_s[1], 100*fn_s[1]))
        print(" => Birds presence accuracy is {:.2f}%".format(100*pres_acc_s[2]))
        print(" => Birds tp: {:.2f}%, tn: {:.2f}%, fp: {:.2f}%, fn: {:.2f}%".format(100*tp_s[2], 100*tn_s[2], 100*fp_s[2], 100*fn_s[2]))


    def test_simple(self, dataset, dataset_name='test', batch_size=43):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False)
        t_pres_preds = []
        for current_batch, x in enumerate(tqdm(self.dataloader, desc='Test')):
            x = Variable(x.type(torch.FloatTensor))
            x = x.cuda()
            
            o = self.model(x)
            o = F.sigmoid(o)
            
            pres_pred = o.round().cpu().data
            t_pres_pred = torch.mean(pres_pred, dim=0)
            t_pres_preds.append(t_pres_pred.cpu().numpy())
        with open(dataset_name+'_pred.txt', 'w') as pf:
            writer = csv.writer(pf)
            for tp in t_pres_preds:
                writer.writerow([np.round(t*10000)/10000 for t in tp])
            
            
            
