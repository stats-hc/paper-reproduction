import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import time, json, datetime 
from tqdm import tqdm

import numpy as np 
import pandas as pd 
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


class DeepFM(nn.Module):
    def __init__(self, sparse_feat_nuniques, emb_size=8, hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]):
        '''
        # sparse_feat_uuniques: a list containing voc_size of each sparse feature
        '''
        super().__init__()
        self.sparse_feat_size = len(sparse_feat_nuniques) 
        '''FM module'''
        # first-order representation
        self.fm_1st_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, 1) for voc_size in sparse_feat_nuniques])
        # second-order representation
        self.fm_2nd_order_sparse_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size) for voc_size in sparse_feat_nuniques])
        '''DNN module'''
        self.all_dims = [self.sparse_feat_size * emb_size] + hid_dims
        self.dnn_linear_list = nn.ModuleList()
        for i in range(1, len(self.all_dims)):
            self.dnn_linear_list.append(nn.Sequential(
                nn.Linear(self.all_dims[i-1], self.all_dims[i]),
                nn.BatchNorm1d(self.all_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout[i-1])
            ))
        self.dnn_linear_list.append(nn.Linear(hid_dims[-1], num_classes))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_sparse):
        '''FM module'''
        # first-order
        fm_1st_sparse_res = [emb(X_sparse[:, i]) for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, sparse_feat_size]
        fm_1st_part = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1]
        # second-order
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  (n=sparse_feat_size)
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        '''dnn module'''
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)
        for linear in self.dnn_linear_list:
            dnn_out = linear(dnn_out)
        out = fm_1st_part + fm_2nd_part + dnn_out
        out = self.sigmoid(out)
        return out
    
    
data = pd.read_pickle('data/ml-1m.pkl')

feat_col = ['userId', 'gender', 'age', 'occupation', 'movieId', 'year']
label_col = 'rating'

train, test = train_test_split(data, test_size=0.3, random_state=20220316)
print('train：', train.shape)
print('test:', test.shape)


train_dataset = TensorDataset(torch.LongTensor(train[feat_col].values),
                              torch.FloatTensor(train[label_col].values), )
test_dataset = TensorDataset(torch.LongTensor(test[feat_col].values),
                             torch.LongTensor(test[label_col].values),)

train_loader = DataLoader(dataset=train_dataset, batch_size=2048, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4096, shuffle=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feat_nuniques = [data[f].nunique() for f in feat_col]

model = DeepFM(feat_nuniques)
model.to(device)


loss = nn.BCELoss() # Binary Cross Entropy Loss
loss = loss.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

total = sum([param.nelement() for param in model.parameters()])
trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])
print({'Total': total, 'Trainable': trainable})


def print_info(info):
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = '{} : {}'.format(t0, info)
    print(info)
    
    
def train_and_test(model, train_loader, test_loader, epochs, device):
    best_auc = 0.0
    for epoch in range(epochs):
        '''training process'''
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            features, label = x[0], x[1]
            features, label = features.to(device), label.float().to(device)
            pred = model(features).view(-1)
            l = loss(pred, label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            train_loss_sum += l.cpu().item()
            if (idx+1) % 50 == 0 or (idx + 1) == len(train_loader):
                
                info = 'Epoch: {:04d} | Step: {:04d} / {} | Loss: {:.4f} | Time: {:.4f}'.format(
                          epoch+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time() - start_time)
                print_info(info)     
            
        scheduler.step()
        
        '''inference process'''
        model.eval()
        with torch.no_grad():
            test_labels, test_preds = [], []
            for idx, x in tqdm(enumerate(test_loader)):
                features, label = x[0], x[1]
                features = features.to(device)
                pred = model(features).view(-1).data.cpu().numpy().tolist()
                test_preds.extend(pred)
                test_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(test_labels, test_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), 'deep_fm_ml_1m.pth')   
        info = 'Current AUC: {:.6f}, Best AUC: {:.6f}\n'.format(cur_auc, best_auc)
        print_info(info)
        
train_and_test(model, train_loader, test_loader, 30, device)