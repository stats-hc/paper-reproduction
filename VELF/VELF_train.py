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



data = pd.read_pickle('data/ml-1m.pkl')

feat_col = ['userId', 'gender', 'age', 'occupation', 'movieId', 'year']
label_col = 'rating'

train, test = train_test_split(data, test_size=0.3, random_state=20220316)
print('train：', train.shape)
print('test:', test.shape)

user_id_col = 'userId'
user_attr_col = ['gender', 'age', 'occupation']
item_id_col = 'movieId'
item_attr_col = ['year']
label_col = 'rating'

train_dataset = TensorDataset(torch.LongTensor(train[user_id_col].values),
                              torch.LongTensor(train[user_attr_col].values),
                              torch.LongTensor(train[item_id_col].values),
                              torch.LongTensor(train[item_attr_col].values),
                              torch.FloatTensor(train[label_col].values),)
test_dataset = TensorDataset(torch.LongTensor(test[user_id_col].values),
                              torch.LongTensor(test[user_attr_col].values),
                              torch.LongTensor(test[item_id_col].values),
                              torch.LongTensor(test[item_attr_col].values),
                              torch.FloatTensor(test[label_col].values),)

train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

def print_info(info):
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = '{} : {}'.format(t0, info)
    print(info)
    
    
def KLD_Gaussian(mu_q, sigma_q, mu_p, sigma_p):
    return torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2)


class VELBase(nn.Module):
    def __init__(self, feature_nuniques_list, emb_size=8):
        super().__init__()
        '''
        feature_nuniques_list: a list of  features' vocabulary size
        for instance, feature_nuniques_list = [2000, [2, 8, 10], 1000, [7, 9]]
        '''
        user_id_size, user_attr_size, item_id_size, item_attr_size = feature_nuniques_list
        self.user_id_emb = nn.Embedding(user_id_size, emb_size) 
        self.user_id_dnn = nn.Sequential(nn.Linear(emb_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(256, emb_size*2), nn.Sigmoid())
        
        self.user_attr_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size) 
                                            for voc_size in user_attr_size])
        self.user_attr_dim = emb_size * len(user_attr_size)
        self.user_attr_dnn = nn.Sequential(nn.Linear(self.user_attr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                           nn.Linear(256, emb_size*2), nn.Sigmoid())
        
        self.item_id_emb = nn.Embedding(item_id_size, emb_size)
        self.item_id_dnn = nn.Sequential(nn.Linear(emb_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(256, emb_size*2), nn.Sigmoid())
        
        self.item_attr_emb = nn.ModuleList([nn.Embedding(voc_size, emb_size) 
                                          for voc_size in item_attr_size])
        self.item_attr_dim = emb_size * len(item_attr_size)
        self.item_attr_dnn = nn.Sequential(nn.Linear(self.item_attr_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                                           nn.Linear(256, emb_size*2), nn.Sigmoid())
        self.feature_dims = emb_size * 2 + self.user_attr_dim + self.item_attr_dim
        
    def forward(self, user_id, user_attr, item_id, item_attr):
        # variational embedding for user id
        user_id_emb_res = self.user_id_emb(user_id)
        user_id_dnn_res = self.user_id_dnn(user_id_emb_res)
        user_id_mu, user_id_sigma = user_id_dnn_res.chunk(2, dim=1)
        user_id_sigma = torch.abs(user_id_sigma)
        # Reparameterize Trick
        user_id_vemb = user_id_mu + user_id_sigma * torch.randn_like(user_id_sigma)
        
        # embedding for user attribute
        user_attr_emb_res = [emb(user_attr[:, i]) for i, emb in enumerate(self.user_attr_emb)]
        user_attr_emb_concat = torch.cat(user_attr_emb_res, dim=1)
        user_attr_dnn_res = self.user_attr_dnn(user_attr_emb_concat)
        user_attr_mu, user_attr_sigma = user_attr_dnn_res.chunk(2, dim=1)
        user_attr_sigma = torch.abs(user_attr_sigma)
        
        # variational embedding for item id
        item_id_emb_res = self.item_id_emb(item_id)
        item_id_dnn_res = self.item_id_dnn(item_id_emb_res)
        item_id_mu, item_id_sigma = item_id_dnn_res.chunk(2, dim=1)
        item_id_sigma = torch.abs(item_id_sigma)
        # Reparameterize Trick
        item_id_vemb = item_id_mu + item_id_sigma * torch.randn_like(item_id_sigma)
        
        # embedding for item attribute
        item_attr_emb_res = [emb(item_attr[:, i]) for i, emb in enumerate(self.item_attr_emb)]
        item_attr_emb_concat = torch.cat(item_attr_emb_res, dim=1)
        item_attr_dnn_res = self.item_attr_dnn(item_attr_emb_concat)
        item_attr_mu, item_attr_sigma = item_attr_dnn_res.chunk(2, dim=1)
        item_attr_sigma = torch.abs(item_attr_sigma)
        
        # concat all embeddings
        all_embs = torch.cat([user_id_vemb, item_id_vemb, user_attr_emb_concat, item_attr_emb_concat], dim=1)
        # users' KL-divergence
        user_kld = KLD_Gaussian(user_id_mu, user_id_sigma, user_attr_mu, user_attr_sigma)
        # items' KL-divergence
        item_kld = KLD_Gaussian(item_id_mu, item_id_sigma, item_attr_mu, item_attr_sigma)
        # user's prior KL-divergence
        user_prior_kld = KLD_Gaussian(user_attr_mu, user_attr_sigma, 0, 1)
        # items's prior KL-divergence
        item_prior_kld = KLD_Gaussian(item_attr_mu, item_attr_sigma, 0, 1)
        kld = user_kld + item_kld + user_prior_kld + item_prior_kld
        kld = torch.mean(torch.sum(kld, 1, keepdim=True))
        return all_embs, kld
    
class VELDeepFM(nn.Module):
    def __init__(self, feature_nuniques_list, alpha=0.01, emb_size=8, 
                 hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]):
        super().__init__()
        self.encoder = VELBase(feature_nuniques_list)
        self.alpha = alpha
        self.all_dims = [self.encoder.feature_dims] + hid_dims
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
        
    def forward(self, user_id, user_attr, item_id, item_attr):
        all_embs, kld = self.encoder(user_id, user_attr, item_id, item_attr)
        '''FM module'''
        fm_1st_part = torch.sum(all_embs, 1, keepdim=True)
        sum_square_emb = torch.sum(all_embs * all_embs, 1, keepdim=True)
        sum_emb = torch.sum(all_embs, 1, keepdim=True)
        square_sum_emb = sum_emb * sum_emb
        fm_2nd_part = (sum_square_emb - square_sum_emb) * 0.5
        '''DNN module'''
        dnn_out = all_embs
        for linear in self.dnn_linear_list:
            dnn_out = linear(dnn_out)
        out = dnn_out + fm_1st_part + fm_2nd_part
        out = self.sigmoid(out)
        kld = kld * self.alpha
        return out, kld
    
feature_nuniques_list = [data[user_id_col].nunique(),
                          [data[f].nunique() for f in user_attr_col],
                          data[item_id_col].nunique(),
                          [data[f].nunique() for f in item_attr_col]]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = VELDeepFM(feature_nuniques_list, 0.03)
model.to(device)

loss = nn.BCELoss() # Binary Cross Entropy Loss
loss = loss.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

total = sum([param.nelement() for param in model.parameters()])
trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])
print({'Total': total, 'Trainable': trainable})

def train_and_test(model, train_loader, test_loader, epochs, device):
    best_auc = 0.0
    for epoch in range(epochs):
        '''training process'''
        model.train()
        print("Current lr : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            user_id, user_attr, item_id, item_attr, label = x[0], x[1], x[2], x[3], x[4]
            user_id, user_attr = user_id.to(device), user_attr.to(device)
            item_id, item_attr = item_id.to(device), item_attr.to(device)
            label = label.float().to(device)
            pred, kld = model(user_id, user_attr, item_id, item_attr)
            pred = pred.view(-1)
            l = loss(pred, label) + kld
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
                user_id, user_attr, item_id, item_attr, label = x[0], x[1], x[2], x[3], x[4]
                user_id, user_attr = user_id.to(device), user_attr.to(device)
                item_id, item_attr = item_id.to(device), item_attr.to(device)
                label = label.float().to(device)
                pred, kld = model(user_id, user_attr, item_id, item_attr)
                pred = pred.view(-1).data.cpu().numpy().tolist()
                test_preds.extend(pred)
                test_labels.extend(label.cpu().numpy().tolist())
        cur_auc = roc_auc_score(test_labels, test_preds)
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), 'deep_fm_ml_1m.pth')   
        info = 'Current AUC: {:.6f}, Best AUC: {:.6f}\n'.format(cur_auc, best_auc)
        print_info(info)
        
train_and_test(model, train_loader, test_loader, 50, device)