import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from scipy import sparse
import random
import timeit

__all__=['DPSkipNet']
class DPSkipNet(nn.Module): 
    def __init__(self, video_emb, graph_dir, embed_dim,
                    train_interaction, validation_interaction, test_interaction, 
                    total_u_id_list, total_i_id_list, 
                    batch_size, num_workers=os.cpu_count() - 1):
        super().__init__()

        self.num_users = len(total_u_id_list)
        self.num_items = len(total_i_id_list)
        
        print('Making Graphs..', end=' ')
        start = timeit.default_timer()
        pos_high, pos_low, z_neg, pos_high_dict, pos_low_dict, neg_dict = self.make_graphs(graph_dir, train_interaction, total_u_id_list, total_i_id_list, self.num_users, self.num_items )
        stop = timeit.default_timer()
        print(f'Completed! (in {round((stop - start), 2)}s)')
        
        self.train_loader, self.val_loader, self.test_loader = self.make_loaders(train_interaction, validation_interaction, test_interaction, total_i_id_list, 
                                                                pos_high_dict, pos_low_dict, neg_dict,
                                                                  batch_size, num_workers)
        
        self.register_buffer('user_item_matrix_high', pos_high)
        self.register_buffer('user_item_matrix_low', pos_low)

        # initializing item embedding
        self.embedding = torch.tensor(video_emb)
        self.embed_dim = self.embedding.shape[1]
        self.video_proj = nn.Linear(embed_dim, embed_dim)
        # user-item interaction matrices

        # weights for GCN layers
        self.gcn_w1_high = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.gcn_w2_high = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.gcn_w1_low = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.gcn_w2_low = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        
        # initialize weights for high and low similarity
        # self.register_buffer('alpha_high', torch.tensor(0.5))
        # self.register_buffer('alpha_low', torch.tensor(0.5))
        self.alpha_high = nn.Parameter(torch.tensor(0.5))
        self.alpha_low = nn.Parameter(torch.tensor(0.5))

        # MLP
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
               
        self.init_weights()
        self.loss_function = nn.BCEWithLogitsLoss()
    def init_weights(self):
        for w in [self.gcn_w1_high, self.gcn_w2_high, self.gcn_w1_low, self.gcn_w2_low]:
            nn.init.xavier_uniform_(w)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch, train_step=False):
        if train_step==True:
            u = torch.tensor([item for item in batch['user_id'] for _ in range(4)])
            v = torch.stack((batch['triple'][0], batch['triple'][1], batch['triple'][2], batch['triple'][3]), dim=1).flatten()
        else:
            u = batch['user_id']
            v = batch['item_id']

        
        self.embedding = self.embedding.to(self.gcn_w1_high.device)
        
        # High similarity GCN
        video_emb_high = F.linear(self.embedding, self.gcn_w1_high)
        user_emb_high = F.relu(torch.sparse.mm(self.user_item_matrix_high, video_emb_high))
        video_emb_high = F.relu(torch.sparse.mm(self.user_item_matrix_low.t(), user_emb_high))
        
        # Low similarity GCN
        video_emb_low = F.linear(self.embedding, self.gcn_w1_low)
        user_emb_low = F.relu(torch.sparse.mm(self.user_item_matrix_high, video_emb_low))
        video_emb_low = F.relu(torch.sparse.mm(self.user_item_matrix_low.t(), user_emb_low))
        

        # Normalize weights
        weights = F.softmax(torch.stack([self.alpha_high, self.alpha_low]), dim=0)
        
        # Combine high and low similarity embeddings
        user_emb = weights[0] * user_emb_high + weights[1] * user_emb_low
        video_emb = weights[0] * video_emb_high + weights[1] * video_emb_low

        # Select user and video embeddings
        user_batch_emb = user_emb[u]  # Shape: [1024, 128]
        video_batch_emb = video_emb[v]  # Shape: [1024, 40, 128]

        pred = self.predictor(torch.cat([user_batch_emb, video_batch_emb], dim=-1)).squeeze(-1)

        if train_step==True:
            original_lists = pred.view(pred.shape[0]//4, 4).unbind(dim=1)
            pos_high_pred = original_lists[0]; pos_low_pred = original_lists[1]; neg_pred = original_lists[2]; unseen_pred = original_lists[3];
            # pos_high-pos_neg bpr loss
            mask = (batch['triple'][0] != -1) &  (batch['triple'][1] != -1)  
            bpr_loss = -torch.log(torch.sigmoid(pos_high_pred-pos_low_pred)+1e-8)[mask].mean()
            mask = (batch['triple'][0] != -1) &  (batch['triple'][2] != -1) 
            bpr_loss += -torch.log(torch.sigmoid(pos_high_pred-neg_pred)+1e-8)[mask].mean()
            bpr_loss /=2
            # print()
            condition_tensor = batch['signal'].to(bpr_loss.device) 
            pred =  torch.where(condition_tensor == 0, pos_high_pred, 
                         torch.where(condition_tensor == 1, pos_low_pred, neg_pred))
            self.bpr_loss = bpr_loss
            return pred
        return pred
    def calculate_loss(self, batch):
        target = batch['label']
        output = self.forward(batch,True)
        return self.bpr_loss+self.loss_function(output, target.float().to(output.device))
    
    def make_graphs(self, graph_dir, train_interaction, total_u_id_list, total_i_id_list, u_num, i_num):
        pos_high_norm_path = f'{graph_dir}/pos_high_norm.npy'
        pos_low_norm_path = f'{graph_dir}/pos_low_norm.npy'
        neg_norm_path = f'{graph_dir}/neg_norm_iter.npy'
        
        if not (os.path.exists(pos_high_norm_path) and os.path.exists(pos_low_norm_path) and os.path.exists(neg_norm_path)):
            make_train_pos_neg_normed(
                train_interaction, total_u_id_list, total_i_id_list, u_num, i_num,
                pos_high_output_path=pos_high_norm_path, pos_low_output_path=pos_low_norm_path, neg_output_path=neg_norm_path
            )
        pos_a_normed_high = sparse.csr_matrix(np.load(pos_high_norm_path, allow_pickle=True).item())
        pos_a_normed_low = sparse.csr_matrix(np.load(pos_low_norm_path, allow_pickle=True).item())
        neg_a_normed = sparse.csr_matrix(np.load(neg_norm_path, allow_pickle=True).item())
        
        z_pos_high = convert_sp_mat_to_sp_tensor(pos_a_normed_high)
        z_pos_low = convert_sp_mat_to_sp_tensor(pos_a_normed_low)
        z_neg = convert_sp_mat_to_sp_tensor(neg_a_normed)
        pos_high_dict = csr_to_user_item_dict(pos_a_normed_high)
        pos_low_dict = csr_to_user_item_dict(pos_a_normed_low)
        neg_dict = csr_to_user_item_dict(neg_a_normed)
        
        return z_pos_high, z_pos_low, z_neg, pos_high_dict, pos_low_dict, neg_dict

    def make_loaders(self, train_interaction, validation_interaction, test_interaction, total_i_id_list, 
                     pos_high_dict, pos_low_dict, neg_dict,
                     batch_size, num_workers):
        dataset_train = DPSkipNet_Dataset(
            interaction=train_interaction, train_bool=True, total_i_id_list=total_i_id_list,
            pos_high_dict=pos_high_dict, pos_low_dict=pos_low_dict, neg_dict=neg_dict
        )
        dataset_val = DPSkipNet_Dataset(interaction=validation_interaction)
        dataset_test = DPSkipNet_Dataset(interaction=test_interaction)
        train_loader = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(dataset_test, num_workers=num_workers, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        return train_loader, val_loader, test_loader


class DPSkipNet_Dataset(Dataset):
    def __init__(self, interaction, train_bool=False, total_i_id_list=None, pos_high_dict=None, pos_low_dict=None, neg_dict=None):
        super(DPSkipNet_Dataset, self).__init__()
        self.interaction = interaction
        self.train_bool= train_bool
        if  self.train_bool== True:
            self.total_i_id_set = set(total_i_id_list)
            self.pos_high_dict = pos_high_dict
            self.pos_low_dict = pos_low_dict
            self.neg_dict = neg_dict
    def __len__(self):
        return np.array(self.interaction).shape[0]
    
    def __getitem__(self, index):
        inter = self.interaction.iloc[index,:]  # one interaction record
        user_id = inter['user_id']
        item_id = inter['item_id']
        playing_time = inter['playing_time_ms']
        video_time = inter['duration_ms']
        ratio = playing_time/video_time

        triple = [0, 0, 0, 0]
        signal = 0 #2 neg 0: high pos
        if self.train_bool==True:
            if playing_time<5000 and ratio<1: #neg
                signal = 2
                triple[0] = random.sample(self.pos_high_dict[user_id],1)[0] if len(self.pos_high_dict[user_id])>0 else -1
                triple[1] = random.sample( self.pos_low_dict[user_id],1)[0] if len(self.pos_low_dict[user_id])>0 else -1
                triple[2] = item_id
                triple[3] = random.sample( list(self.total_i_id_set-set(self.pos_high_dict[user_id])-set(self.pos_low_dict[user_id])-set(self.neg_dict[user_id])),1)[0]
            elif playing_time>=5000 and ratio<1: #low
                signal = 1
                triple[0] = random.sample(self.pos_high_dict[user_id],1)[0] if len(self.pos_high_dict[user_id])>0 else -1
                triple[1] = item_id
                triple[2] = random.sample(self.neg_dict[user_id],1)[0] if len(self.neg_dict[user_id])>0 else -1
                triple[3] = random.sample( list(self.total_i_id_set-set(self.pos_high_dict[user_id])-set(self.pos_low_dict[user_id])-set(self.neg_dict[user_id])),1)[0]
            elif ratio>=1: # high
                signal = 0
                triple[0] = item_id
                triple[1] = random.sample( self.pos_low_dict[user_id],1)[0] if len(self.pos_low_dict[user_id])>0 else -1
                triple[2] = random.sample(self.neg_dict[user_id],1)[0] if len(self.neg_dict[user_id])>0 else -1
                triple[3] = random.sample( list(self.total_i_id_set-set(self.pos_high_dict[user_id])-set(self.pos_low_dict[user_id])-set(self.neg_dict[user_id])),1)[0]
       
        if playing_time<5000 and ratio<1:
            negative=1
        else:
            negative=0

        if ratio>=1:
            label = 1
        else:
            label = 0
            
        return dict(
            user_id = user_id,
            item_id = item_id,
            label = label,
            triple = triple,
            signal = signal
        )
    
def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

def make_normalized_graph(user_item_matrix):
    user_num = user_item_matrix.shape[0]
    item_num = user_item_matrix.shape[1]
    adj_mat = sparse.dok_matrix((user_num + item_num, user_num + item_num), dtype=np.float32).tolil()
    user_item_matrix = user_item_matrix.tolil()
    adj_mat[:user_num, user_num:] = user_item_matrix
    adj_mat[user_num:, :user_num] = user_item_matrix.T
    adj_mat = adj_mat.todok()
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sparse.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    norm_adj = norm_adj[:user_num, user_num:]
    return norm_adj

def csr_to_user_item_dict(csr):
    user_item_dict = {}
    for user in range(csr.shape[0]):
        item_indices = csr[user].indices  # 해당 사용자가 상호작용한 아이템들의 인덱스를 가져옴
        user_item_dict[user] = item_indices.tolist()
    return user_item_dict

def make_train_pos_neg_normed(interaction, total_u_id_list, total_i_id_list, 
                              u_num, i_num,
                              pos_high_output_path, pos_low_output_path, neg_output_path):
    u_pos_high = []
    v_pos_high = []
    u_pos_low = []
    v_pos_low = []
    u_neg = []
    v_neg = []
    cnt_pos_high = 0
    cnt_pos_low = 0
    cnt_neg = 0
    for i in range(len(interaction)):
        user = int(interaction.iloc[i,:]['user_id'])
        video = interaction.iloc[i,:]['item_id']
        ratio = interaction.iloc[i,:]['playing_time_ms']/interaction.iloc[i,:]['duration_ms']
        playing_time = interaction.iloc[i,:]['playing_time_ms']
        duration_ms = interaction.iloc[i,:]['duration_ms']
        if playing_time<5000 and ratio<1:
            u_neg.append(np.where(total_u_id_list == user)[0][0])
            v_neg.append(np.where(total_i_id_list == video)[0][0])
            cnt_neg = cnt_neg+1
        elif playing_time>=5000 and ratio<1:
            u_pos_low.append(np.where(total_u_id_list == user)[0][0])
            v_pos_low.append(np.where(total_i_id_list == video)[0][0])
            cnt_pos_low = cnt_pos_low+1
        elif ratio>=1:
            u_pos_high.append(np.where(total_u_id_list == user)[0][0])
            v_pos_high.append(np.where(total_i_id_list == video)[0][0])
            cnt_pos_high = cnt_pos_high+1
    data_pos_high = np.ones(cnt_pos_high)
    data_pos_low = np.ones(cnt_pos_low)
    data_neg = np.ones(cnt_neg)
    pos_a_high = csr_matrix((data_pos_high, (u_pos_high, v_pos_high)), shape=(u_num, i_num), dtype=np.float32)
    pos_a_low = csr_matrix((data_pos_low, (u_pos_low, v_pos_low)), shape=(u_num, i_num), dtype=np.float32)
    neg_a = csr_matrix((data_neg, (u_neg, v_neg)), shape=(u_num, i_num), dtype=np.float32)
    pos_a_low=pos_a_low-pos_a_high
    pos_a_low.data[pos_a_low.data < 0] = 0
    neg_a = neg_a-pos_a_low-pos_a_high
    neg_a.data[neg_a.data < 0] = 0

    pos_a_normed_high = make_normalized_graph(pos_a_high)
    pos_a_normed_low = make_normalized_graph(pos_a_low)
    neg_a_normed = make_normalized_graph(neg_a)
    
    np.save(pos_high_output_path, pos_a_normed_high)
    np.save(pos_low_output_path, pos_a_normed_low)
    np.save(neg_output_path,neg_a_normed)