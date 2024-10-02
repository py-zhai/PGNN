
import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from PGNN import PGNN, collate
from dgl import load_graphs
import pickle
from utils import myFloder, mkdir_if_not_exist, Logger
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score
print(torch.__version__)


def str2list(v):
    v=v.split(',')
    v=[int(_.strip('[]')) for _ in v]
    return v
def str2list2(v):
    v=v.split(',')
    v=[float(_.strip('[]')) for _ in v]
    return v

def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='Alimama', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=1024, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=16, help='hidden state size')
parser.add_argument('--epoch', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=16, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=16, help='the max length of use sequence')
parser.add_argument('--k_hop', type=int, default=2, help='sub-graph size')
parser.add_argument('--gpu', default='0')
parser.add_argument("--record", default = True, help='record experimental results')
parser.add_argument("--model_record", action='store_true', default=True, help='record model')
parser.add_argument('--seq_path', type=str, default='../../all_datasets/Alimama/', help='root path of the data file')
parser.add_argument('--raw_data_path', type=str, default='../../all_datasets/Alimama/nonrawdata/', help='data file')
parser.add_argument('--data_path', type=str, default='../../all_datasets/Alimama/graph_data/', help='generated file')

parser.add_argument('--has_residual', default=True,  help='add residual')
parser.add_argument('--blocks', type=int, default=2, help='#blocks')
parser.add_argument('--block_shape', type=str2list, default=[64, 64], help='output shape of each block')
parser.add_argument('--ks', type=str2list, default=[12, 10, 8], help='the size of sampled neighborhood')
parser.add_argument('--heads', type=int, default=2, help='#heads')
parser.add_argument('--layers', type=str2list, default=[64, 64, 64], help='output shape of each layer')
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--dropout_keep_prob', type=str2list2, default=[0.8, 0.8, 0.9])

parser.add_argument('--embedding_size', type=int, default=16)
parser.add_argument('--optimizer_type', type=str, default='adam')
parser.add_argument('--random_seed', type=int, default=2024) 
parser.add_argument('--field_size', type=int, default=15, help='#fields') 
parser.add_argument('--loss_type', type=str, default='logloss')
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--top_k', type=int, default=4)

opt = parser.parse_args()
args, extras = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda:0')
print(opt)

if opt.record:
    log_file = f'../../work/log/PGNN/result'
    mkdir_if_not_exist(log_file)
    sys.stdout = Logger(log_file)
    print(f'Logging to {log_file}')
if opt.model_record:
    model_file = f'../../work/model/{opt.data}/PGNN'
    mkdir_if_not_exist(model_file)
    
data=pd.read_json(opt.raw_data_path + 'data_process_lbe.json', orient='split')#
print(data.shape)
print(data.head())

user = data['userid'].unique()
item = data['itemid'].unique()
user_num = len(user)
item_num = len(item)
print('user_num, item_num:', user_num, item_num)

train_root = f'../../all_datasets/{opt.data}/graph_data/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
test_root = f'../../all_datasets/{opt.data}/graph_data/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
val_root = f'../../all_datasets/{opt.data}/graph_data/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'

train_set = myFloder(train_root, load_graphs)
test_set = myFloder(test_root, load_graphs)
val_set = myFloder(val_root, load_graphs)

f = open(opt.seq_path + 'user_ad_res/user_res_all_loss_{}.npy'.format(opt.top_k),'rb')
user_res = np.load(f)
user_res = torch.tensor(user_res)

f = open(opt.seq_path + 'user_ad_res/ad_res_all_loss_{}.npy'.format(opt.top_k),'rb')
ad_res = np.load(f)
ad_res = torch.tensor(ad_res)

opt.feature_size = pd.read_table(opt.raw_data_path + '/feature_size.txt',header=None).values[0][0]+1

train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=0)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=collate, pin_memory=True, num_workers=0)
val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=collate, pin_memory=True, num_workers=0)
 
model = PGNN(opt, opt.feature_size, user_num=user_num, item_num=item_num, user_res= user_res, ad_res= ad_res, input_dim=opt.hidden_size, item_max_length=opt.item_max_length, user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, layer_num=opt.layer_num).cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()
log_loss_func = nn.functional.binary_cross_entropy

best_val_loss = 1.0
for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    epoch_logloss = 0
    epoch_pred_logloss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    all_train_label = []
    all_train_predictions = []
    all_pred_train_prediction = []
    model.train()
    for user_org, user, batch_graph, target_org, target, last_item, label, cur_time, feature, u_t, ad_t in train_data:
        iter += 1
        score, pred = model(user_org.to(device), user.to(device), batch_graph.to(device), target_org.to(device), target.to(device), last_item.to(device),  feature.to(device), u_t.to(device), ad_t.to(device), dropout_keep_prob=opt.dropout_keep_prob, is_training=True)
        pred_train_logloss = log_loss_func(torch.sigmoid(pred), label.float().to(device))
    
        optimizer.zero_grad()
        pred_train_logloss.backward()
        optimizer.step()
        
        epoch_pred_logloss += pred_train_logloss.item()
        all_train_label.extend(label.tolist())
        all_pred_train_prediction.extend(torch.sigmoid(pred).tolist())

        train_pred_auc = roc_auc_score(all_train_label, all_pred_train_prediction)
        if iter % 20 == 0:
            model.eval()
            all_loss = []
            all_logloss = []
            all_pred_logloss = []
            all_label = []
            all_predictions = []
            all_pred_predictions = []
            with torch.no_grad():
                for user_org, user, batch_graph, target_org, target, last_item, label, cur_time, feature, u_t, ad_t in val_data:
                    score, pred = model(user_org.to(device), user.to(device), batch_graph.to(device), target_org.to(device), target.to(device), last_item.to(device), feature.to(device), u_t.to(device), ad_t.to(device), dropout_keep_prob=opt.dropout_keep_prob, is_training=False)

                    val_pred_logloss = log_loss_func(torch.sigmoid(pred), label.float().to(device))

                    all_label.extend(label.tolist())
                    all_pred_predictions.extend(torch.sigmoid(pred).tolist())
                    all_pred_logloss.append(val_pred_logloss.item())

                pred_auc = roc_auc_score(all_label, all_pred_predictions)
                    
                if np.mean(all_pred_logloss) < best_val_loss:
                    best_val_loss = np.mean(all_pred_logloss)
                    torch.save(model.state_dict(),  model_file + '/model_weight.mdl')
            model.train()        
            print('Iter {}, train: pred_logloss {:.4f}, pred_auc {:.4f}'.format(iter, epoch_pred_logloss/iter,  train_pred_auc), 'valid: pred_logloss {:.4f}, pred_auc {:.4f}'.format(np.mean(all_pred_logloss), pred_auc), datetime.datetime.now())
    epoch_pred_logloss /= iter  

    model.load_state_dict(torch.load(model_file+'/model_weight.mdl'))
    model.to(device)
    model.eval()
    
    print('start predicting with the best model: ', datetime.datetime.now())
  
    all_pred_logloss = []
    all_label = []
    all_pred_predictions = []
    with torch.no_grad():
        for user_org, user, batch_graph, target_org, target, last_item, label, curr_time, feature, u_t, ad_t in test_data:
            score, pred = model(user_org.to(device), user.to(device), batch_graph.to(device), target_org.to(device), target.to(device), last_item.to(device), feature.to(device), u_t.to(device), ad_t.to(device), dropout_keep_prob=opt.dropout_keep_prob,   is_training=False)

            test_pred_logloss = log_loss_func(torch.sigmoid(pred), label.float().to(device))
            
            all_label.extend(label.tolist())
            all_pred_predictions.extend(torch.sigmoid(pred).tolist())
            all_pred_logloss.append(test_pred_logloss.item()) 
        pred_auc = roc_auc_score(all_label, all_pred_predictions)
           
        print('test_pred_logloss {:.4f}, pre_AUC {:.4f}'.format(np.mean(all_pred_logloss), pred_auc), datetime.datetime.now())
                