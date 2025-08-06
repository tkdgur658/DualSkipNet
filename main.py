import os
import sys
import timeit
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import *
from datetime import datetime
from tqdm import tqdm
sys.path.append("..")
def str_to_class(classname):
    """Convert string to class object"""
    # First try to get from current module
    try:
        return getattr(sys.modules[__name__], classname)
    except AttributeError:
        # If not found, try to get from globals (imported classes)
        if classname in globals():
            return globals()[classname]
        else:
            raise AttributeError(f"Class '{classname}' not found. Make sure it's properly imported.")

# FILENAME = os.getcwd()+'/'+str(__session__).split('/')[-1]
train_mode = 'train' # train, inference
if train_mode=='inference':
    past_output_root = f'output' # if train_mode=='inference'
    past_result_csv = 'output/Mircro_Video_Recommendation_250522_021058.csv' # output vis시 수정 1/2
    past_result_df = pd.read_csv(past_result_csv)
    iteration_dataset_model_tuples = list(zip(past_result_df['Iteration'], past_result_df['Dataset Name'], past_result_df['Model Name']))
SAVE_RESULT = False
Dataset_root = 'Micro_Video_Datasets'
Dataset_Name_list = ['KuaiRand-1K-Rand'] 
model_names = [ 'DPSkipNet']
model_dir = 'models'
output_root = 'output'
Graph_root = 'graphs'
for model_name in model_names:
    exec(f'from {model_dir}.{model_name} import *')

iterations = [1, 1] # 1
epochs = 30
batch_size = 1024 # 2
num_workers = 16 # 0
EARLY_STOP = 3

optimizer = 'AdamW'
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
optim_args = {'optimizer': optimizer, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}

lr_scheduler = 'CosineAnnealingLR'
T_max = epochs
T_0 = epochs
eta_min = 1e-6
lr_scheduler_args = {'lr_scheduler': lr_scheduler, 'T_max': T_max, 'T_0': T_0, 'eta_min': eta_min}

devices = [0]
train_ratio, val_ratio, test_ratio = ratio_combination = (0.7, 0.15, 0.15)
TOP_Ks= [3, 5, 10, 20]
base_metrics = ['Experient Time', 'Train Time', 'Dataset Name', 'Model Name', 'Iteration', 'Val Recall','AUC','PR-AUC','LogLoss']
end_metrics = ['Best_Epoch', 'Time per Epoch', 'Infer Time', 'DIR']
val_criterion_top_k = 1
Metrics = base_metrics.copy()
for k in TOP_Ks:
    Metrics.extend([f'Precision@{k}', f'Recall@{k}', f'MAP@{k}', f'NDCG@{k}'])
Metrics.extend(end_metrics)

Experiments_Time = datetime.now().strftime("%y%m%d_%H%M%S")
print(f'Experiments Start ({Experiments_Time})')
df = pd.DataFrame(index=None, columns=Metrics)
output_root_ex = f'{output_root}/output_{Experiments_Time}'
for iteration in range(iterations[0], iterations[1] + 1):
    for j, Dataset_Name in enumerate(Dataset_Name_list):
        Dataset_dir = Dataset_root +'/' + Dataset_Name
        total_interaction, total_u_id_list, total_v_id_list, u_num, v_num = prepare_interaction(Dataset_dir)
        if 'Kuaishou' in Dataset_Name:
            embed_dim=64
        else:
            embed_dim=128
        for k, model_name in enumerate(model_names):
            if train_mode=='inference' and ((iteration, Dataset_Name, model_name) not in iteration_dataset_model_tuples):
                continue
            print(f'({model_name} ({k+1}/{len(model_names)})) Iter {iteration} Start ({datetime.now().strftime("%y%m%d_%H%M%S")})',end= ' ')
            print(f'Dataset: {Dataset_Name} ({j+1}/{len(Dataset_Name_list)})')
            seed = iteration
            Graph_path = Graph_root + f'/{model_name}/{Dataset_Name}/Iter_{iteration}'
            os.makedirs(Graph_path, exist_ok=True)
            output_dir = output_root_ex + f'/{model_name}_{Dataset_Name}_Split_{int(ratio_combination[0]*100)}_{int(ratio_combination[1]*100)}_{int(ratio_combination[2]*100)}_Iter_{iteration}'
            os.makedirs(output_dir, exist_ok=True)
            
            if Dataset_Name in ['KuaiRand-Pure', 'KuaiRand-1K','KuaiRand-1K-Rand','KuaiRec_small','KuaiRec_big']:
                control_random_seed(seed)
                visual_emb = nn.Parameter(torch.randn(v_num, 128))
            else:
                if 'FRAME' in model_name and Dataset_Name=='MVA':
                    visual_emb = torch.tensor(np.load(Dataset_dir+'/visual_features_FRAME.npy'))
                else:
                    visual_emb = torch.tensor(np.load(Dataset_dir+'/visual_features.npy'))
       
            split_root = f"{Dataset_root}/(DataSplit)_Data_splits_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}_10/{Dataset_Name}"
            train_interaction, validation_interaction, test_interaction = load_data_split(Dataset_root, train_ratio, val_ratio, test_ratio, iteration, Dataset_Name)
            
            ########### 모델 개발 테스트 코드 실행시 주석 해제
            # selected_users = train_interaction['user_id'].unique()[:10]
            # train_interaction = train_interaction[train_interaction['user_id'].isin(selected_users)]
            # validation_interaction = validation_interaction[validation_interaction['user_id'].isin(selected_users)]
            # test_interaction = test_interaction[test_interaction['user_id'].isin(selected_users)]
            # Graph_path = f'graphs_toy/{model_name}/{Dataset_Name}'
            # os.makedirs(Graph_path, exist_ok=True)
            # output_dir = output_dir.replace(output_root,output_root+'_toy')
            # os.makedirs(output_dir, exist_ok=True)
            ###########
            
            device = torch.device("cuda:" + str(devices[0]))
            
            # 그래프 생성
            control_random_seed(seed)
            # if model_name in ['DPSkipNet','LightGT','FRAME']:
            if 'FRAME' in model_name:
                num_clips = 4 if Dataset_Name=='MVA' else 1
                model = str_to_class(model_name)(visual_emb, Graph_path, embed_dim,
                                                 train_interaction, validation_interaction, test_interaction, 
                                                 total_u_id_list, total_v_id_list, 
                                                 batch_size, num_clips=num_clips)
            else:
                model = str_to_class(model_name)(visual_emb, Graph_path, embed_dim,
                                                 train_interaction, validation_interaction, test_interaction, 
                                                 total_u_id_list, total_v_id_list, 
                                                 batch_size)
            if len(devices) > 1:
                model = torch.nn.DataParallel(model, device_ids=devices).to(device)
            else:
                model = model.to(device)
            if optim_args['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            if lr_scheduler_args['lr_scheduler'] == 'CosineAnnealingLR':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=lr_scheduler_args['T_max'], eta_min=lr_scheduler_args['eta_min']
                )
            
            # 학습 수행
            
            if train_mode == 'train':
                Train_date = datetime.now().strftime("%y%m%d_%H%M%S") 
                print(f'Train Start ({Train_date})')
                model_path=f'{output_dir}/{Train_date}_{model_name}_{Dataset_Name}_Split_{int(ratio_combination[0]*100)}_{int(ratio_combination[1]*100)}_{int(ratio_combination[2]*100)}_Iter_{iteration}.pt'
                Best, Best_Epoch, train_time, end_epoch = train_model(model, model.train_loader, model.val_loader, output_dir, optimizer, lr_scheduler, device, iteration, model_name, model_path,epochs, TOP_Ks, EARLY_STOP, val_criterion_top_k, tqdm)
            elif train_mode == 'inference':
                corresponding_df = past_result_df[(past_result_df['Model Name']==model_name)&
                (past_result_df['Iteration']==iteration)&
                (past_result_df['Dataset Name']==Dataset_Name)]
                Train_date = corresponding_df['Train Time'].item()
                ex_time = corresponding_df['Experient Time'].item()
                Best, Best_Epoch, train_time, end_epoch = 0, 1, 0, 1
                model_path=f'{past_output_root}/output_{ex_time}/{model_name}_{Dataset_Name}_Split_{int(ratio_combination[0]*100)}_{int(ratio_combination[1]*100)}_{int(ratio_combination[2]*100)}_Iter_{iteration}/{Train_date}_{model_name}_{Dataset_Name}_Split_{int(ratio_combination[0]*100)}_{int(ratio_combination[1]*100)}_{int(ratio_combination[2]*100)}_Iter_{iteration}.pt'
                model.load_state_dict(torch.load(model_path))
            # 테스트 수행
            evals_bi, evals_topk, eval_time, predictions = test_model(model, model.test_loader, model_path, TOP_Ks, tqdm)
            FILENAME = "None"
            if SAVE_RESULT ==True:
                save_results(predictions, output_dir)
                if train_mode == 'inferece':
                    torch.save(model.state_dict(), model_path.replace(ex_time, Experiments_Time))
            Performances = collect_performances(evals_bi, evals_topk, Experiments_Time, Train_date, Dataset_Name, model_name, iteration, Best, Best_Epoch, train_time, end_epoch, eval_time, FILENAME, TOP_Ks)
            df = pd.concat([df, pd.DataFrame([Performances], columns=Metrics)], ignore_index=True)
            df.to_csv(f'{output_root_ex}/Mircro_Video_Recommendation_{Experiments_Time}.csv', index=False, header=True, encoding="cp949")
