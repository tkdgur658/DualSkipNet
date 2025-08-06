import numpy as np
import pandas as pd
import os
from datetime import datetime
import timeit
import random
import sys
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import neg, optim
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split
from tkinter import N

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False
def format_time(time):
    m, s = divmod(time, 60);h, m = divmod(m, 60);Time = "%02d:%02d:%02d" % (h, m, s);
    return Time
    
class AverageMeter(object):
    def __init__(self):
        self.reset ()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def precision_at_k(r, k):
    """Calculate precision at k"""
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def recall_at_k(r, k, all_pos_items):
    """Calculate recall at k"""
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_items

def average_precision_at_k(r, k):
    """Calculate average precision at k"""
    r = np.asarray(r)[:k]
    out = [precision_at_k(r, i + 1) for i in range(len(r)) if r[i]]
    if not out:
        return 0.
    return np.mean(out)

def dcg_at_k(r, k):
    """Calculate discounted cumulative gain at k"""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """Calculate normalized discounted cumulative gain at k"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_metrics(score, true, k=10):
    actual = np.array(true)
    predicted = np.array(score)
    
    # Get the top k predicted indices
    top_k_indices = np.argsort(predicted)[::-1][:k]
    top_k_predicted = actual[top_k_indices]
    
    return {
        "Recall@K": recall_at_k(top_k_predicted, k, np.sum(actual)),
        "Precision@K": precision_at_k(top_k_predicted, k),
        "MAP@K": average_precision_at_k(top_k_predicted, k),
        "NDCG@K": ndcg_at_k(top_k_predicted, k)
    }

def evaluation(uids, outputs, targets, TOP_Ks):
    eval_data = np.vstack([uids, outputs, targets]).T
    user_watch={}
    for i in range(len(eval_data)):
        user_id = eval_data[i][0]
        watch = eval_data[i][2]
        probability = eval_data[i][1]
        if(user_id not in user_watch):
            user_watch[user_id]=[]
        user_watch[user_id].append((watch,probability))
    precision=[0]*len(TOP_Ks);recall=[0]*len(TOP_Ks);map=[0]*len(TOP_Ks);ndcg=[0]*len(TOP_Ks);

    count=0
    for i, key in enumerate(user_watch.keys()):
        arr = np.array(user_watch[key]).T
        gt_scores = arr[0]
        predicted_scores = arr[1]
        if np.sum(gt_scores) == 0:
            continue
        count+=1
        for j, TOP_K in enumerate(TOP_Ks):
            # print(TOP_K, end=' ')
            metrics = calculate_metrics(predicted_scores, gt_scores, k=TOP_K)
            
            precision[j]+=metrics['Precision@K'];recall[j]+=metrics['Recall@K'];map[j]+=metrics['MAP@K'];ndcg[j]+=metrics['NDCG@K'];
    precision = np.round(np.array(precision)/count,3);recall = np.round(np.array(recall)/count,3);map = np.round(np.array(map)/count,3); ndcg = np.round(np.array(ndcg)/count,3); 
    # Create result list with one dict per TOP_K
    evals = []
    for i in range(len(TOP_Ks)):
        evals.append({
            "Precision@K": precision[i],
            "Recall@K": recall[i],
            "MAP@K": map[i],
            "NDCG@K": ndcg[i]
        })
    
    return evals
def collect_performances(evals_bi, evals_topk, Experiments_Time, Train_date, Dataset_Name, model_name, iteration, Best, Best_Epoch, train_time, end_epoch, eval_time, FILENAME, TOP_Ks):
    # Start with base metrics
    Performances = [Experiments_Time, Train_date, Dataset_Name, model_name, iteration, Best]
    Performances.extend([
            evals_bi["AUC"], 
            evals_bi["PR-AUC"], 
            evals_bi["LogLoss"], 
        ])
    # Add metrics for each TOP_K
    for i in range(len(TOP_Ks)):
        Performances.extend([
            evals_topk[i]["Precision@K"], 
            evals_topk[i]["Recall@K"], 
            evals_topk[i]["MAP@K"], 
            evals_topk[i]["NDCG@K"]
        ])
    
    # Add end metrics
    Performances.extend([Best_Epoch, format_time(train_time/end_epoch), format_time(eval_time), FILENAME])
    
    return Performances
def prepare_interaction(Dataset_dir, 
                        file_name='Total_Interactions.csv', 
                        ):
    # CSV 파일 경로를 인자로 전달받은 파일 이름으로 구성합니다.
    file_path = os.path.join(Dataset_dir, file_name)
    total_interaction = pd.read_csv(file_path)
    
    total_u_id_list = np.array(range(np.unique(np.array(total_interaction['user_id'])).shape[0]))
    total_v_id_list = np.array(range(np.unique(np.array(total_interaction['item_id'])).shape[0]))
    
    v_num = len(total_v_id_list)
    u_num = len(total_u_id_list)
    return total_interaction, total_u_id_list, total_v_id_list, u_num, v_num
def load_data_split(dataset_root, train_ratio, val_ratio, test_ratio, iteration, dataset_name):
    """
    주어진 매개변수를 사용하여, 지정된 폴더 내에서 Train, Validation, Test 데이터셋 CSV 파일들을 읽어 반환합니다.
    
    Parameters:
        dataset_root (str): 데이터셋의 루트 경로.
        train_ratio (float): 학습 데이터 비율 (예: 0.6)
        val_ratio (float): 검증 데이터 비율 (예: 0.2)
        test_ratio (float): 테스트 데이터 비율 (예: 0.2)
        iteration (int): 불러올 iteration 번호.
        dataset_name (str): 해당 데이터셋의 이름.
        
    Returns:
        tuple: (train_interaction, validation_interaction, test_interaction)
            - train_interaction (DataFrame): 학습 데이터셋.
            - validation_interaction (DataFrame): 검증 데이터셋.
            - test_interaction (DataFrame): 테스트 데이터셋.
    """
    split_root = f"{dataset_root}/(DataSplit)_Data_splits_{int(train_ratio*100)}_{int(val_ratio*100)}_{int(test_ratio*100)}_10/{dataset_name}"
    
    train_path = os.path.join(split_root, f"Train_Data_Iter_{iteration}.csv")
    validation_path = os.path.join(split_root, f"Validation_Data_Iter_{iteration}.csv")
    test_path = os.path.join(split_root, f"Test_Data_Iter_{iteration}.csv")
    
    train_interaction = pd.read_csv(train_path)
    validation_interaction = pd.read_csv(validation_path)
    test_interaction = pd.read_csv(test_path)
    
    return train_interaction, validation_interaction, test_interaction


def train_model(model, train_loader, val_loader, output_dir, optimizer, lr_scheduler, device, iteration, model_name, model_path, epochs, TOP_Ks, EARLY_STOP, val_criterion_top_k, tqdm):
    """
    Train model with Loss.csv file creation for tracking metrics.
    """
    Best = -1
    Best_Epoch = 1
    Early_Stop = 0
    train_start_time = timeit.default_timer()
    
    # Create a DataFrame to track metrics
    loss_csv_path = os.path.join(output_dir, 'Loss.csv')
    loss_df = pd.DataFrame(columns=['Epoch', 'Time', 'Train_Loss', 'Validation_Loss'])
    
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = AverageMeter()
        for step, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch", leave=False)):
            optimizer.zero_grad()
            train_loss = model.calculate_loss(batch)
            train_loss.backward(retain_graph=True)
            optimizer.step()
            try:
                train_losses.update(train_loss.detach().cpu().numpy(), batch['label'].shape[0])
            except:
                train_losses.update(train_loss.detach().cpu().numpy(), batch[0].shape[0])
                
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc="Validation", unit="batch", leave=False)):
                val_output = model(batch)
                if step == 0:
                    targets = batch['label'].cpu().numpy()
                    outputs = val_output.cpu().numpy()
                    uids = batch['user_id']
                else:
                    targets = np.concatenate((targets, batch['label'].cpu().numpy()), axis=0)
                    outputs = np.concatenate((outputs, val_output.cpu().numpy()), axis=0)
                    uids = np.concatenate((uids, batch['user_id']), axis=0)
        evals = evaluation(uids, outputs, targets, TOP_Ks)    
        infer_date = datetime.now().strftime("%y%m%d_%H%M%S")
        train_loss_avg = round(train_losses.avg.item(), 6)
        val_recall = evals[val_criterion_top_k]['Recall@K']
        # Save metrics to CSV file
        try:
            # Add current epoch data to DataFrame
            new_row = pd.DataFrame({
                'Epoch': [epoch], 
                'Time': [infer_date], 
                'Train_Loss': [train_loss_avg], 
                'Validation_Recall': [val_recall]
            })
            loss_df = pd.concat([loss_df, new_row], ignore_index=True)
            
            # Save the updated DataFrame to CSV
            loss_df.to_csv(loss_csv_path, index=False)
        except Exception as e:
            print(f"\nError saving metrics to CSV: {e}", end=' ')
        
        print(f'{epoch} EP({infer_date}): Loss: train:{train_loss_avg}, val recall@{TOP_Ks[val_criterion_top_k]}:{val_recall}', end=' ')
        if Best < val_recall:
            torch.save(model.state_dict(), model_path)
            Best = val_recall
            Best_Epoch = epoch
            Early_Stop = 0
            print(f"Best Epoch: {epoch}, val recall@{TOP_Ks[val_criterion_top_k]}: {val_recall}")
        else:
            print('')
            Early_Stop += 1
        if Early_Stop >= EARLY_STOP:
            break
    print(f"Train End ({datetime.now().strftime('%y%m%d_%H%M%S')})")
    train_stop_time = timeit.default_timer()
    return Best, Best_Epoch, train_stop_time-train_start_time, epoch 

def test_model(model, test_loader, file_path, TOP_Ks, tqdm):
    Test_Time = datetime.now().strftime("%y%m%d_%H%M%S")
    print(f"Test Start ({Test_Time})")
    eval_start_time = timeit.default_timer()
    model.load_state_dict(torch.load(file_path))
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc="Test", unit="batch", leave=False)):
            test_output = model(batch)
            if step == 0:
                targets = batch['label'].cpu().numpy()
                outputs = test_output.cpu().numpy()
                uids = batch['user_id']
                vids = batch['item_id']
            else:
                targets = np.concatenate((targets, batch['label'].cpu().numpy()), axis=0)
                outputs = np.concatenate((outputs, test_output.cpu().numpy()), axis=0)
                uids = np.concatenate((uids, batch['user_id']), axis=0)
                vids = np.concatenate((vids, batch['item_id']), axis=0)
    auc = roc_auc_score(targets, sigmoid(outputs))
    pr_auc = average_precision_score(targets, sigmoid(outputs))
    logloss = log_loss(targets, sigmoid(outputs))
    evals_bi = {
        "AUC": auc,
        "PR-AUC": pr_auc,
        "LogLoss": logloss
    }
    evals = evaluation(uids, outputs, targets, TOP_Ks)
    print(f"Test({datetime.now().strftime('%y%m%d_%H%M%S')}):")
    print("    AUC:", f"{auc:.4f},", end=' ')
    print("PR-AUC:", f"{pr_auc:.4f},", end=' ')
    print("LogLoss:", f"{logloss:.4f},")
    for i, k in enumerate(TOP_Ks):
        print(f"    precision@{k}: {evals[i]['Precision@K']}, recall@{k}: {evals[i]['Recall@K']}, map@{k}: {evals[i]['MAP@K']}, ndcg@{k}: {evals[i]['NDCG@K']}")
    
    eval_stop_time = timeit.default_timer()
    return evals_bi, evals, eval_stop_time-eval_start_time, (outputs, targets, uids, vids)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_results(predictions, output_dir):
    outputs, targets, uids, vids = predictions
    df = pd.DataFrame({
        'uid': uids,
        'vid': vids,
        'target': targets,
        'output': 1 / (1 + np.exp(-outputs)),
    })
    df['rank_within_uid'] = df.groupby('uid')['output'].rank(method='first', ascending=False).astype(int)
    df['uid_count'] = df.groupby('uid')['uid'].transform('count')
    df = df.sort_values(['uid','target'], ascending=[True,False])
    df.to_csv(f'{output_dir}/test_results.csv', index=False)