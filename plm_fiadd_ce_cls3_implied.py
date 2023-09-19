import datasets
import csv
import pandas as pd
import preprocessor as p
import transformers
from transformers import BertTokenizer, TFBertModel
import pickle
from scipy import spatial
from scipy.spatial.distance import cdist
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from transformers import AutoModel,AutoTokenizer, AutoModelForMaskedLM

import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import random
from sklearn.model_selection import train_test_split
import sys
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from kmeans_pytorch import kmeans
from read_dataset import 
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

#____________________________________ Define Parameters _________________________________
random_seed = int(sys.argv[1])#1 4 7
dataset_name = "abuse" # gab, latent
ifHateBert = False
max_length = 90
model_name = "bert-base-uncased"
out_file_n = "output.csv"
model_saving_path = "tr_models/model_"+dataset_name+"_c3m3_implied.pt"
experiment_type = ace #ace, ace_add_foc, ace_add_inf_foc
N_EPOCHS = 5000

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

set_seed(random_seed)

#____________________________________ Reading Dataset _________________________________

train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr =get_dataset(dataset_name,random_seed, ifHateBert):


class torchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)



train_labels1 =train_labels# [s if (s==0) else 1 for s in train_labels]
valid_labels1 =valid_labels# [s if (s==0) else 1 for s in valid_labels]
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
#train_dataset = torchDataset(train_encodings, train_labels1)
#valid_dataset = torchDataset(valid_encodings, valid_labels1)

t3labels = torch.tensor([train_labels]).transpose(0,1).to(device)
v3labels = torch.tensor([valid_labels]).transpose(0,1).to(device)
#tdata = {k:torch.tensor(v).to(device) for k,v in train_encodings.items()}
tdata = torch.tensor(train_texts).to(device)
tlabels = torch.tensor([train_labels1]).transpose(0,1).to(device)

#vdata = {k:torch.tensor(v).to(device) for k,v in valid_encodings.items()}
vdata = torch.tensor(valid_texts).to(device)
vlabels = torch.tensor([valid_labels1]).transpose(0,1).to(device)

#____________________________________ Defining FiADD Assuems Frozen PLM Input _________________________________

import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, drop):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(768,128)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        logits = self.fc(output)
        return logits,output


    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


INPUT_DIM = 30523
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
OUTPUT_DIM = 3
N_LAYERS =1
BIDIRECTIONAL = True
DROPOUT = 0.2

model = Model(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

#pretrained_embeddings = TEXT.vocab.vectors
#model.embedding.weight.data.copy_(pretrained_embeddings)

import torch.optim as optim

#____________________________________ Define Learning Settings _________________________________

optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = 1e-3,steps_per_epoch=1,epochs=5000,anneal_strategy = 'linear',final_div_factor = 100000.0)
#criterion = nn.CrossEntropyLoss()
model = model.to(device)
#criterion = criterion.to(device)

if dataset_name == "abuse":
    weights = torch.tensor([585/2820.0,2394/2820.0,2661/2820.0])
if dataset_name == "gab":
    weights = torch.tensor([512/5533.0,5101/5533.0,5452/5533.0])
if dataset_name == "latent":
    weights = torch.tensor([1637/4296.0,4078/4296.0,2876/4296.0])

#criterion=nn.NLLLoss(weight= weights)
criterion = nn.CrossEntropyLoss(weight = weights)
criterion = criterion.to(device)


#____________________________________ Extract Metrics from Classification Head _________________________________
def categorical_accuracy(preds, y):

    if dataset_name == "abuse":
        valid_set_detect = 3000
    if dataset_name == "gab":
        valid_set_detect = 6000
    if dataset_name == "latent":
        valid_set_detect = 6000
    
    lab = y.flatten().cpu().numpy()
    pred = preds.argmax(-1).cpu().numpy()

    acc =",".join(list(map(str,confusion_matrix(lab, pred,normalize="true").diagonal().tolist())))
    f1_cl = ",".join(list(map(str,f1_score(lab, pred,average=None).tolist())))
    f1_macro = str(f1_score(lab, pred,average='macro'))
    acc2 = ""
    #for validation data
    if(len(pred)<valid_set_detect):
        pred_3lables = [2 if (valid_labels[r]==2 and pred[r]==1) else pred[r] for r in range(len(pred))]
        acc2 = ",".join(list(map(str,confusion_matrix(valid_labels, pred_3lables,normalize="true").diagonal().tolist())))
    
    if dataset_name == "abuse":
        acc = acc +"," +str((float(acc.split(",")[0])+float(acc.split(",")[1]) + float(acc.split(",")[2]))/3.0) +","+ str((float(acc.split(",")[0])*2235+float(acc.split(",")[1])*426 + float(acc.split(",")[2])*159 )/2820)+","+f1_cl+","+f1_macro+","+acc2
    if dataset_name == "gab":
        acc = acc +"," +str((float(acc.split(",")[0])+float(acc.split(",")[1]) + float(acc.split(",")[2]))/3.0) +","+ str((float(acc.split(",")[0])*5020+float(acc.split(",")[1])*431 + float(acc.split(",")[2])*80 )/5533)+","+f1_cl+","+f1_macro+","+acc2
    if dataset_name == "latent":
        acc = acc +"," +str((float(acc.split(",")[0])+float(acc.split(",")[1]) + float(acc.split(",")[2]))/3.0) +","+ str((float(acc.split(",")[0])*2658+float(acc.split(",")[1])*217 + float(acc.split(",")[2])*1420 )/4296)+","+f1_cl+","+f1_macro+","+acc2


    #ff.write(acc+"\n")
    return acc

train_instance_losses = torch.tensor([0.0]*len(train_labels1)).to(device)
val_instance_losses = torch.tensor([0.0]*len(valid_labels1)).to(device)
train_cluster_means = torch.tensor([]).to(device)
train_valid_cluster = torch.tensor([0,0,0,0,0,0,0,0,0]).to(device)

#____________________________________ ADD_INF_FOC LOSS _________________________________

def m_loss(model,out,current_ep):
    _,pool_imp_implied = model(implied_tr)
    
    magnet_loss = torch.tensor(0).to(device)
    emb_size = 128
    pool =out
    #print (pool)
    #print(model.training)
    if model.training:
        lab_list = t3labels.flatten()
    else:
        lab_list = v3labels.flatten()

    idx0 = (lab_list==0).nonzero(as_tuple=False).squeeze()
    pool0 = torch.index_select(pool,0,idx0)
    idx1 = (lab_list==1).nonzero(as_tuple=False).squeeze()
    pool1 = torch.index_select(pool,0,idx1)
    idx2 = (lab_list==2).nonzero(as_tuple=False).squeeze()
    pool2 =  torch.index_select(pool,0,idx2)

    #### EXp4 ################
    if model.training:
        lab_list2 = tlabels.flatten()
    else:
        lab_list2 = vlabels.flatten()

    idxoff = (lab_list2==1).nonzero(as_tuple=False).squeeze()
    poolOFF = torch.index_select(pool,0,idxoff)
        
    num_clus = 3
    max_sample_in_cluster=500
    cluster_id_n, centroid_n = kmeans(X=pool0,num_clusters=3,iter_limit=100,device=torch.device('cuda:0'))
    #print(cluster_id_n)
    cluster_id_o, centroid_o = kmeans(X=pool1,num_clusters=3,iter_limit=100,device=torch.device('cuda:0'))
    cluster_id_i, centroid_i = kmeans(X=pool2,num_clusters=3,iter_limit=100,device=torch.device('cuda:0'))
    
    cluster_id_n=cluster_id_n.to(device)
    cluster_id_o=cluster_id_o.to(device)
    cluster_id_i=cluster_id_i.to(device)
    valid_clusters = torch.tensor([0,0,0,0,0,0,0,0,0])
    mean_n0 = torch.tensor([0.0]*emb_size).to(device)
    mean_n1 = torch.tensor([0.0]*emb_size).to(device)
    mean_n2 = torch.tensor([0.0]*emb_size).to(device)
    idc_n0 = (cluster_id_n==0).nonzero(as_tuple=False).squeeze()
    #select_id = random.sample(range(0,idc_n0.size(dim=0)),100)
    if (cluster_id_n==0).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_n0.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_n0 = (torch.index_select(idc_n0,0,select_id))
    cluster_n0 = torch.index_select(pool0,0,idc_n0)
    idc_n1 = (cluster_id_n==1).nonzero(as_tuple=False).squeeze()
    if (cluster_id_n==1).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_n1.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_n1 = (torch.index_select(idc_n1,0,select_id))
    cluster_n1 = torch.index_select(pool0,0,idc_n1)
    idc_n2 = (cluster_id_n==2).nonzero(as_tuple=False).squeeze()
    if (cluster_id_n==2).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_n2.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_n2 = (torch.index_select(idc_n2,0,select_id))
    cluster_n2 = torch.index_select(pool0,0,idc_n2)
    if((cluster_id_n==0).sum()>1):
        mean_n0 = torch.mean(cluster_n0,0)
        valid_clusters[0] = 1
    if((cluster_id_n==1).sum()>1):
        mean_n1 = torch.mean(cluster_n1,0)
        valid_clusters[1] = 1
    if((cluster_id_n==2).sum()>1):
        mean_n2 = torch.mean(cluster_n2,0)
        valid_clusters[2] = 1
            
    mean_o0 = torch.tensor([0.0]*emb_size).to(device)
    mean_o1 = torch.tensor([0.0]*emb_size).to(device)
    mean_o2 = torch.tensor([0.0]*emb_size).to(device)
    idc_o0 = (cluster_id_o==0).nonzero(as_tuple=False).squeeze()
    if (cluster_id_o==0).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_o0.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_o0 = (torch.index_select(idc_o0,0,select_id))
    cluster_o0 = torch.index_select(pool1,0,idc_o0)
    idc_o1 = (cluster_id_o==1).nonzero(as_tuple=False).squeeze()
    if (cluster_id_o==1).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_o1.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_o1 = (torch.index_select(idc_o1,0,select_id))
    cluster_o1 = torch.index_select(pool1,0,idc_o1)
    idc_o2 = (cluster_id_o==2).nonzero(as_tuple=False).squeeze()
    if (cluster_id_o==2).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_o2.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_o2 = (torch.index_select(idc_o2,0,select_id))
    cluster_o2 = torch.index_select(pool1,0,idc_o2)
    if((cluster_id_o==0).sum()>1):
        mean_o0 = torch.mean(cluster_o0,0)
        valid_clusters[3] = 1
    if((cluster_id_o==1).sum()>1):
        mean_o1 = torch.mean(cluster_o1,0)
        valid_clusters[4] = 1
    if((cluster_id_o==2).sum()>1):
        mean_o2 = torch.mean(cluster_o2,0)
        valid_clusters[5] = 1

    
    mean_i0 = torch.tensor([0.0]*emb_size).to(device)
    mean_i1 = torch.tensor([0.0]*emb_size).to(device)
    mean_i2 = torch.tensor([0.0]*emb_size).to(device)
    
    mean_implied_i0 = torch.tensor([0.0]*emb_size).to(device)
    mean_implied_i1 = torch.tensor([0.0]*emb_size).to(device)
    mean_implied_i2 = torch.tensor([0.0]*emb_size).to(device)

    idc_i0 = (cluster_id_i==0).nonzero(as_tuple=False).squeeze()
    if (cluster_id_i==0).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_i0.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_i0 = (torch.index_select(idc_i0,0,select_id))
    cluster_i0 = torch.index_select(pool2,0,idc_i0)
    cluster_imp_i0 = torch.index_select(pool_imp_implied,0,idc_i0)
    idc_i1 = (cluster_id_i==1).nonzero(as_tuple=False).squeeze()
    if (cluster_id_i==1).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_i1.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_i1 = (torch.index_select(idc_i1,0,select_id))
    cluster_i1 = torch.index_select(pool2,0,idc_i1)
    cluster_imp_i1 = torch.index_select(pool_imp_implied,0,idc_i1)
    idc_i2 = (cluster_id_i==2).nonzero(as_tuple=False).squeeze()
    if (cluster_id_i==2).sum()>max_sample_in_cluster:
        select_id = torch.randint(0,idc_i2.size(dim=0),(max_sample_in_cluster,)).to(device)
        idc_i2 = (torch.index_select(idc_i2,0,select_id))
    cluster_i2 = torch.index_select(pool2,0,idc_i2)
    cluster_imp_i2 = torch.index_select(pool_imp_implied,0,idc_i2)
    if((cluster_id_i==0).sum()>1):
        mean_i0 = torch.mean(cluster_i0,0)
        mean_implied_i0 = torch.mean(cluster_imp_i0,0)
        valid_clusters[6] = 1
    if((cluster_id_i==1).sum()>1):
        mean_i1 = torch.mean(cluster_i1,0)
        mean_implied_i1 = torch.mean(cluster_imp_i1,0)
        valid_clusters[7] = 1
    if((cluster_id_i==2).sum()>1):
        mean_i2 = torch.mean(cluster_i2,0)
        mean_implied_i2 = torch.mean(cluster_imp_i2,0)
        valid_clusters[8] = 1
    implied_imp_means = torch.stack((mean_implied_i0,mean_implied_i1,mean_implied_i2))

    cluster_means = torch.stack((mean_n0,mean_n1,mean_n2,mean_o0,mean_o1,mean_o2,mean_i0,mean_i1,mean_i2)).to(device)
    global train_valid_cluster
    global train_cluster_means
    global train_instance_losses
    global val_instance_losses
    if model.training:
        train_cluster_means=cluster_means.detach().clone()
        train_valid_cluster= valid_clusters.detach().clone()
    M =2
    #print(cluster_n0,cluster_n1,cluster_n2,cluster_o0,cluster_o1,cluster_o2)
    random.seed(None)
    #previous loss
    prev_c_loss_n0 = torch.tensor(-1.0).to(device)
    prev_c_loss_n1 = torch.tensor(-1.0).to(device)
    prev_c_loss_n2 = torch.tensor(-1.0).to(device)
    prev_c_loss_o0 = torch.tensor(-1.0).to(device)
    prev_c_loss_o1 = torch.tensor(-1.0).to(device)
    prev_c_loss_o2 = torch.tensor(-1.0).to(device)
    prev_c_loss_i0 = torch.tensor(-1.0).to(device)
    prev_c_loss_i1 = torch.tensor(-1.0).to(device)
    prev_c_loss_i2 = torch.tensor(-1.0).to(device)

    if(current_ep>50):
        if model.training:
            if(cluster_n0.size(dim=0)!=0):
                prev_c_loss_n0 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx0,0,idc_n0))).item()
            if(cluster_n1.size(dim=0)!=0):
                prev_c_loss_n1 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx0,0,idc_n1))).item()
            if(cluster_n2.size(dim=0)!=0):
                prev_c_loss_n2 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx0,0,idc_n2))).item()
            if(cluster_o0.size(dim=0)!=0):
                prev_c_loss_o0 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx1,0,idc_o0))).item()
            if(cluster_o1.size(dim=0)!=0):
                prev_c_loss_o1 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx1,0,idc_o1))).item()
            if(cluster_o2.size(dim=0)!=0):
                prev_c_loss_o2 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx1,0,idc_o2))).item()
            if(cluster_i0.size(dim=0)!=0):
                prev_c_loss_i0 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx2,0,idc_i0))).item()
            if(cluster_i1.size(dim=0)!=0):
                prev_c_loss_i1 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx2,0,idc_i1))).item()
            if(cluster_i2.size(dim=0)!=0):
                prev_c_loss_i2 = torch.mean(torch.index_select(train_instance_losses,0,torch.index_select(idx2,0,idc_i2))).item()
        else:
            if(cluster_n0.size(dim=0)!=0):
                prev_c_loss_n0 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx0,0,idc_n0))).item()
            if(cluster_n1.size(dim=0)!=0):
                prev_c_loss_n1 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx0,0,idc_n1))).item()
            if(cluster_n2.size(dim=0)!=0):
                prev_c_loss_n2 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx0,0,idc_n2))).item()
            if(cluster_o0.size(dim=0)!=0):
                prev_c_loss_o0 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx1,0,idc_o0))).item()
            if(cluster_o1.size(dim=0)!=0):
                prev_c_loss_o1 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx1,0,idc_o1))).item()
            if(cluster_o2.size(dim=0)!=0):
                prev_c_loss_o2 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx1,0,idc_o2))).item()
            if(cluster_i0.size(dim=0)!=0):
                prev_c_loss_i0 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx2,0,idc_i0))).item()
            if(cluster_i1.size(dim=0)!=0):
                prev_c_loss_i1 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx2,0,idc_i1))).item()
            if(cluster_i2.size(dim=0)!=0):
                prev_c_loss_i2 = torch.mean(torch.index_select(val_instance_losses,0,torch.index_select(idx2,0,idc_i2))).item()
        #seed_idx = random.choice([0,1,2,3,4,5])
        prev_cluster_losses = [prev_c_loss_n0,prev_c_loss_n1,prev_c_loss_n2,prev_c_loss_o0,prev_c_loss_o1,prev_c_loss_o2,prev_c_loss_i0,prev_c_loss_i1,prev_c_loss_i2]
        #seed_idx = 3 + [prev_c_loss_o0,prev_c_loss_o1,prev_c_loss_o2].index(max(prev_c_loss_o0,prev_c_loss_o1,prev_c_loss_o2))
        seed_idx = prev_cluster_losses.index(max(prev_cluster_losses))
    else:
        #seed_idx = random.choice([3,4,5])
        seed_idx = random.choice([0,1,2,3,4,5,6,7,8])
    #seed_idx = random.choice([0,1,2,3,4,5])
    #seed_idx = random.choice([3,4,5])

    cluster_index = {0:idc_n0,1:idc_n1,2:idc_n2,3:idc_o0,4:idc_o1,5:idc_o2,6:idc_i0,7:idc_i1,8:idc_i2}
    cluster_dict = {0:cluster_n0,1:cluster_n1,2:cluster_n2,3:cluster_o0,4:cluster_o1,5:cluster_o2,6:cluster_i0,7:cluster_i1,8:cluster_i2}
    implied_cluster_dict = {0:cluster_imp_i0,1:cluster_imp_i1,2:cluster_imp_i2}
    if (cluster_dict[seed_idx].size(dim=0) == 0):
        #scope_indexs = [3,4,5]
        scope_indexs = [0,1,2,3,4,5,6,7,8]
        #scope_counts = [cluster_dict[3].size(dim=0),cluster_dict[4].size(dim=0),cluster_dict[5].size(dim=0)]
        scope_indexs = [a for a in scope_indexs if cluster_dict[a].size(dim=0) >1]
        #scope_indexs.remove(seed_idx)
        seed_idx = random.choice(scope_indexs)
    random.seed(random_seed)

    #random.seed(1)

    cluster_classes = torch.tensor([0,0,0,1,1,1,2,2,2])
    cluster_distance = torch.cdist(cluster_means[seed_idx].unsqueeze(0),cluster_means,p=2)
    sorted_idx = torch.sort(cluster_distance)[1][0]
    im = 0
    imposter_idx = torch.tensor([seed_idx]).to(device)
    #print(sorted_idx[0])
    for i in range(1,sorted_idx.size(dim=0)):
        if (cluster_classes[sorted_idx[i]] != cluster_classes[seed_idx]) and (valid_clusters[sorted_idx[i]]==1):
            im = im+1
            imposter_idx= torch.cat((imposter_idx,sorted_idx[i].unsqueeze(0)))
            if im>=M:
                break
    #print(imposter_idx)
    D = 0
    D_implied =0
    alpha =torch.tensor(1.0)
    epsilon = torch.tensor(0.00000001)
    if dataset_name == "abuse":
        class_weights = torch.tensor([585/2820.0,2394/2820.0,2661/2820.0])
    if dataset_name == "gab":
        class_weights = torch.tensor([512/5533.0,5101/5533.0,5452/5533.0])
    if dataset_name == "latent":
        class_weights = torch.tensor([1637/4296.0,4078/4296.0,2876/4296.0])

    avg_intra_cluster_distance_implied = torch.tensor([]).to(device)
    avg_intra_cluster_distance2 = torch.tensor([]).to(device)
            
    for i in range(imposter_idx.size(dim=0)):
        if cluster_classes[imposter_idx[i].item()] ==2:
            D_implied = D_implied + cluster_dict[imposter_idx[i].item()].size(dim=0)
            avg_intra_cluster_distance_implied =  torch.cat((avg_intra_cluster_distance_implied,(torch.sum((cluster_dict[imposter_idx[i].item()]-implied_imp_means[imposter_idx[i].item()-6])**2)/(emb_size)).unsqueeze(0)))
        D = D + cluster_dict[imposter_idx[i].item()].size(dim=0)
        avg_intra_cluster_distance2 =  torch.cat((avg_intra_cluster_distance2,(torch.sum((cluster_dict[imposter_idx[i].item()]-cluster_means[imposter_idx[i].item()])**2)/(emb_size)).unsqueeze(0)))

    variance = torch.sum(avg_intra_cluster_distance2)/(D-1)
    #var_norm = -(1 / (2*(variance**2)))
    var_norm = -(1 / (2*(variance)))
    #print(imposter_idx)
    if(D_implied>1):
        variance_implied = torch.sum(avg_intra_cluster_distance_implied)/(D_implied-1)
    else:
        variance_implied =  torch.sum(avg_intra_cluster_distance_implied)/(D_implied)
    #print("var_imp_orig",variance_implied)
    if variance_implied == 0.0:
        variance_implied = 0.00001
    #variance_implied = torch.sum(avg_intra_cluster_distance_implied)/(D_implied-1)
    var_norm_implied = -(1 / (2*(variance_implied)))
    reduce_dist = torch.tensor(0.0).to(device)
    for i in range(imposter_idx.size(dim=0)):
        #print(cluster_dict[imposter_idx[i].item()].size(dim=0))
        denom = torch.tensor([0.0]*cluster_dict[imposter_idx[i].item()].size(dim=0)).to(device)
        if cluster_classes[imposter_idx[i].item()] ==1:
            class_balance_factor = class_weights[1]
        else:
            class_balance_factor = class_weights[0]
        for j in range(imposter_idx.size(dim=0)):
            if i!=j and cluster_classes[imposter_idx[i].item()] != cluster_classes[imposter_idx[j].item()]:
                denom_cluster = torch.sum((cluster_dict[imposter_idx[i].item()]-cluster_means[imposter_idx[j].item()])**2,axis=1)/emb_size
                denom = denom +  torch.clamp(torch.exp(var_norm*denom_cluster),min=1e-15,max=1e15)

        implied_los = torch.tensor(0.0).to(device)
        implied_numerator = torch.tensor(0.0).to(device)
        if cluster_classes[imposter_idx[i].item()] ==2:
            #implied_los = torch.sum((cluster_dict[imposter_idx[i].item()]-implied_cluster_dict[imposter_idx[i].item()-6])**2,axis=1)/emb_size
            implied_los = torch.sum((cluster_dict[imposter_idx[i].item()]-implied_imp_means[imposter_idx[i].item()-6])**2,axis=1)/emb_size
            current_mean1 = cluster_means[imposter_idx[i].item()]
            implied_numerator = torch.clamp(torch.exp(var_norm_implied*implied_los-alpha),min=1e-15,max=1e15)
            #current_mean1 = implied_imp_means[imposter_idx[i].item()-6]
        else:
            current_mean1 = cluster_means[imposter_idx[i].item()]
        #implied_numerator = torch.clamp(torch.exp(var_norm_implied*implied_los-alpha),min=1e-15,max=1e15)
        num_cluster = torch.sum((cluster_dict[imposter_idx[i].item()]-current_mean1)**2,axis=1)/emb_size
        if experiment_type == "ace_add_foc":
            numerator = torch.clamp(torch.exp(var_norm*num_cluster-alpha),min=1e-15,max=1e15)
        if experiment_type == "ace_add_inf_foc":
            numerator = torch.clamp(torch.exp(var_norm*num_cluster-alpha),min=1e-15,max=1e15) + implied_numerator # implied cost
        reduce_dist += torch.mean(implied_numerator)        
        mask = (numerator == 0.0).int()# log of 0 is inf
        numerator = numerator + mask*epsilon
        denom = denom *(1-mask) +mask*epsilon

        mask2 = (denom == 0.0).int()
        denom = denom + mask2 *epsilon
        numerator = numerator *(1-mask2) + mask2*epsilon
        gama = 2

        losses_before_log = numerator / denom

        instance_losses = torch.nn.functional.relu(torch.clamp(torch.round(((1-losses_before_log)* (10**2))) / (10**2),min=0,max=1)**gama * (-torch.log(losses_before_log)))

        try:
            if (imposter_idx[i].item()>=6):
                if len(list(cluster_index[imposter_idx[i].item()].size())) ==0:
                    if model.training:
                        train_instance_losses[idx2[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                    else:
                        val_instance_losses[idx2[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                else:
                    for b in range(instance_losses.size(dim=0)):
                        if model.training:
                            train_instance_losses[idx2[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
                        else:
                            val_instance_losses[idx2[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
            elif (imposter_idx[i].item()>=3):
                if len(list(cluster_index[imposter_idx[i].item()].size())) ==0:
                    if model.training:
                        train_instance_losses[idx1[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                    else:
                        val_instance_losses[idx1[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                else:
                    for b in range(instance_losses.size(dim=0)):
                        if model.training:
                            train_instance_losses[idx1[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
                        else:
                            val_instance_losses[idx1[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
            else:
                if len(list(cluster_index[imposter_idx[i].item()].size())) ==0:
                    if model.training:
                        train_instance_losses[idx0[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                    else:
                        val_instance_losses[idx0[cluster_index[imposter_idx[i].item()].item()].item()] = instance_losses[b].item()
                else:
                    for b in range(instance_losses.size(dim=0)):
                        if model.training:
                            train_instance_losses[idx0[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
                        else:
                            val_instance_losses[idx0[cluster_index[imposter_idx[i].item()][b].item()].item()] = instance_losses[b].item()
        except:
            pass
        magnet_loss = magnet_loss + torch.nn.functional.relu(torch.sum(instance_losses)/(cluster_dict[imposter_idx[i].item()].size(dim=0) + epsilon))

    magnet_loss = magnet_loss / imposter_idx.size(dim=0)

    return magnet_loss , reduce_dist

torch.autograd.set_detect_anomaly(True)

#____________________________________ Training Step _________________________________

import pickle
def train(model, iterator, optimizer, criterion,epoch):
    loss1 = torch.tensor(0)
    loss2= torch.tensor(0)
    loss_implied = torch.tensor(0)
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
        
    optimizer.zero_grad()
        
    predictions,out = model(tdata)
    loss1 = criterion(predictions.view(-1, 3), tlabels.view(-1))

    if experiment_type == "ace_add_foc" or experiment_type == "ace_add_inf_foc":    
        loss2, loss_implied = m_loss(model,out,epoch)
        implied_loss = loss_implied.item()

    loss = loss1+loss2 

    acc = categorical_accuracy(predictions, tlabels)
    #acc = magnet_accuracy(out,tlabels)
    loss.backward()
    
    optimizer.step()
    scheduler.step()
    mg_loss = loss2.item()
    
    epoch_loss += loss1.item()
    #epoch_acc += acc.item()
    epoch_acc = acc

    return epoch_loss, epoch_acc, mg_loss, implied_loss 
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)

#____________________________________ Evaluation Step _________________________________
def evaluate(model, iterator, criterion,epoch):
    loss1 = torch.tensor(0)
    loss2 = torch.tensor(0)
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():

        predictions,out = model(vdata)
        loss1 = criterion(predictions.view(-1, 3), vlabels.view(-1))       

        if experiment_type == "ace_add_foc" or experiment_type == "ace_add_inf_foc":    
            loss2,_ = m_loss(model,out,epoch)
        loss = loss1 + loss2
        #acc = magnet_accuracy(out,vlabels)
        acc = categorical_accuracy(predictions, vlabels)
        mg_loss = loss2.item()
        epoch_loss += loss1.item()
        #epoch_acc += acc.item()
        epoch_acc = acc
    return epoch_loss, epoch_acc, mg_loss
    #return epoch_loss / len(iterator), epoch_acc / len(iterator)


#____________________________________ Main _________________________________
fi = open(out_file_n,"w")
train_iterator = torch.tensor(0)
valid_iterator = torch.tensor(0)
fi.write("ep,TCLOSS,TMLOSS,TNOFFR,TEXPR,TIMPR,TMACROA,TMICROA,TF1NOFF,TF1EXP,TF1IMP,TF1MACRO,VCLOSS,VMLOSS,VNOFFR,VEXPR,VIMPR,VMACROA,VMICROA,VF1NOFF,VF1EXP,VF1IMP,VF1MACRO,VNOFFR,VEXPR,VIMPR"+"\n")
max_val_macro_f1 = 0.0
for epoch in range(N_EPOCHS):

    train_loss, train_acc, train_mloss, implied_loss = train(model, train_iterator, optimizer, criterion,epoch)
    valid_loss, valid_acc, valid_mloss = evaluate(model, valid_iterator, criterion,epoch)
    if max_val_macro_f1<float(valid_acc.split(",")[6]):
        torch.save(model.state_dict(),model_saving_path)
        max_val_macro_f1 = float(valid_acc.split(",")[6])
        print("saving model- best f1:",max_val_macro_f1)

    fi.write(str(epoch)+','+str(train_loss)+","+str(train_mloss)+","+str(train_acc)+str(valid_loss)+","+str(valid_mloss)+","+str(valid_acc)+","+str(implied_loss)+"\n")
    #print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')
    print(f'| Epoch: {epoch+1:02} | Train CLoss: {train_loss:.3f}  | Train MLoss: {train_mloss:.3f}| Train Acc: {train_acc} | Val. CLoss: {valid_loss:.3f} | Val. Acc: {valid_acc} |')
fi.close()
#test_loss, test_acc = evaluate(model, test_iterator, criterion)


