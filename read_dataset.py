import datasets
import csv
import pandas as pd
import preprocessor as p
import pickle
import numpy as np
import random
import sys
import argparse
from sklearn.model_selection import train_test_split
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


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

def read_abuse(ifHateBert):

    train_label = pd.read_csv('abuse/train.tsv', sep='\t')
    val_label = pd.read_csv('abuse/test.tsv', sep='\t')

    full_train = pd.read_csv('abuse/training.tsv', sep='\t')
    full_test = pd.read_csv('abuse/testing.tsv', sep='\t')

    labels_full = train_label["abuse"].append(val_label["abuse"], ignore_index=True).to_list()
    lables_cat = []
    for k in range(len(labels_full)):
        if labels_full[k] == "NOTABU":
            lables_cat.append(0)
        elif labels_full[k] == "EXP":
            lables_cat.append(1)
        elif labels_full[k] == "IMP":
            lables_cat.append(2)

    if ifHateBert:
        pre_ext_emb_full = "abuse/hateBERT_abusiveEval.pickle"
        pre_ext_emb_implied = "abuse/hateBERT_abuse_imp_implied.pickle"
    else:
        pre_ext_emb_full = "abuse/BERT_abusiveEval.pickle"
        pre_ext_emb_implied = "abuse/BERT_abuse_imp_implied.pickle"
    
    #pre_extracted_embeddingspre_ext_emb_full"gab/BERT_gab.pickle","rb")
    embd = pickle.load(pickle_in)
    pickle_in.close()

    #pre_extracted_embeddings_for_implied
    pickle_in = open(pre_ext_emb_implied,"rb")
    implied_embd = pickle.load(pickle_in)
    pickle_in.close()


    if ifHateBert:
        embd =  np.float32(embd)
        implied_embd = np.float32(implied_embd)

    implicit_fulldata_indices = []
    for i in range(len(lables_cat)):
        if lables_cat[i]==2:
            implicit_fulldata_indices.append(i)

    indices = np.arange(len(lables_cat))
    train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val = train_test_split(embd, lables_cat, indices, test_size=0.2,stratify=lables_cat)

    implied_tr1 = []
    implicit_tr1 =[]
    for kk in range(len(indices_tr)):
        if lables_cat[indices_tr[kk]] ==2:
            get_ind = implicit_fulldata_indices.index(indices_tr[kk])
            #get_ind = indices_tr[kk]-14380
            implied_tr1.append(implied_embd[get_ind])
            implicit_tr1.append(embd[indices_tr[kk]])
    implied_tr1 = np.asarray(implied_tr1)
    implied_tr = torch.tensor(implied_tr1).to(device)
    implicit_tr1 = np.asarray(implicit_tr1)
    implicit_tr = torch.tensor(implicit_tr1).to(device)

    return train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr



def read_gab(ifHateBert):

    data = pd.read_csv('gab/gab_final_data_gen.csv', sep=',')
    labels_full = data["label"].to_list()
    lables_cat = []
    lables_cat = labels_full
    text = data["text"].to_list()

    if ifHateBert:
        pre_ext_emb_full = "gab/hateBERT_gab.pickle"
        pre_ext_emb_implied = "gab/hateBERT_gab_imp_implied.pickle"
    else:
        pre_ext_emb_full = "gab/BERT_gab.pickle"
        pre_ext_emb_implied = "gab/BERT_gab_imp_implied.pickle"
    #pre_extracted_embeddingspre_ext_emb_full"gab/BERT_gab.pickle","rb")
    embd = pickle.load(pickle_in)
    pickle_in.close()

    #pre_extracted_embeddings_for_implied
    pickle_in = open(pre_ext_emb_implied,"rb")
    implied_embd = pickle.load(pickle_in)
    pickle_in.close()

    implicit_fulldata_indices = [i if lables_cat[i]==2 for i in range(len(lables_cat))]

    indices = np.arange(len(lables_cat))
    train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val = train_test_split(embd, lables_cat, indices, test_size=0.2,stratify=lables_cat)

    implied_tr1 = []
    implicit_tr1 =[]
    for kk in range(len(indices_tr)):
        if lables_cat[indices_tr[kk]] ==2:
            get_ind = implicit_fulldata_indices.index(indices_tr[kk])
            #get_ind = indices_tr[kk]-14380
            implied_tr1.append(implied_embd[get_ind])
            implicit_tr1.append(embd[indices_tr[kk]])
    implied_tr1 = np.asarray(implied_tr1)
    implied_tr = torch.tensor(implied_tr1).to(device)
    implicit_tr1 = np.asarray(implicit_tr1)
    implicit_tr = torch.tensor(implicit_tr1).to(device)

    return train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr


def read_latent(ifHateBert):

    data = pd.read_csv('latent/latent_full_data.csv', sep=',')
    labels_full = data["label"].to_list()
    lables_cat = []
    lables_cat = labels_full
    text = data["text"].to_list()

    if ifHateBert:
        pre_ext_emb_full = "latent/hateBERT_latent.pickle"
        pre_ext_emb_implied = "latent/hateBERT_latent_imp_implied.pickle"
    else:
        pre_ext_emb_full = "latent/BERT_latent.pickle"
        pre_ext_emb_implied = "latent/BERT_latent_imp_implied.pickle"
    
    #pre_extracted_embeddingspre_ext_emb_full"gab/BERT_gab.pickle","rb")
    embd = pickle.load(pickle_in)
    pickle_in.close()

    #pre_extracted_embeddings_for_implied
    pickle_in = open(pre_ext_emb_implied,"rb")
    implied_embd = pickle.load(pickle_in)
    pickle_in.close()


    if ifHateBert:
        embd =  np.float32(embd)
        implied_embd = np.float32(implied_embd)

    indices = np.arange(len(lables_cat))
    train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val = train_test_split(embd, lables_cat, indices, test_size=0.2,stratify=lables_cat)

    implied_tr1 = []
    implicit_tr1 =[]
    for kk in range(len(indices_tr)):
        if lables_cat[indices_tr[kk]] ==2:
            get_ind = indices_tr[kk]-14380
            implied_tr1.append(implied_embd[get_ind])
            implicit_tr1.append(embd[indices_tr[kk]])
    implied_tr1 = np.asarray(implied_tr1)
    implied_tr = torch.tensor(implied_tr1).to(device)
    implicit_tr1 = np.asarray(implicit_tr1)
    implicit_tr = torch.tensor(implicit_tr1).to(device)


    return train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr



def get_dataset(name,seed, ifHateBert):
    set_seed(seed)
    max_length = 90

    if name=='abuse':
        train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr = read_abuse(ifHateBert)

    if name=='gab':
        train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr = read_gab(ifHateBert)

    if name=='latent':
        train_texts,valid_texts,train_labels,valid_labels, indices_tr, indices_val, embd, implied_embd,implied_tr1,implied_tr,implicit_tr1,implicit_tr = read_latent(ifHateBert)







