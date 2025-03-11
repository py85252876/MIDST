import json

import sys
import csv



from midst_models.single_table_TabDDPM.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

sys.modules['midst_competition'] = sys.modules['midst_models']
sys.modules['midst_competition.single_table_ClavaDDPM'] = sys.modules['midst_models.single_table_TabDDPM']
sys.modules['midst_competition.single_table_ClavaDDPM.tab_ddpm'] = sys.modules['midst_models.single_table_TabDDPM.tab_ddpm']
sys.modules['midst_competition.single_table_ClavaDDPM.tab_ddpm.gaussian_multinomial_diffsuion'] = sys.modules['midst_models.single_table_TabDDPM.tab_ddpm.gaussian_multinomial_diffsuion']

from midst_models.single_table_TabDDPM.pipeline_modules import *

import os

import pickle

import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data

        
    
def reorder_columns(df, column_orders):
    
    for col in column_orders:
        if col not in df.columns:
            df[col] = None
    
    df = df[column_orders]
    return df


def prepare_dataset(base_dir, column_orders):
    challenge_with_id = pd.read_csv(os.path.join(base_dir, "challenge_with_id.csv"))
    
    is_train = pd.read_csv(os.path.join(base_dir, "challenge_label.csv"))  
    
    challenge_with_id["is_train"] = is_train["is_train"]

    challenge_set = challenge_with_id.drop(columns=["trans_id", "account_id", "is_train"])
    challenge_set = reorder_columns(challenge_set,column_orders)
    # print(column_orders)
    challenge_set = challenge_set.drop(columns=["placeholder"])

    is_train = torch.tensor(is_train.values).T

    return challenge_set, is_train


def load_pretrained(relation_order, save_dir):
    models = {}
    for parent, child in relation_order:
        assert os.path.exists(
            os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl")
        )
        print(f"{parent} -> {child} checkpoint found, loading...")
        models[(parent, child)] = pickle.load(
            open(os.path.join(save_dir, f"{parent}_{child}_ckpt.pkl"), "rb")
        )

    return models

def main(relation_order = None, save_dir = None):
    relation_order = [[None, "trans"]]
    base_dir = "./tabddpm_white_box/train" 
    all_train_set = []
    all_is_train = []
    for i in range(1, 31):
        save_dir = os.path.join(base_dir, f"tabddpm_{i}")
        pretrained_model = load_pretrained(relation_order, save_dir)
        pretrained_model = pretrained_model[(None, 'trans')]
        pretrained_model["diffusion"].eval()
        pretrained_model["diffusion"]._denoise_fn.eval()
        num_transform = pretrained_model["dataset"].num_transform
        label_encoder = pd.read_pickle(os.path.join(save_dir, f"trans_label_encoders.pkl"))
        train_set, is_train = prepare_dataset(save_dir, pretrained_model["column_orders"])
        for parent, child in relation_order:
            train_result, _ = get_model_gradient(
                train_set, parent, child, pretrained_model,num_transform,label_encoder
            )
        all_train_set.append(torch.stack(train_result).squeeze())
        all_is_train.append(is_train)
        print(f"finished number {i}")
    torch.save(torch.cat(all_train_set).squeeze().detach().cpu(), "./train_data.pt")
    torch.save(torch.stack(all_is_train).contiguous().view(1, -1).squeeze(), "./train_label.pt")

    
    




if __name__ == "__main__":
    main()