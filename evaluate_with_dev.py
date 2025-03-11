import json

import sys
from sklearn import preprocessing
import sys
import csv
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from midst_models.single_table_TabDDPM.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

sys.modules['midst_competition'] = sys.modules['midst_models']
sys.modules['midst_competition.single_table_ClavaDDPM'] = sys.modules['midst_models.single_table_TabDDPM']
sys.modules['midst_competition.single_table_ClavaDDPM.tab_ddpm'] = sys.modules['midst_models.single_table_TabDDPM.tab_ddpm']
sys.modules['midst_competition.single_table_ClavaDDPM.tab_ddpm.gaussian_multinomial_diffsuion'] = sys.modules['midst_models.single_table_TabDDPM.tab_ddpm.gaussian_multinomial_diffsuion']

from midst_models.single_table_TabDDPM.pipeline_modules import *

import os

import pickle

import joblib
import torch
import torch.nn as nn
import torch.utils.data

import pandas as pd

class ClassificationModel(nn.Module):
    def __init__(self, input_size):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.fc(x)

def calculate_acc(classifier,scaler, evl_data, save_dir):
    evl_data = scaler.transform(evl_data)
    evl_data = torch.tensor(evl_data).float()
    with torch.no_grad():
        outputs = classifier(evl_data).squeeze()
    prediction_file = os.path.join(save_dir,"prediction.csv")
    with open(prediction_file, mode="w", newline="") as file:  
        writer = csv.writer(file)
        for value in outputs.squeeze():
            writer.writerow([value.item()])
    print(f"save at {prediction_file}")
    return outputs

    

def reorder_columns(df, column_orders):
    
    for col in column_orders:
        if col not in df.columns:
            df[col] = None
    
    df = df[column_orders]
    return df

def prepare_dataset(base_dir, column_orders):
    challenge_with_id = pd.read_csv(os.path.join(base_dir, "challenge_with_id.csv"))
    # is_train = pd.read_csv(os.path.join(base_dir, "challenge_label.csv"))  # 只有 is_train 列
    
    # challenge_with_id["is_train"] = is_train["is_train"]
    challenge_set = challenge_with_id.drop(columns=["trans_id", "account_id"])
    
    challenge_set = reorder_columns(challenge_set,column_orders)
    # 转换 is_train 为张量
    # is_train = torch.tensor(is_train.values).T
    challenge_set = challenge_set.drop(columns=["placeholder"])

    return challenge_set


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
    base_dir = "./tabddpm_white_box/dev" 
    classifier = ClassificationModel(200)
    classifier.load_state_dict(torch.load("./best_classification_model.pth"))
    scaler = joblib.load("./scaler.pkl")
    classifier.eval()

    for i in range(31, 41):
        save_dir = os.path.join(base_dir, f"tabddpm_{i}")
        pretrained_model = load_pretrained(relation_order, save_dir)
        pretrained_model = pretrained_model[(None, 'trans')]
        test_set = prepare_dataset(save_dir, pretrained_model["column_orders"])
        pretrained_model["diffusion"].eval()
        pretrained_model["diffusion"]._denoise_fn.eval()
        label_encoder = None
        num_transform = pretrained_model["inverse_transform"].__self__
        
        for parent, child in relation_order:
            test_result, _ = get_model_gradient(
                test_set, parent, child, pretrained_model,num_transform,label_encoder
            )
        calculate_acc(classifier, scaler, torch.stack(test_result).detach().cpu().squeeze(), save_dir)

    for i in range(71, 81):
        save_dir = os.path.join(base_dir, f"tabddpm_{i}")
        pretrained_model = load_pretrained(relation_order, save_dir)
        pretrained_model = pretrained_model[(None, 'trans')]
        test_set = prepare_dataset(save_dir, pretrained_model["column_orders"])
        pretrained_model["diffusion"].eval()
        pretrained_model["diffusion"]._denoise_fn.eval()
        label_encoder = None
        num_transform = pretrained_model["inverse_transform"].__self__
        for parent, child in relation_order:

            test_result, _ = get_model_gradient(
                test_set, parent, child, pretrained_model,num_transform,label_encoder
            )
        calculate_acc(classifier, scaler, torch.stack(test_result).detach().cpu().squeeze(), save_dir)

    
    base_dir = "./tabddpm_white_box/final" 
    for i in range(41, 51):
        save_dir = os.path.join(base_dir, f"tabddpm_{i}")
        pretrained_model = load_pretrained(relation_order, save_dir)
        pretrained_model = pretrained_model[(None, 'trans')]
        test_set = prepare_dataset(save_dir, pretrained_model["column_orders"])
        pretrained_model["diffusion"].eval()
        pretrained_model["diffusion"]._denoise_fn.eval()
        label_encoder = None
        num_transform = pretrained_model["inverse_transform"].__self__
        for parent, child in relation_order:
            test_result, _ = get_model_gradient(
                test_set, parent, child, pretrained_model,num_transform,label_encoder
            )
        calculate_acc(classifier, scaler, torch.stack(test_result).detach().cpu().squeeze(), save_dir)
        
    for i in range(81, 91):
        save_dir = os.path.join(base_dir, f"tabddpm_{i}")
        pretrained_model = load_pretrained(relation_order, save_dir)
        pretrained_model = pretrained_model[(None, 'trans')]
        test_set = prepare_dataset(save_dir, pretrained_model["column_orders"])
        pretrained_model["diffusion"].eval()
        pretrained_model["diffusion"]._denoise_fn.eval()
        label_encoder = None
        num_transform = pretrained_model["inverse_transform"].__self__
        for parent, child in relation_order:

            test_result, _ = get_model_gradient(
                test_set, parent, child, pretrained_model,num_transform,label_encoder
            )
        calculate_acc(classifier, scaler, torch.stack(test_result).detach().cpu().squeeze(), save_dir)
        

    
    




if __name__ == "__main__":
    main()


