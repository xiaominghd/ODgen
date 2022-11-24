import torch

import arr_dataloader
import pandas as pd
import common
from arr_model import gen_arrive
from arr_predictor import arr_predictor
df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df, num=200)
inp, target = arr_dataloader.prepare_pretrain_arrive(traject)
model = gen_arrive(loc_emb_dim=16,tim_emb_dim=10,pos_emb_dim=10,emb_loc=None)
point = model(inp[0][0])
print(model.get_arrive(point))
