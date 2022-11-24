from arr_model import gen_arrive
import arr_train
import pandas as pd
import common
from emb.model import Emb_loc
df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df)

emb_loc = Emb_loc(loc_emb_size=16)
model = gen_arrive(loc_emb_dim=16,tim_emb_dim=10,pos_emb_dim=10,emb_loc=emb_loc)
arr_train.pretrain_gen_arrive(model=model, traject=traject, lr=1e-2, batch_size=256, Epoch=100)
