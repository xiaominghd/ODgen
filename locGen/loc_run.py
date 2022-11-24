from locGen import loc_model, loc_train

import pandas as pd
import common
from emb.model import Emb_loc

df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df)
emb_loc = Emb_loc(loc_emb_size=16)
loc = loc_model.gen_loc(emb_loc=emb_loc, emb_tim=None, loc_embedding_dim=16, time_embedding_dim=5, hidden_dim=32,
                        point_size=2500,dropout=0.5)
loc_train.pretrain_gen_loc(model=loc, traject=traject, Epoch=100, lr=1e-2, batch_size=256)
