from dur_model import gen_duration
import dur_dataloader
import dur_train
import pandas as pd
import common
from emb.model import Emb_loc

df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df)
emb_loc = Emb_loc(loc_emb_size=16)

inp,target = dur_dataloader.prepare_pretrain_duration(traject)

model = gen_duration(loc_emb_dim=16,tim_emb_dim=10,point_size=2500,emb_loc=emb_loc)

dur_train.pretrain_gen_duration(model, traject, lr=1e-2, batch_size=256, Epoch=100)
