
import pandas as pd
import common
from emb.model import Emb_loc
import loc_model,loc_train,loc_dataloader
df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df,num=200)
emb_loc = Emb_loc(loc_emb_size=16)
loc = loc_model.gen_loc(emb_loc=emb_loc, emb_tim=None, loc_embedding_dim=16, time_embedding_dim=5, hidden_dim=32,
                        point_size=2500)
data,inp,target=loc_dataloader.prepare_pretrain_loc(traject)

hidden=loc.init_hidden()
point,hidden=loc(inp[0][0],hidden)
print(point)