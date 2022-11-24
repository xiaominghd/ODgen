import common
import pandas as pd
import dur_dataloader
from emb.model import Emb_loc
from dur_model import gen_duration
df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df,num=200)
emb_loc = Emb_loc(loc_emb_size=16)

inp,target = dur_dataloader.prepare_pretrain_duration(traject)

model = gen_duration(loc_emb_dim=16,tim_emb_dim=10,point_size=2500,emb_loc=emb_loc)
print(model(inp[0][0]))
