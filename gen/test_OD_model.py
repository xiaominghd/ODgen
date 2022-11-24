import arriveGen
import disCrinmator.tim_dis
from locGen import loc_model, loc_dataloader
from durraGen.dur_model import gen_duration
from arriveGen.arr_model import gen_arrive
from OD_model import ODgen
from roll_out import roll
import pandas as pd
import common
import torch
from disCrinmator import loc_dis
from emb.model import Emb_loc


def prepare_data(file='../data/haikou_8.csv'):
    df = pd.read_csv(file)
    traject = common.choice(df, num=200)
    data, inp, target = loc_dataloader.prepare_pretrain_loc(traject)
    return data, inp, target


emb_loc = Emb_loc(loc_emb_size=16)

duration = gen_duration(loc_emb_dim=16, tim_emb_dim=10, point_size=2500, emb_loc=emb_loc)
duration.load_state_dict(torch.load('../pretrain/gen_dura_20.pth'))
loc = loc_model.gen_loc(emb_loc=emb_loc, emb_tim=None, loc_embedding_dim=16, time_embedding_dim=5, hidden_dim=32,
                        point_size=2500)
loc.load_state_dict(torch.load('../pretrain/gen_loc_20.pth'))
arrive = arriveGen.arr_model.gen_arrive(loc_emb_dim=16,tim_emb_dim=10,pos_emb_dim=10,emb_loc=emb_loc)
arrive.load_state_dict(torch.load('../pretrain/gen_arrive_20.pth'))

gen = ODgen(gen_loc=loc, gen_arrive=arrive, gen_duration=duration, hidden_dim=32, condition_num=4)
df = pd.read_csv('../data/haikou_8.csv')

dis = disCrinmator.tim_dis.TimDis(tim_embedding_dim=10,hidden_dim=32)
dis.load_state_dict(torch.load("../pretrain/tim_dis_40.pth"))
traject = common.choice(df,num=200)
data, inp, target = loc_dataloader.prepare_pretrain_loc(traject)
data = inp[:10, :2, :].view(10, -1, 3)
crite = roll(model=gen, update_rate=0.8)
hidden = gen.init_hidden()
samples = gen.sample(data)
loss=gen.trainPGLoss_tim(inp,target,crite=crite,num_sample=4,dis=dis)
print(loss)
