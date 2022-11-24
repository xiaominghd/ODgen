import torch
import pandas as pd
import common
import dis_dataloader
from disCrinmator.tim_dis import TimDis

df = pd.read_csv("../data/haikou_8.csv")
traject = common.choice(df, num=200)
df1 = pd.read_csv("../data/gen_gan_0.csv")
traject1 = common.choice(df1, num=200)
pos = dis_dataloader.discriminator_traject(traject)
neg = dis_dataloader.discriminator_traject(traject)
inp, target = dis_dataloader.prepare_discriminator(pos_sample=pos, neg_sample=neg)

model = TimDis(tim_embedding_dim=10,hidden_dim=32)
model.pretrain(inp,target)