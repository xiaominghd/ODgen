import math
import disCrinmator.tim_dis_run
import disCrinmator.loc_dis_run
from disCrinmator.loc_dis import LocDis
from disCrinmator.tim_dis import TimDis
from emb.model import Emb_loc
import OD_dataloader
import disCrinmator.dis_dataloader as dis_dataloader
from locGen import loc_model, loc_dataloader
from OD_model import ODgen
import pandas as pd
import common
import torch
import torch.optim as optim
from roll_out import roll
from locGen import loc_model
from durraGen import dur_model
from arriveGen import arr_model

# 训练的参数
EPOCH = 100
BATCH_SIZE = 256
GEN_STEP = 3
DIS_STEP = 1
SAMPLE_NUM = 4000

emb_loc = Emb_loc(loc_emb_size=16)

duration = dur_model.gen_duration(loc_emb_dim=16, tim_emb_dim=10, point_size=2500, emb_loc=emb_loc)
duration.load_state_dict(torch.load('../pretrain/gen_dura_20.pth'))

loc = loc_model.gen_loc(emb_loc=emb_loc, emb_tim=None, loc_embedding_dim=16, time_embedding_dim=5, hidden_dim=32,
                        point_size=2500)
loc.load_state_dict(torch.load('../pretrain/gen_loc_20.pth'))

arrive = arr_model.gen_arrive(loc_emb_dim=16, tim_emb_dim=10, pos_emb_dim=10, emb_loc=emb_loc)
arrive.load_state_dict(torch.load('../pretrain/gen_arrive_20.pth'))

gen = ODgen(gen_loc=loc, gen_arrive=arrive, gen_duration=duration, hidden_dim=32, condition_num=4)

loc_dis = LocDis(emb_loc=emb_loc)
loc_dis.load_state_dict(torch.load("../pretrain/loc_dis_20.pth"))

tim_dis = TimDis(tim_embedding_dim=10, hidden_dim=32)
tim_dis.load_state_dict(torch.load("../pretrain/tim_dis_40.pth"))
# 优化器的选择
opt = optim.Adam(gen.parameters(), lr=4e-3)
# 读取数据集
df = pd.read_csv('../data/haikou_8.csv')
traject = common.choice(df)
data, inp, target = loc_dataloader.prepare_pretrain_loc(traject)

batch_num = math.floor(len(inp) / BATCH_SIZE)
inp1 = dis_dataloader.discriminator_traject(traject)


OD_dataloader.data2csv(gen, inp[:SAMPLE_NUM, 0, :].view(SAMPLE_NUM, -1, 3), '../data/gen_gan_0.csv')

crite = roll(model=gen, update_rate=0.8)
print('\n 开始对抗训练\n')
for epoch in range(EPOCH):

    for batch in range(batch_num):
        print('epoch : {}/200'.format(epoch + 1))
        print('batch : {}/{}'.format(batch, batch_num))

        for step in range(GEN_STEP):
            data = inp[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE, 0, :].view(BATCH_SIZE, -1, 3)

            s = gen.sample(data)

            gan_inp, gan_target = OD_dataloader.prepare_genGAN_batch(s)

            loss1 = gen.trainPGLoss_loc(gan_inp, gan_target.type(torch.float32), crite=crite, num_sample=8, dis=loc_dis)
            loss2 = gen.trainPGLoss_loc(gan_inp, gan_target.type(torch.float32), crite=crite, num_sample=8, dis=tim_dis)

            opt.zero_grad()
            loss1.backward()
            loss2.backward()
            opt.step()
            print("地点判别器的loss为:{}".format(loss1))
            print("时间判别器的loss为:{}".format(loss2))

        crite.update_params()

        OD_dataloader.data2csv(gen, inp[:SAMPLE_NUM, 0, :].view(SAMPLE_NUM, -1, 3), '../data/train.csv')
        df1 = pd.read_csv('../data/train.csv')

        traject1 = common.choice(df1, num=SAMPLE_NUM)
        neg = dis_dataloader.discriminator_traject(traject1)

        disCrinmator.loc_dis_run.pretrain_discrimintor(loc_dis, inp1[0:int(SAMPLE_NUM/2)], neg, lr=1e-3,
                                                       batch_size=16,
                                                       Epoch=DIS_STEP)  # 对抗训练判别器的过程
        disCrinmator.tim_dis_run.pretrain_tim_dis(tim_dis, inp1[0:int(SAMPLE_NUM/2)], neg, lr=1e-3,
                                                  batch_size=16, Epoch=DIS_STEP)

        OD_dataloader.data2csv(gen, inp[:SAMPLE_NUM, 0, :].view(SAMPLE_NUM, -1, 3),
                               '../data/gen_gan_{}.csv'.format(batch + 1))

    if epoch % 5 == 0:  # 每5个epoch保存一次模型

        torch.save(gen.state_dict(), '../pretrain/gen_gan_{}_train.pth'.format(epoch))
