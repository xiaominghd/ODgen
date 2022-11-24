import torch
import torch.optim as optim
import math
import pandas as pd
import common
from disCrinmator import dis_dataloader, loc_dis
from emb.model import Emb_loc


def pretrain_discrimintor(model, pos_sample, neg_sample, lr, batch_size, Epoch):
    inp, target = dis_dataloader.prepare_discriminator(pos_sample, neg_sample)

    inp_train, target_train = inp[:int(len(inp) * 0.9)], target[:int(len(target) * 0.9)]
    inp_test, target_test = inp[int(len(inp) * 0.9):], target[int(len(target) * 0.9):]

    opt = optim.Adam(model.parameters(), lr=lr)

    batch_num = math.floor(len(inp_train) / batch_size)
    for epoch in range(Epoch):

        mid_loss = 0

        for i in range(batch_num):
            opt.zero_grad()
            loss = model.pretrain(inp_train[i * batch_size:i * batch_size + batch_size],
                                  target_train[i * batch_size:i * batch_size + batch_size])
            loss.backward()
            opt.step()
            mid_loss += loss

        inp_pred = torch.where(model.Batchclassify(inp_test) > 0.5, 1, 0)

        m = 0
        for i in range(len(inp_test)):
            if inp_pred[i] == target_test[i][0]:
                m += 1
        print("第{}次训练".format(epoch))
        print("在训练集上判别器的loss为:{}".format(mid_loss / len(inp_train)))
        print("在测试集上判别器的loss为:{}".format(model.pretrain(inp_test, target_test) / len(inp_test)))
        print("在测试集上的准确率为:{}".format(m / len(inp_test)))
        """""""""
        if epoch % 20 == 0:
            torch.save(model.state_dict(), '../pretrain/loc_dis_{}.pth'.format(epoch))
        """""""""

    print("判别器预训练完成")
    return model


"""""""""

df1 = pd.read_csv('../data/haikou_8.csv')
traject1 = common.choice(df1, num=4000)
pos = dis_dataloader.discriminator_traject(traject1)

df2 = pd.read_csv('../data/gen_gan_0.csv')
traject2 = common.choice(df2, num=4000)
neg = dis_dataloader.discriminator_traject(traject2)

emb_loc = Emb_loc(loc_emb_size=16)


model = loc_dis.LocDis(emb_loc=emb_loc)

pretrain_discrimintor(model, pos, neg, lr=1e-2, batch_size=256, Epoch=100)
"""""""""
