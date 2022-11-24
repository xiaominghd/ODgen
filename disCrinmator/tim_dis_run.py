import math
import torch
import torch.autograd
import torch.optim as opt

from disCrinmator import dis_dataloader


def pretrain_tim_dis(model, pos_sample, neg_sample, Epoch, lr, batch_size):

    inp, target = dis_dataloader.prepare_discriminator(pos_sample, neg_sample)
    inp_train, inp_test = inp[:int(len(inp) * 0.8)], inp[int(len(inp) * 0.8):]
    target_train, target_test = target[:int(len(target) * 0.8)], target[int(len(target) * 0.8):]

    optim = opt.Adam(model.parameters(), lr=lr)
    batch_num = math.floor(len(inp_train) / batch_size)
    for epoch in range(Epoch):
        mid_loss = 0
        for batch in range(batch_num):

            loss = model.pretrain(inp_train[batch * batch_size:batch * batch_size + batch_size],
                                  target_train[batch * batch_size:batch * batch_size + batch_size])
            loss.backward()
            optim.step()
            mid_loss += loss
        loss1 = model.pretrain(inp_test, target_test)

        inp_pred = torch.where(model.Batchclassify(inp_test) > 0.5, 1, 0)

        m = 0
        for i in range(len(inp_test)):
            if inp_pred[i] == target_test[i][0]:
                m += 1

        print("-----------------")
        print("第{}次训练".format(epoch + 1))
        print("在训练集上的loss为:{}".format(mid_loss / (batch_num * batch_size)))
        print("在测试集上的loss为:{}".format(loss1 / len(inp_test)))
        print("在测试集上的准确率为:{}".format(m / len(inp_test)))
        print("-----------------")
        """""""""
        if epoch % 20 == 0:
            torch.save(model.state_dict(), '../pretrain/tim_dis_{}.pth'.format(epoch))
        """""""""


"""""""""
model = TimDis(tim_embedding_dim=10, hidden_dim=32)
df1 = pd.read_csv('../data/haikou_8.csv')
traject1 = common.choice(df1, num=4000)
pos = dis_dataloader.discriminator_traject(traject1)

df2 = pd.read_csv('../data/gen_gan_0.csv')
traject2 = common.choice(df2, num=4000)
neg = dis_dataloader.discriminator_traject(traject2)

inp, target = dis_dataloader.prepare_discriminator(pos_sample=pos, neg_sample=neg)
pretrain_tim_dis(model, inp=inp, target=target, lr=1e-3, batch_size=256, Epoch=50)
"""""""""
