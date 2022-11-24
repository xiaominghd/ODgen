import torch
import math
import torch.optim as optim
import dur_dataloader


def pretrain_gen_duration(model, traject, lr, batch_size, Epoch):
    opt = optim.Adam(model.parameters(), lr=lr)

    train = traject[:int(len(traject) * 0.9)]
    test = traject[int(len(traject) * 0.9):]

    inp_train, target_train = dur_dataloader.prepare_pretrain_duration(train)
    inp_test, target_test = dur_dataloader.prepare_pretrain_duration(test)

    batch_num = math.floor(len(inp_train) / batch_size)

    for epoch in range(Epoch):

        mid_loss = 0
        for i in range(1, batch_num):
            opt.zero_grad()

            loss = model.pretrain(inp_train[i * batch_size:i * batch_size + batch_size],
                                  target_train[i * batch_size:i * batch_size + batch_size])

            loss.backward()
            opt.step()
            mid_loss += loss

        print("第{}次训练".format(epoch))
        print("停留时间生成器的loss为:{}".format(mid_loss / (len(train))))
        print("停留时间生成器在测试集上的loss为:{}".format(model.pretrain(inp_test, target_test) / len(test)))

        print("*-------------------------*")

        if (epoch % 20 == 0):
            torch.save(model.state_dict(), '../pretrain/gen_dura_' + str(epoch) + '.pth')

    print("停留时间预训练完成")

    return model
