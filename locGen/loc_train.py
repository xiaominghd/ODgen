import torch
import math
import torch.optim as optim
from locGen import loc_dataloader


def pretrain_gen_loc(model, traject, Epoch, lr, batch_size):
    opt = optim.Adam(model.parameters(), lr=lr)

    train = traject[:int(len(traject) * 0.8)]
    test = traject[int(len(traject) * 0.8):]

    batch_num = math.floor(len(train) / batch_size)
    inp_data,inp_train, target_train = loc_dataloader.prepare_pretrain_loc(train)
    inp_data,inp_test, target_test = loc_dataloader.prepare_pretrain_loc(test)

    for epoch in range(Epoch):
        mid_loss = 0
        for i in range(batch_num):
            opt.zero_grad()

            loss = model.pretrain(inp_train[i * batch_size:i * batch_size + batch_size],
                                  target_train[i * batch_size:i * batch_size + batch_size])

            loss.backward()
            opt.step()
            mid_loss += loss

        print("第{}次训练后".format(epoch))
        print("地点生成器的loss为：{}".format(mid_loss / len(inp_train)))
        print("地点生成器在测试集上的loss为:{}".format(model.pretrain(inp_test, target_test) / len(inp_test)))
        print("*-------------------------*")
        """""""""
        for name, parms in model.named_parameters():
            print('-->name', name)
            print('-->para', parms)
            print('-->grad_requirs', parms.requires_grad)
            print('-->grad_value', parms.grad)
            print("===")
        """""""""

        if (epoch % 10 == 0):
            torch.save(model.state_dict(), '../pretrain/gen_loc_' + str(
                epoch) + '.pth')

    print("地点预训练完成")

    return model
