import common
import torch


def prepare_pretrain_duration(data):
    data_len = len(data)

    inp = torch.zeros(data_len, 10, 2)
    target = torch.zeros(data_len, 10, 1)

    for i in range(len(data)):

        for j in range(len(data[i])):
            inp[i, j, ] = torch.tensor([data[i][j][0], data[i][j][1]]).type(torch.long)
            target[i, j, ] = torch.tensor([data[i][j][2]]).type(torch.long)

    return inp, target
