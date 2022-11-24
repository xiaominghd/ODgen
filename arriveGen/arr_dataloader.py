import torch
import common


def prepare_pretrain_arrive(data):
    data_len = len(data)

    inp = torch.zeros(data_len, 10, 4)
    target = torch.zeros(data_len, 10, 1)

    for i in range(len(data)):

        for j in range(len(data[i]) - 1):
            inp[i, j, :] = torch.LongTensor(
                [data[i][j][0], data[i][j + 1][0], data[i][j][1] + data[i][j][2], torch.tensor(j)])
            target[i, j, :] = torch.tensor([max(0, data[i][j + 1][1] - data[i][j][1] - data[i][j][2] - 2)])

    return inp, target
