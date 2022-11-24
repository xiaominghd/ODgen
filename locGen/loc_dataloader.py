
import torch

def prepare_pretrain_loc(data):
    data_len = len(data)

    inp = torch.zeros(data_len, 10, 3).type(torch.long)
    target = torch.zeros(data_len, 10, 3).type(torch.long)

    for i in range(len(data)):

        for j in range(len(data[i]) - 1):
            inp[i, j, :] = torch.tensor(data[i][j])

            target[i, j, :] = torch.tensor(data[i][j + 1])

    return data,inp, target