import torch


def prepare_discriminator(pos_sample, neg_sample):
    inp = torch.cat([pos_sample, neg_sample], dim=0)

    target = torch.zeros(inp.size()[0], 1)
    target[:pos_sample.size()[0], 0] = 1
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    return inp, target


def discriminator_traject(traject):
    inp = torch.zeros(len(traject), 10, 3).type(torch.long)

    for i in range(len(traject)):
        inp[i, :len(traject[i]), :] = torch.tensor(traject[i])

    return inp
