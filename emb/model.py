import torch
import torch.nn as nn
import numpy as np


class Emb_loc(nn.Module):

    def __init__(self, loc_emb_size, poi_file='../data/poi.npy', point_size=2500):
        super(Emb_loc, self).__init__()
        self.loc_emb_size = loc_emb_size
        self.point_size = point_size
        self.poi = torch.from_numpy(np.load(poi_file)).type(torch.float32)
        self.emb_poi = nn.Embedding(11, loc_emb_size)
        self.emb_loc = nn.Embedding(point_size, loc_emb_size)
        self.alpha = 0.9

    def forward(self, x):
        x = x.type(torch.long)
        one_hot_label = torch.zeros(x.size()[0], self.point_size).scatter_(1, x.unsqueeze(1), 1)

        poi = torch.mm(one_hot_label, self.poi).type(torch.long)
        poi.require_grad = False

        poi = torch.exp(self.emb_poi(poi).permute(0, 2, 1))

        poi_all = 1 / torch.sum(poi, dim=2).view(-1, self.loc_emb_size, 1)

        poi_w = torch.mul(poi, poi_all)

        poi = poi * poi_w

        poi = torch.sum(poi, dim=2)

        loc = self.emb_loc(x)
        emb = loc * self.alpha + poi * (1 - self.alpha)
        return emb

