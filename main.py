import torch

import torch.nn as nn
import emb.model as emb
import os
import sys
dir_mytest = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, dir_mytest)
import locGen.loc_model as loc_model
emb_loc = torch.load('pretrain/loc_emb_raw.pt')
emb_poi = torch.load('pretrain/poi_emb_raw.pt')


class embconbine(nn.Module):

    def __init__(self, poi_emb, loc_emb):
        super(embconbine, self).__init__()
        self.poi_emb = poi_emb
        self.loc_emb = loc_emb

    def forward(self, x):
        poi = self.poi_emb(x)
        loc_emb = self.loc_emb.get_emb(x).reshape(x.size()[0], -1)

        return torch.cat([poi, loc_emb], dim=1)


point_emb = embconbine(poi_emb=emb_poi, loc_emb=emb_poi)
model=torch.load('pretrain/gen_loc_40.pt')
