import torch.nn as nn
import torch

import common


class LocDis(nn.Module):

    def __init__(self, emb_loc=None, loc_embedding_dim=16, hidden_dim=32,dropout=0.5):
        super(LocDis, self).__init__()
        if emb_loc is not None:
            self.emb_loc = emb_loc
        else:
            self.emb_loc = nn.Embedding(2500, loc_embedding_dim)

        self.hidden_dim = hidden_dim
        self.loc_embedding_dim = loc_embedding_dim

        self.gru = nn.GRU(loc_embedding_dim, hidden_dim, num_layers=2, bidirectional=True,
                          dropout=dropout, batch_first=True)  # 判别器这里也是用的GRU

        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size=1):

        return torch.autograd.Variable(torch.zeros(4, batch_size, self.hidden_dim))  # 初始化隐状态

    def forward(self, x, hidden):

        traject = x[:common.act_len(x) - 1].permute(1, 0).contiguous()

        point = self.emb_loc(traject[0]).view(1, -1, self.loc_embedding_dim)

        out, hidden = self.gru(point, hidden)

        out = self.hidden2out(hidden.view(-1, 4 * self.hidden_dim)).view(-1) # batch_size x 1，输出结果，是batch_size*1的结果

        return out

    def pretrain(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn1 = nn.BCELoss()

        loss = 0

        for i in range(batch_size):
            hidden = self.init_hidden()

            out = self.forward(inp[i], hidden)

            loss1 = loss_fn1(out, target[i].type(torch.float32).view(-1))

            loss += loss1

        return loss

    def Batchclassify(self, inp):

        re = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):

            hidden = self.init_hidden()

            out = self.forward(inp[i], hidden)

            re[i] = out

        return re
