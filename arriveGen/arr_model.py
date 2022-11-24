import torch
import torch.nn as nn
import torch.nn.functional as F


class gen_arrive(nn.Module):

    def __init__(self, loc_emb_dim, tim_emb_dim, pos_emb_dim, emb_loc=None):
        super(gen_arrive, self).__init__()
        self.loc_emb_dim = loc_emb_dim
        self.pos_emb_dim = pos_emb_dim
        self.tim_emb_dim = tim_emb_dim
        if emb_loc is None:
            self.emb_loc = nn.Embedding(2500, loc_emb_dim)
        else:
            self.emb_loc = emb_loc
        self.emb_tim = nn.Embedding(24, tim_emb_dim)
        self.conbine = nn.Bilinear(loc_emb_dim, loc_emb_dim, loc_emb_dim)
        self.emb_pos = nn.Embedding(10, self.pos_emb_dim)
        self.LaNorm = nn.LayerNorm(self.pos_emb_dim + self.loc_emb_dim + self.tim_emb_dim)
        self.predict = nn.Sequential(
            nn.Linear(self.pos_emb_dim + self.loc_emb_dim + self.tim_emb_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 24)
        )

    def forward(self, inp):

        start_loc = self.emb_loc(inp[0].view(-1).type(torch.long)).view(-1)
        end_loc = self.emb_loc(inp[1].view(-1).type(torch.long)).view(-1)
        loc = self.conbine(start_loc, end_loc)

        start_time = self.emb_tim(inp[2].type(torch.long)).view(-1)
        pos = self.emb_pos(inp[3].type(torch.long)).view(-1)

        pos = torch.concat([start_time, pos, loc], dim=0)
        pos = self.LaNorm(pos)

        arr = self.predict(pos)

        arr = F.log_softmax(arr, dim=0)

        return arr

    def pretrain(self, inp, target):  # 预训练，输入的是inp，数据维度为batch_size*seq_len*4

        loss_fn1 = nn.NLLLoss()

        batch_size, seqlen, _ = inp.size()

        loss = 0
        for i in range(batch_size):

            for j in range(seqlen):

                if inp[i][j][0] != torch.tensor(0):  # 当碰到0位置时停止，就是说序列到达了结束

                    out = self.forward(inp[i][j])  # loss函数，使用的是MSELoss

                    loss1 = loss_fn1(out.view(1,-1), target[i][j].type(torch.long))

                    loss = loss + loss1

        return loss  # 返回一个inp维度的loss

    def get_arrive(self, point):

        point = torch.multinomial(torch.exp(point.view(1, -1)), 1)  # 直接输出到exp之后，然后做一个softmax的输出

        return point.view(-1).data
