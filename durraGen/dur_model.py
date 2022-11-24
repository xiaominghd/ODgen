import torch
import torch.nn as nn
import torch.nn.functional as F


class gen_duration(nn.Module):

    def __init__(self, loc_emb_dim, tim_emb_dim, point_size, emb_loc=None, emb_tim=None):
        super(gen_duration, self).__init__()
        if emb_loc is None:
            self.emb_loc = nn.Embedding(point_size, loc_emb_dim)
        else:
            self.emb_loc = emb_loc

        if emb_tim is None:
            self.emb_tim = nn.Embedding(24, tim_emb_dim)
        else:
            self.emb_tim = emb_tim

        self.loc_emb_dim = loc_emb_dim

        self.tim_emb_dim = tim_emb_dim

        self.layer_Norm = nn.LayerNorm(self.tim_emb_dim + self.loc_emb_dim, eps=1e-6)
        self.predict = nn.Sequential(
            nn.Linear(self.tim_emb_dim + self.loc_emb_dim, 15),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(15, 5),
            nn.ReLU(),
            nn.Linear(5, 24)
        )

    def forward(self, x):

        start_loc = self.emb_loc(x[0].view(-1)).view(-1)

        start_time = self.emb_tim(x[1].type(torch.long)).view(-1)

        point = torch.cat([start_time, start_loc], dim=0)
        point = self.layer_Norm(point)

        dur = self.predict(point)

        dur = F.log_softmax(dur, dim=0)

        return dur

    def pretrain(self, inp, target):  # 预训练，输入的是inp，数据维度为batch_size*seq_len*4

        loss_fn2 = nn.NLLLoss()
        batch_size, seqlen, _ = inp.size()

        loss = 0
        for i in range(batch_size):

            for j in range(seqlen):

                if inp[i][j][0] != torch.tensor(0):  # 当碰到0位置时停止，就是说序列到达了结束

                    out = self.forward(inp[i][j])  # loss函数，使用的是MSELoss
                    loss += loss_fn2(out.view(1, -1), target[i][j].view(-1).type(torch.long))

        return loss  # 返回一个inp维度的loss

    def get_dur(self, point):
        point = torch.multinomial(torch.exp(point.view(1, -1)), 1)  # 直接输出到exp之后，然后做一个softmax的输出

        return point.view(-1).data
