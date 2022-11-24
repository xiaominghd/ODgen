import torch
import torch.nn as nn
import common


class TimDis(nn.Module):
    def __init__(self, tim_embedding_dim, hidden_dim, dropout=0.8, tim_emb=None):
        super(TimDis, self).__init__()
        self.tim_embedding_dim = tim_embedding_dim
        self.hidden_dim = hidden_dim
        if tim_emb is None:
            self.tim_emb = nn.Embedding(25, tim_embedding_dim)
        else:
            self.tim_emb = tim_emb
        self.GRU = nn.GRU(tim_embedding_dim * 2, hidden_dim, num_layers=2, bidirectional=True,
                          dropout=dropout, batch_first=True)
        self.gru2out = nn.Sequential(
            nn.Linear(2 * 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, batch_size=1):
        return torch.autograd.Variable(torch.zeros(4, batch_size, self.hidden_dim))  # 初始化隐状态

    def forward(self, x, hidden):
        traject = x[:common.act_len(x) - 1].view(-1, 3)

        time_series = torch.cat([self.tim_emb(traject[:,1]), self.tim_emb(traject[:,2])], dim=1).view(1, -1,
                                                                                                  self.tim_embedding_dim * 2)

        out, hidden = self.GRU(time_series, hidden)

        out = self.gru2out(hidden.view(-1, 4 * self.hidden_dim)).view(-1)

        return out

    def pretrain(self, inp, target):

        batch_size = inp.size()[0]
        loss_fn = nn.BCELoss()
        loss = 0

        for i in range(batch_size):

            hidden = self.init_hidden()
            out = self.forward(inp[i], hidden)

            loss += loss_fn(out, target[i].view(-1))

        return loss

    def Batchclassify(self, inp):

        re = torch.Tensor(inp.size()[0])

        for i in range(len(inp)):
            hidden = self.init_hidden()

            out = self.forward(inp[i], hidden)

            re[i] = out

        return re


"""""""""
model = TimDis(tim_embedding_dim=10, hidden_dim=32)
x = torch.tensor([[20, 0, 5], [15, 10, 24], [0, 0, 0], [0, 0, 0]])
hidden = model.init_hidden()
print(model(x, hidden))
"""""""""
