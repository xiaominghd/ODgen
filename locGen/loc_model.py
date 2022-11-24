import torch
import torch.nn as nn
import torch.nn.functional as F
from emb.model import Emb_loc


class gen_loc(nn.Module):
    """""""""
    地点生成器
    对于时间的处理 具体的某一个时刻
    先将其进行分桶 分到一个24维one-hot的向量 再将它拼接到point上面 作为时间信息
    但是对于时间间隔的处理
    """""""""

    def __init__(self, emb_loc: nn.Module, emb_tim: nn.Module, loc_embedding_dim: int, time_embedding_dim: int,
                 hidden_dim: int, point_size: int, dropout: float = 0.8) -> None:

        super(gen_loc, self).__init__()

        self.loc_embedding_dim = loc_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.point_size = point_size
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)

        if emb_loc is None:
            self.emb_loc = nn.Embedding(point_size, loc_embedding_dim)  # 对地点信息进行嵌入
        else:
            self.emb_loc = emb_loc  # 似乎直接用nn.embedding也没有关系？错了 因为在嵌入之前都是先给位置进行了编号 导致我挨得近的位置编号相同

        if emb_tim is None:
            self.emb_time = nn.Embedding(24, time_embedding_dim)
        else:
            self.emb_time = emb_tim

        self.gru = nn.GRU(self.loc_embedding_dim + self.time_embedding_dim, hidden_dim, dropout=0.8)  #
        # 由于加上了时间戳的信息，所以生成器的维度是两者相加
        self.layer_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.ODE = nn.Sequential(
            nn.Linear(hidden_dim + self.time_embedding_dim, 10),
            nn.Linear(10, self.hidden_dim),
            nn.ReLU()
        )  # 将时间间隔信息加入到hidden里面对其进行更新,这个是对模型提升比较明显的一点

        self.gru2loc = nn.Sequential(
            nn.Linear(self.hidden_dim, point_size)
        )

        # 输出位置，如果使用两层线性层进行输出的话参数太多反而不太好输出位置

    def init_hidden(self, batch_size=1):

        return torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))  # 初始化隐状态

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """""""""
        对单点进行生成 
        输入：
        x      : Tensor[[位置，到达时间，停留时间]] max_seq_len * 3 
        hidden : Tensor[[[0]]]                 1 * 1 * hidden_dim   
        
        输出：
        point  : Tensor[[下一位置，到达时间，停留时间]] max_seq_len * 3
        hidden : Tensor[[[0]]]                    1 * 1 * hidden_dim                
        
        """""""""

        loc = self.emb_loc(x[0].view(-1)).view(-1)

        tim = self.emb_time(x[1]).view(-1)  # 对时间信息进行嵌入

        point = torch.cat([loc, tim], dim=0).view(1, 1, -1)  # 位置点应该是位置和时间进行拼接之后的结果

        duration = self.emb_time(x[2]).view(1, 1, -1)

        hidden = torch.cat([hidden, duration], dim=2)

        hidden = self.ODE(hidden)  # 经过ODE模型更新隐状态

        point, hidden = self.gru(point, hidden)  # 经过GRU进行状态的更新

        point = point.view(-1, self.hidden_dim)

        point = self.layer_norm(point)

        point = self.gru2loc(point.view(-1, self.hidden_dim))

        point = F.log_softmax(point, dim=1)  # 得到下一个位置点的概率分布作为输出

        return point, hidden

    def pretrain(self, inp, target):  # 预训练，输入的是batch_size*seqlen*3

        batch_size, seqlen, _ = inp.shape
        loss_fn = nn.NLLLoss()

        loss = 0

        for i in range(batch_size):
            hidden = self.init_hidden()

            for j in range(seqlen):

                if inp[i][j][0] != torch.tensor(0):  # 当检测到第一个位置为0的时候，就代表位置到达了序列的末尾

                    point, hidden = self.forward(inp[i][j], hidden)  # 输出在每一个位置的概率

                    loss = loss + loss_fn(point.view(-1),
                                          target[i][j][0].type(torch.long))  # 其实这里应该是要让模型能够自己知道到了什么时候应该停下来

        return loss

    def get_loc(self, point):

        point = torch.multinomial(torch.exp(point.view(1, -1)), 1)  # 直接输出到exp之后，然后做一个softmax的输出

        return point.view(-1).data
