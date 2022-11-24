import torch
import torch.nn as nn
import gen.roll_out


class ODgen(nn.Module):

    def __init__(self, gen_loc: nn.Module, gen_arrive: nn.Module, gen_duration: nn.Module,
                 hidden_dim: int, condition_num: int) -> None:

        super(ODgen, self).__init__()

        self.gen_loc = gen_loc
        self.gen_arrive = gen_arrive
        self.gen_duration = gen_duration
        self.condition_num = condition_num

        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size: int = 1):

        return torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))  # 初始化隐状态

    def forward1(self, x: torch.Tensor, hidden: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """""""""
        预测地点
        
        输入：
        x :                  Tensor [[当前位置，到达时间，停留时间]]      1 * 3
        hidden :             Tensor [[[...]]]                       1 * 1 * hidden_dim
        
        输出：
        x :                  Tensor [...]                           point_size
        hidden :             Tensor [[[...]]]                       1 * 1 * hidden_dim
        
        """""""""

        x, hidden = self.gen_loc(x, hidden)

        return x.view(-1), hidden

    def forward2(self, inp) -> [torch.Tensor]:

        """""""""
        预测到达下一地点的时间
        
        输入：
        x :                     Tensor []                             1 * 3
        next :                  Tensor []                             1
        distance :              float                                 1
        hidden :                Tensor [[[...]]]                      1 * 1 * hidden_dim
        condition :             Tensor []                             1  
        
        输出：（注意这里的arrive是移动时间，而非真正的到达时间）
        arrive :                Tensor []                             1
        hidden :                Tensor []                             1 * 1 * hidden_dim
        """""""""

        arrive = self.gen_arrive(inp)

        return arrive

    def forward3(self, inp) -> [torch.Tensor]:

        duration = self.gen_duration(inp)

        return duration

    def sample(self, start_letter=None):

        num_samples = start_letter.size()[0]
        input_seqlen = start_letter.size()[1]

        samples = torch.zeros([num_samples, 10, 3]).type(torch.long)
        samples[:, :input_seqlen, :] = start_letter

        for i in range(num_samples):
            if input_seqlen == 1:
                hidden = self.init_hidden()
            else:
                hidden = self.init_hidden()
                for j in range(input_seqlen):
                    x, loc, hidden = self.forward(samples[i][j], hidden, torch.tensor(j))
            pos = torch.tensor(input_seqlen)
            mid = samples[i][input_seqlen - 1]
            t = input_seqlen
            while mid[1] + mid[2] < 24 and mid[0] != 0:
                mid, loc, hidden = self.forward(mid, hidden, pos)
                if mid[0] == 0:
                    samples[i, t] = torch.tensor([0, 0, 0])
                else:
                    samples[i, t] = mid
                t += 1

        return samples

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, pos) -> [torch.Tensor,
                                                                      torch.Tensor,
                                                                      torch.Tensor]:
        """""""""
        对单点进行采样 输入当前位置 生成下一个位置
        
        输入：
        x :                 Tensor [[当前位置，到达时间,停留时间]]                         1 * 3
        hidden :            Tensor [[[...]]]                                          1 * 1 * hidden_dim
        condition :         Tensor []                                                 1
        
        输出：
        x :                 Tensor [[下一位置，到达时间,停留时间]]                         1 * 3
        loc:                Tensor [...]                                              point_size
        hidden :            Tensor [[[...]]]                                          1 * 1 * hidden_dim
        """""""""

        x = x.type(torch.long)
        loc, hidden = self.forward1(x, hidden)

        point = self.gen_loc.get_loc(loc)

        arrive = self.forward2([x[0], point[0], x[1] + x[2], pos])
        start_tim = arrive
        arrive = self.gen_arrive.get_arrive(arrive)
        arrive = arrive + x[1] + x[2] + 2

        if arrive > 22:  # 如果到达的时间大于24点的话，那么就不生成了

            return torch.tensor([0, 24, 0]).type(torch.long), [loc, start_tim, torch.tensor(0)], hidden

        else:

            duration = self.forward3([point, arrive])
            dur_tim = duration
            duration = self.gen_duration.get_dur(duration)

        x = torch.cat([point, arrive, duration], dim=0)

        return x, [loc, start_tim, dur_tim], hidden,

    def trainPGLoss_loc(self, inp: torch.Tensor, target: torch.Tensor, crite: gen.roll_out.roll,
                        num_sample: int, dis) -> torch.Tensor:
        """""""""
        对模型进行对抗训练
        输入：
        inp :          Tensor [[[]]]                  batch_size * max_seq_len * 3
        target:        Tensor [[[]]]                  batch_size * max_seq_len * 3
        crite :        roll_out_object              
        num_sample :   int (4 , 8 , 16 , 32)
        
        输出：
        loss :        Tensor (.)
        
        """""""""
        loss1 = 0
        batch_size, seq_len, _ = inp.size()

        for i in range(batch_size):

            h = self.init_hidden()

            reward = crite.get_reward(inp[i], num_sample, discriminator=dis)

            for j in range(len(inp[i])):

                if target[i][j][0] != 0:
                    x, loc, h = self.forward(inp[i][j], h, torch.tensor(j))
                    loss1 = -loc[0][target[i][j][0].type(torch.long)] * reward[j]

        return loss1

    def trainPGLoss_tim(self, inp: torch.Tensor, target: torch.Tensor, crite: gen.roll_out.roll,
                        num_sample: int, dis) -> torch.Tensor:
        """""""""
        对模型进行对抗训练
        输入：
        inp :          Tensor [[[]]]                  batch_size * max_seq_len * 3
        target:        Tensor [[[]]]                  batch_size * max_seq_len * 3
        crite :        roll_out_object              
        num_sample :   int (4 , 8 , 16 , 32)

        输出：
        loss :        Tensor (.)

        """""""""
        start_loss = 0
        batch_size, seq_len, _ = inp.size()

        for i in range(batch_size):

            h = self.init_hidden()

            reward = crite.get_reward(inp[i], num_sample, discriminator=dis)

            for j in range(len(inp[i])):
                if target[i][j][0] != 0:
                    x, loc, h = self.forward(inp[i][j], h, torch.tensor(j))
                    start_loss += -loc[1][target[i][j][1].type(torch.long)] * reward[j]

        return start_loss
