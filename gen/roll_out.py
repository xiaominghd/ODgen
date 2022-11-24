import numpy
import torch
import numpy as np
import copy


class roll(object):

    def __init__(self, model, update_rate) -> None:

        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x: torch.Tensor, sample_num: int, discriminator) -> numpy.ndarray:
        """""""""
        计算当前决策的reward
        
        输入
        x:           Tensor [[位置一,到达时间一,停留时间一]]       max_seq_len*3
        sample_num : Int                                      4、8、16、32
        
        输出
        reward:      numpy.array([r1,r2,...,rn])              max_seq_len    
        
        """""""""

        if x.dim() != 2 or x.size()[1] != 3:
            raise ValueError(f"输入{x}的格式不对")

        rewards = []

        for s in range(sample_num):  # 循环采样sample_num次

            sample_reward = []

            for j in range(1, len(x)):

                if x[j][0] != 0:  # 如果没有到末尾，那么就当前状态进行采样直到到达末尾计算reward，作为当前步的reward

                    data = x[0:j, :].view(1, -1, 3)

                    samples = self.own_model.sample(data)

                    pred = discriminator.Batchclassify(samples).data[0].numpy()

                    sample_reward.append(pred)

                if x[j][0] == 0:
                    data = x.view(1, -1, 3).type(torch.long)

                    pred = discriminator.Batchclassify(data).data[0].numpy()

                    sample_reward.append(pred)

                    break

            rewards.append(sample_reward)

        rewards = np.sum(np.array(rewards), axis=0) / sample_num  # 在sample_num次采样后的平均

        if len(rewards) == 0:
            raise ValueError(f"输入的序列{x}不应该为空")

        # rewards_diff = np.concatenate((np.array([1]), rewards[:len(rewards) - 1]))

        return np.exp(rewards)

    def update_params(self):

        dic = {}
        for name,param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name,param in self.own_model.named_parameters():

            param.data = self.update_rate * param.data + (1-self.update_rate) * dic[name]

