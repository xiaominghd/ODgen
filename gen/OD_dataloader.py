import pandas as pd
import torch
import gen.OD_model


def prepare_genGAN_batch(points: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
    """""""""
    将采样出来的轨迹转化为判别器的输入:
    
    输入：
    
    points :        Tensor [[[]]]                  num_samples * inp_seq_len * 3
    
    """""""""
    if points.dim() != 3:

        raise ValueError (f"输入的向量{points}维度不对")

    inp = torch.zeros(len(points), 10, 3)
    target = torch.zeros(len(points), 10, 3)

    for i in range(len(points)):

        if len(points[i]) > 1:

            for j in range(0, len(points[i]) - 1):

                inp[i, j,] = points[i][j]
                target[i, j,] = points[i][j + 1]  # 错位组成数据集

    return inp.type(torch.long), target.type(torch.long)


def data2csv(model: gen.OD_model.ODgen, inp: torch.Tensor, file: str) -> None:
    """""""""
    通过model，从inp开始对数据进行采样，并将采样之后的数据输出到file文件当中
    输入:

    inp :          Tensor[[[]]]           num_samples * inp_seq_len * 3

    """""""""
    samples = model.sample(inp).numpy()

    id = []
    grid = []
    start = []
    duration = []

    m = 0

    for s in samples:

        m += 1
        for i in range(len(s)):

            if s[i][0] == 0:

                break
            else:

                id.append(m)
                grid.append(s[i][0])
                start.append(s[i][1])
                duration.append(s[i][2])

    data = {'id': id, 'gps2id': grid, 'start_hour': start, 'duration': duration}

    df = pd.DataFrame(data)  # 转化为dataframe

    df.to_csv(file)
