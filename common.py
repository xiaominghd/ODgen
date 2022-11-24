import math
import random
from geopy.distance import geodesic
import pandas as pd
import numpy as np
import torch


def choice(data, num=60000):
    """""""""
    在轨迹数据集中选择从早到晚的轨迹
    """""""""
    traject = []
    mid = []
    for i in range(num):

        if data.iloc[i]['id'] == data.iloc[i + 1]['id']:
            mid.append([data.iloc[i]['gps2id'], data.iloc[i]['start_hour'], data.iloc[i]['duration']])

        else:
            mid.append([data.iloc[i]['gps2id'], data.iloc[i]['start_hour'], data.iloc[i]['duration']])
            traject.append(mid)
            mid = []

    random.shuffle(traject)  # 生成随机索引

    return traject


def dist(i, j):
    lon1 = i / 50
    lat1 = i % 50
    lon2 = j / 50
    lat2 = j % 50
    re = np.sqrt((lon1 - lon2) * (lon1 - lon2) + (lat1 - lat2) * (lat1 - lat2))
    return re


def act_len(seq):
    m = 0
    for s in seq:
        m += 1
        if s[0] == 0:
            return m
    return 10


def sample_condition(p=np.array([0.15, 0.32, 0.47, 0.06])):
    np.random.seed(0)

    condition = np.random.choice([0, 1, 2, 3], p=p)

    return torch.tensor(condition).view(-1).type(torch.float32)


def gen_condition(inp):
    condition = torch.zeros(len(inp),1).type(torch.long)
    for i in range(len(inp)):
        traject=inp[i]
        dur = 0
        t_len = 0
        for t in traject:
            if t[0] == 0:
                break
            else:
                dur += t[2]
                t_len += 1
        if t_len == 0:
            return torch.tensor(2)
        av_dur = dur / t_len

        if av_dur >= 4 and t_len > 3:
            condition[i,:]=torch.tensor(0)
        if av_dur < 4 and t_len > 3:
            condition[i,:]=torch.tensor(1)
        if av_dur >= 4 and t_len < 4:
            condition[i,:]=torch.tensor(2)
        if av_dur < 4 and t_len < 4:
            condition[i,:]=torch.tensor(3)
    return condition
