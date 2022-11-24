import pandas as pd
import random
import numpy as np

data_file = 'data/gen_gan_9.csv'
df = pd.read_csv(data_file)


def gen(left, right, df):
    res = [0] * 24
    mid = 0
    for i in range(left, right):
        start_hour = df.iloc[i]['start_hour']
        end_hour = df.iloc[i]['duration']+start_hour
        loc = df.iloc[i]['gps2id']
        mid = loc

        for j in range(start_hour, end_hour):
            res[j] = loc
    for i in range(len(res)):
        if res[i] == 0:
            random_sample = random.sample(range(-5, 5), 1)[0]
            res[i] = mid + random_sample
    return res


res = []
left = 0
right = 0
while right < len(df) - 2:
    if df.iloc[right]['id'] != df.iloc[right + 1]['id']:
        data = gen(left, right + 1, df)
        res.append(data)
        left = right + 1

    right += 1
file = 'fake_data.data'
with open(file, 'w') as fout:
    for sample in res:
        string = ' '.join([str(s) for s in sample])
        fout.write('%s\n' % string)
