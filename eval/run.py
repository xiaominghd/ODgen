import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import common
from locGen import loc_dataloader
from JSD import JSD_Metrix


def view(data_len):
    df = pd.read_csv('../data/haikou_8.csv')
    traject = common.choice(df)
    # inp,_=loc_dataloader.prepare_pretrain_loc(traject)

    duration = []
    traject_len = []
    topk = []
    start = []
    end = []
    for i in range(0, data_len):
        df1 = pd.read_csv('../data/gen_gan_{}.csv'.format(i))
        df1.columns = ['no','id', 'gps2id', 'start_hour', 'duration']
        traject1 = common.choice(df1, num=4000)
        m = JSD_Metrix(traject, traject1)
        traject_len.append(m.get_JSD_trajlen())
        topk.append(m.get_JSD_topk())
        duration.append(m.get_JSD_duration())
        start.append(m.get_JSD_start())
        end.append(m.get_JSD_end())
    print(traject_len)
    print(topk)
    print(duration)
    print(start)
    print(end)

    x = np.linspace(0, data_len, data_len)
    ax1 = plt.subplot(221)
    ax1.plot(x, np.array(duration), label='duration')
    ax2 = plt.subplot(222)
    ax2.plot(x, np.array(topk), label='visit_frequency')
    ax3 = plt.subplot(223)
    ax3.plot(x, np.array(traject_len), label='traject_len')
    ax4 = plt.subplot(224)
    ax4.plot(x, np.array(start), label='start')

    plt.legend()
    plt.show()


view(data_len=8)
