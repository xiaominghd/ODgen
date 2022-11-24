import numpy as np
import scipy.stats
from common import dist


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


class JSD_Metrix(object):

    def __init__(self, inp, target):

        self.inp = inp
        self.target = target

    def get_JSD_trajlen(self):

        p = np.zeros([11])
        q = np.zeros([11])

        for i in range(len(self.inp)):
            p[len(self.inp[i])] += 1

        for i in range(len(self.target)):
            q[len(self.target[i])] += 1

        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_topk(self):

        p = np.zeros([2500])
        q = np.zeros([2500])

        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[int(self.inp[i][j][0])] += 1

        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[int(self.target[i][j][0])] += 1

        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_duration(self):

        p = np.zeros([24])
        q = np.zeros([24])

        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[self.inp[i][j][2]] += 1

        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[self.target[i][j][2]] += 1

        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_start(self):
        p = np.zeros([24])
        q = np.zeros([24])
        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[self.inp[i][j][1]] += 1
        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[self.target[i][j][1]] += 1
        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)

    def get_JSD_end(self):
        p = np.zeros([24])
        q = np.zeros([24])
        for i in range(len(self.inp)):
            for j in range(len(self.inp[i])):
                p[int((self.inp[i][j][1] + self.inp[i][j][2]) / 2)] += 1
        for i in range(len(self.target)):
            for j in range(len(self.target[i])):
                q[int((self.target[i][j][1] + self.target[i][j][2]) / 2)] += 1
        p = np.array(p / sum(p))
        q = np.array(q / sum(q))

        return JS_divergence(p, q)



