import numpy as np

from matplotlib import pyplot as plt


class KMeans:
    def __init__(self):
        pass

    def train(self, X, k, loop=10):
        # initialized
        np.random.shuffle(X)
        sample = X[0:k]
        classes = dict(((index, sample[index]) for index in range(0, k)))
        for i in range(0, k):
            h = list()
            h.append(classes[i])
            classes[i] = h

        for line in X[k:]:
            distMin = np.iinfo('int64').max
            for i in range(0, k):
                dist = self.dist(line, classes[i][0])
                if dist < distMin:
                    distMin = dist
                    minIndex = i
            classes[minIndex].append(line)

        # iterate
        iter = 0
        while iter < loop:
            classes_tmp = dict()
            for i in range(k):
                classes_tmp[i] = list()
                classes_tmp[i].append(np.array(reduce(lambda x, y: x + y, classes[i])) / len(classes[i]))
            for line in X:
                distMin = np.iinfo('int64').max
                for i in range(k):
                    dist = self.dist(line, classes_tmp[i][0])
                    if dist < distMin:
                        distMin = dist
                        minIndex = i
                classes_tmp[minIndex].append(line)
            classes = classes_tmp
            iter += 1
        return classes

    def dist(self, x, y):
        return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))

    def norm_rand(self):
        t = 4
        combis = [(1,1), (t,t),(1,t),(t,1)]
        len = 10
        kvs = list()
        x_ = list()
        y_ = list()
        for k,v in combis:
            xs = np.random.randn(1, len)+k
            ys = np.random.randn(1,len)+v
            x_ = x_+ list(xs)
            y_ = y_ + list(ys)
            for i in range(len):
                kvs.append(np.array((xs[0,i], ys[0,i])))
        return kvs


if __name__ == "__main__":
    km = KMeans()
    X = km.norm_rand()
    cls = km.train(X, 4)
    col = list(['r', 'b', 'y', 'g']).__iter__()

    for K, V in cls.items():
        xs = list()
        ys = list()
        for x, y in V:
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, col.next()+'o')
    plt.show()
