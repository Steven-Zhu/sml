import numpy as np


class SGD:
    def __init__(self):
        self.movies = None
        self.user_features = dict()
        self.item_features = dict()
        self.lamb = None
        self.rank = None
        self.ratings = dict()

    def train(self, file_path, lamb=0.1, rank=6, step=0.1):
        self.lamb = lamb
        self.rank = rank
        self.movies = self.get_movies(file_path + "movies.dat")
        for id, name in self.movies.items():
            self.item_features[id] = np.matrix(np.random.rand(self.rank, 1))

        # get ratings and train
        out_file = open(file_path + "ra.result2", 'a')
        out_file.write("start. lamb:" + str(lamb )+ ",step="+str(step) + "\n")
        iter = 0
        for loop in range(5):
            rating_lines = self.get_ratings(file_path + "ra.train")
            while len(rating_lines) >0:
                rand_index = np.random.randint(0, len(rating_lines))
                line = rating_lines[rand_index]
                del rating_lines[rand_index]
            # for line in rating_lines:
                infos = line.split("::")
                # if infos[0] not in self.ratings:
                #     self.ratings[infos[0]] = dict()
                # self.ratings[infos[0]][infos[1]] = infos[2]
                self.sgd_train(infos[0], infos[1], infos[2], step)

                iter += 1
                if iter == 100000:
                    print("updated features")
                    print("user_features: ", infos[0], self.user_features[infos[0]])
                    print("item_features: ", infos[1], self.item_features[infos[1]])
                    iter = 0

            out_file.write("end of traing.RMSE = "  + str(self.rmse(file_path + "ra.test")) + "\n")
            out_file.flush()
        out_file.close()

    def sgd_train(self, user_id, item_id, rating, step=0.01):
        if user_id not in self.user_features:
            self.user_features[user_id] = np.matrix(np.random.rand(self.rank, 1))

        err = float(rating) - (self.user_features[user_id].T * self.item_features[item_id])[0, 0]
        # print("user:", self.user_features[user_id])
        # print("item:", self.item_features[item_id])
        # print("err:", err)

        if user_id == '249':
            print("user:", user_id, self.user_features[user_id])
        try:
            user_feature = self.user_features[user_id] \
                       + step * (err * self.item_features[item_id] - self.lamb * self.user_features[user_id])
        except RuntimeWarning, w:
            print("user_warn", w)
            print("user:", self.user_features[user_id])
            print("item:", self.item_features[item_id])
            print("err:", err)
            user_feature = np.matrix(np.ones((self.rank, 1)))
        try:
            item_feature = self.item_features[item_id] \
                       + step * (err * self.user_features[user_id] - self.lamb * self.item_features[item_id])
        except RuntimeWarning, w:
            print("item_warn", w)
            print("user:", self.user_features[user_id])
            print("item:", self.item_features[item_id])
            print("err:", err)
            item_feature = np.matrix(np.ones((self.rank, 1)))
        self.user_features[user_id] = user_feature
        self.item_features[item_id] = item_feature

    def rmse(self, file_path):
        rating_lines = self.get_ratings(file_path)
        sum_se = 0.0
        for line in rating_lines:
            infos = line.split("::")
            pred = (self.user_features[infos[0]].T * self.item_features[infos[1]])[0, 0]
            sum_se += (pred - float(infos[2])) ** 2
        return (sum_se / len(rating_lines)) ** .5

    def get_movies(self, file_path):
        file = open(file_path, 'r')
        movies = dict()
        lines = file.readlines()
        file.close()
        for line in lines:
            items = line.split("::")
            movies[items[0]] = items[1]

        return movies

    def get_ratings(self, file_path):
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()
        return lines


if __name__ == "__main__":

    for lamb in (0.1, ):
        for step in (0.05, ):
            R = SGD()
            print("start training:", "lamb:", lamb, "rank:", 8, "step:", step,)
            R.train("/opt/ml-10M100K/", lamb=lamb, rank=8, step=step)
