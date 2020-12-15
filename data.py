from typing import Iterator
import numpy as np
import matplotlib.pyplot as plt


def create_data_gauss(n = 10000, dim = 15):
    # 首先在15维的球形高斯分布数据集中随机选取k个中心点（高斯分布的中心为原点，方差 R \in \{1,10,100\})。
    # 接着，以k个中心点为高斯分布的中心，方差为1，生成数据点。k个高斯分布产生的数据点是等量的。总的数据点数 n = 10,000。
    R_list = [1, 10, 100]
    k_list = [20, 50, 100]
    
    points = np.empty((n, dim), float)
    for R in R_list:
        for k in k_list:
            print("R:{}, K:{}".format(R, k))
            # 由于是球形高斯分布，那么使用15个独立的高斯分布拼接
            centers = np.random.normal(0, R, (k, dim))
            points_num_per_center = n // k
            for i, center in enumerate(centers):
                points[i * points_num_per_center: (i+1) * points_num_per_center, :] \
                    = np.random.normal(center, 1, (points_num_per_center, dim))
            file = "data/guass_{}_{}.data".format(R, k)
            points.dump(file)
            #radius = np.sqrt((points**2).sum(axis=1))
            #points = points/radius
            # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            # ax.scatter(points[:,0],points[:,1],points[:,2])
            # ax.set_aspect('auto')
            # plt.show()

def load_data_guass(R, k):
    file = "data/guass_{}_{}.data".format(R, k)
    return np.load(file, allow_pickle=True)

def load_data_spam():
    spam = np.loadtxt("data/spambase.data", delimiter=",")
    return spam

if __name__ == "__main__":
    create_data_gauss()
    
        