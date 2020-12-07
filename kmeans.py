# -*- coding:UTF-8 -
from utility import dis
import numpy as np
import logging

class k_means:
    def __init__(self):
        #传入初始数据或初始数据的超参数
        self.dim = 2
        self.n = 100000
        self.k = 3
        self.points = np.random.random((self.n, self.dim))
    
        #k-means算法随机选取中心点
        self.centers = self.points[np.random.choice(np.arange(self.n), self.k),:]

    def train(self):
        self.iter = 1000
        for i in range(self.iter):
            index_array = np.argmin(dis(self.points, self.centers), axis=1)
            new_centers = np.zeros((self.k, self.dim))
            for j in range(self.k):
                if self.points[index_array == j,:].size != 0:
                    new_centers[j] = np.mean(self.points[index_array == j, :], axis=0)
                else:
                    new_centers[j] = self.centers[j]
            if np.array_equal(new_centers, self.centers):
                print("basic k-means converges at {} iterations".format(i))
                return
            self.centers = new_centers
        print("basic k-means hasn't converged at {} iterations".format(self.iter))

    

if __name__ == "__main__":
    basic_model = k_means()
    basic_model.train()