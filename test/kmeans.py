# -*- coding:UTF-8 -
from utility import *
import numpy as np
import logging
import time

class kmeans_basic:
    def __init__(self, points, k, iter, weighted=False, weights=None, centers = None):
        self.n, self.dim = points.shape
        self.k = k
        self.points = points
        
        if centers is None:
            # 随机初始化
            self.centers = self.points[np.random.choice(np.arange(self.n), self.k),:]
        else:
            self.centers = centers

        self.weighted = weighted
        self.weights = weights
        self.iter = iter

    def cost(self):
        distances = dis(self.points, self.centers)
        return cost(distances)
    
    def time(self):
        return self.time

    def train(self):
        self.time = start = time.time_ns() 
        for i in range(self.iter):
            distances = dis(self.points, self.centers)
            cst = cost(distances)
            self.time = time.time_ns()
            print("iter {}, cost: {}, time: {}ns", i+1, cst, self.time - start)
            if self.weighted:
                distances = distances * self.weights[:, np.newaxis] # 每个点加权
            index_array = np.argmin(distances, axis=1)
            new_centers = np.zeros((self.k, self.dim))
            for j in range(self.k):
                if self.points[index_array == j,:].size != 0:
                    new_centers[j] = np.mean(self.points[index_array == j, :], axis=0)
                else:
                    new_centers[j] = self.centers[j]
            if np.array_equal(new_centers, self.centers):
                if self.weighted:
                    print("weighted k-means converges at {} iterations".format(i))
                else:
                    print("basic k-means converges at {} iterations".format(i))                    
                return self.centers
            self.centers = new_centers

        if self.weighted:
            print("weighted k-means hasn't converged at {} iterations".format(self.iter))
        else:
            print("basic k-means hasn't converged at {} iterations".format(self.iter))
        #return self.centers

class kmeans_plusplus:
    """
    k-means++ 主要的改进是，迭代的产生中心，每轮中心++
    新增的中心从数据点中挑，每次尽可能选离所有现有中心都比较远的数据点作为新的中心
    """
    def __init__(self, points, k, iter):
        self.n, self.dim = points.shape
        self.iter = iter
        self.k = k
        self.points = points
        
        # 初始仅有一个center
        self.centers = self.points[np.random.choice(np.arange(self.n), 1),:]
        
        # 初始分布
        distances = dis(self.points, self.centers)
        self.distribution = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1)) 
    
    def time(self):
        return self.time
    
    def cost(self):
        return self.cost

    def train(self):        
        # k-means++ 初始化，并不保证收敛，因此还要跑 k-means
        start = time.time_ns()
        for i in range(self.k - 1):
            next_center = self.points[np.random.choice(len(self.points), 1, p=self.distribution),:]
            self.centers = np.vstack((self.centers, next_center))

            # 更新分布，已经被选中的点是不会被再次选中的
            distances = dis(self.points, self.centers)
            self.distribution = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1)) 
            self.cost = cost(distances)
        self.time = time.time_ns() - start
    
        print("cost after init:{}, init time:{}".format(self.cost, self.time))
        kmeans = kmeans_basic(self.points, self.k, self.iterm, centers=self.centers)
        kmeans.train()
        self.time = kmeans.time()
        self.cost = kmeans.cost()
            

class kmeans_parallel:
        """
        k-means|| 每轮迭代增长 k 个中心， 增长 iter 次，最后再缩减到 k 个中心 
        有两种缩减的办法，但首先都要算出 weights ： 每个中心的 weights ， 即距离各个中心最近的点的数量的分布
        最后可以直接根据 weights 直接筛选出 k 个中心； 也可以随机选 k 个中心， 把 k * iter 作为 points 运行 k-means 算法
        但要注意这里的 k-means 算法算距离的时候要乘上 weights
        """
        def __init__(self, points, k, iter, weighted = False):
            self.n, self.dim = points.shape
            self.k = k
            self.iter = iter
            self.weighted = weighted
            self.points = points
            self.centers = self.points[np.random.choice(np.arange(self.n), self.k),:]
            # 初始分布
            distances = dis(self.points, self.centers)
            self.distribution = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1)) 

        def cost(self):
            distances = dis(self.points, self.centers)
            return cost(distances)
            
        def time(self):
            return self.time

        def get_weights(self):
            distances = dis(self.points, self.centers)
            counts = np.zeros(distances.shape)
            counts[range(counts.shape[0]), np.argmin(distances, axis=1)] = 1
            return np.sum(counts, axis=0)/ np.sum(np.sum(counts, axis=0))

        def train(self):
            """
            centers num : self.iter * self.k 
            """
            for i in range(self.iter - 1):
                next_centers = self.points[np.random.choice(len(self.points), self.k, p=self.distribution), :]
                self.centers = np.vstack((self.centers, next_centers))
                # 更新分布，已经被选中的点是不会被再次选中的
                distances = dis(self.points, self.centers)
                self.distribution = np.min(distances, axis=1) / np.sum(np.min(distances, axis=1)) 

            weights = self.get_weights()
            if not self.weighted:
                self.centers = self.centers[np.random.choice(len(self.centers), self.k, p=weights), :]
            else:
                # iter 可以取很大不用担心性能问题，因为可以提前收敛
                kmeans_weighted = kmeans_basic( points=self.centers, k=self.k, iter=self.iter * 1000, weighted = True, weights=weights)
                kmeans_weighted.train()
                self.time = kmeans_weighted.time()
                self.centers =  kmeans_weighted.centers

            


