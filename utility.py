import numpy as np

def dis(points, centers):
    """
    points  n * d
    centers k * d
    dis     n * k
    """
    
    # np.newaxis 会新增一个维度，其他维度：代表用原有维度填充
    # centers 会升维， 默认升第一个维度
    # 两个三位矩阵加减运算，会对齐/补全 补完后都为 n * k * d 矩阵 （复制补全）
    return np.sum( (points[:,np.newaxis,:] - centers) ** 2, axis=2)

def create_data_uniform(n, dim):
    """
    产生均匀分布的数据
    如果太大需要存进文件
    """
    points = np.random.random((n, dim))
    return points