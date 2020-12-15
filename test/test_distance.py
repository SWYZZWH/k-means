import numpy as np
from numpy.testing import assert_almost_equal
from .utility import dis as distance
from .utility import cost 
from .kmeans import kmeans_basic
from .kmeans import kmeans_plusplus, kmeans_parallel

def test_non_negativity():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)>= 0).all()
    
def test_coincidence_when_zero():
    u = np.zeros((3,4))
    v = np.zeros((5,4))
    assert (distance(u, v)==0).all()

def test_coincidence_when_not_zero():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)!= 0).any()

def test_symmetry():
    u = np.random.normal(size=(3,4))
    v = np.random.normal(size=(5,4))
    assert (distance(u, v)== distance(v, u).T).all()

def test_known1():
    u = np.array([[0,0],[1,1]])
    v = np.array([[0,0],[1,1]])
    dist = np.array([[0,2],[2,0]])
    assert_almost_equal(distance(u, v), dist)
    
def test_known2():
    u = np.array([[0,0,0],[1,1,1],[2,2,2]])
    v = np.array([[1,1,1],[2,2,2],[3,3,3]])
    dist = np.array([[3,12,27],[0,3,12],[3,0,3]])
    assert_almost_equal(distance(u, v), dist)

def test_non_negative():
    for i in range(10):
        data = np.random.normal(size=(5,4))
        c = data[np.random.choice(range(4),2),]
        dist = distance(data,c)
        assert cost(dist) >= 0

def test_non_negative():
    data = np.random.normal(size=(20,4))
    centroids = data[np.random.choice(range(4),4),]
    dist = distance(data,centroids)
    c = cost(dist)
    p = distribution(dist,c)
    assert (p>=0).all()
    
# def test_sum_to_one():
#     data = np.random.normal(size=(20,4))
#     centroids = data[np.random.choice(range(4),4),]
#     dist = distance(data,centroids)
#     c = cost(dist)
#     p = distribution(dist,c)
#     assert_almost_equal(np.sum(p),1)

def test_label():
    for i in range(10):
        data = np.random.normal(size=(50,2))
        k = 3
        centroids = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
        label = kmeans_basic(data, k, centroids)["Labels"]
        assert max(label) == k-1 and len(label)==data.shape[0]