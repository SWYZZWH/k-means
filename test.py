from kmeans import *

if __name__ == "__main__":
    n = 100000
    dim = 2
    points = create_data_uniform(n, dim)
    k = 3
    iter = 5
    muti = 1000

    basic = kmeans_basic(points, k = k, iter = iter * muti)
    pp = kmeans_plusplus(points, k = k)
    par = kmeans_parallel(points, k = k, iter = iter)
    par_weighted = kmeans_parallel(points, k = k, iter = iter, weighted = True)

    basic.train()
    pp.train()
    par.train()
    par_weighted.train()

    # analyze results...
    # benchmarks...