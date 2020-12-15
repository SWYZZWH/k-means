from kmeans import *
from data import *

def benchmark(title, points, k, iter, dup):
    basic_cost_avg, pp_cost_avg, par_cost_avg = 0,0,0
    basic_time_avg, pp_time_avg, par_time_avg = 0,0,0

    for i in range(dup):
        basic = kmeans_basic(points, k, iter=iter)
        pp = kmeans_plusplus(points, k, iter)
        par_weighted = kmeans_parallel(points, k, iter//100, weighted=True)
        basic.train()
        pp.train()
        par_weighted.train()

        print("final cost\n random_init:{}, kmeans++_init:{}, kmeans||_init:{} \
            time\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}  \
        ".format(    
            basic.get_cost(),
            pp.get_cost(),
            par_weighted.get_cost(),
            basic.get_time(),
            pp.get_time(),
            par_weighted.get_time()
        ))
        basic_cost_avg += basic.get_cost() / dup
        pp_cost_avg += pp.get_cost() / dup
        par_cost_avg += par_weighted.get_cost() / dup
        basic_time_avg += basic.get_time() / dup
        pp_time_avg += pp.get_time() / dup
        par_time_avg += par_weighted.get_time() / dup

    print("{}\n on average\n final cost\n random_init:{}, kmeans++_init:{}, kmeans||_init:{} \
            time\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}  \
        ".format(    
            title, basic_cost_avg, pp_cost_avg, par_cost_avg,basic_time_avg, pp_time_avg, par_time_avg
    ))
    with open("data/result.data", "a") as f:
        f.write(title + "\n")
        f.write("{},{},{},{},{},{}\n".format(basic_cost_avg, pp_cost_avg, par_cost_avg,basic_time_avg, pp_time_avg, par_time_avg))

if __name__ == "__main__":
    iter = 1000
    with open("data/result.data", "w") as f:
        pass
        
    for k in [20, 50, 100]:
        spam = load_data_spam()
        benchmark("spam", spam, k, iter, 11)

    
    for R in [1, 10, 100]:
        for k in [20, 50, 100]: 
            guass = load_data_guass(R, k)
            benchmark("guass_{}_{}".format(R,k), guass, k, iter, 11)
    
    

    