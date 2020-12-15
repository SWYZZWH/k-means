
from kmeans import *
from data import *

def benchmark(title, points, k, iter, dup):
    name_lst = ["basic", "pp", "par", "par2"]
    model_lst = [
        kmeans_basic(points, k, iter=iter), 
        kmeans_plusplus(points, k, iter), 
        kmeans_parallel(points, k,  k // 2 , 5, weighted=True ),
        kmeans_parallel(points, k, k * 2, 5, weighted=True)
    ]
    init_cost_lst = [0,0,0,0]
    final_cost_lst = [0,0,0,0]
    init_iter_lst = [0,0,0,0]
    iter_lst = [0,0,0,0]
    init_time_lst = [0,0,0,0]
    total_time_lst = [0,0,0,0]

    for i in range(dup):
        for j, model in enumerate(model_lst):
            model.train()
            init_cost_lst[j] += model.get_init_cost() / dup
            final_cost_lst[j] += model.get_final_cost() / dup
            init_iter_lst[j] += round(model.get_init_iter()) / dup
            iter_lst[j] += round(model.get_init_iter()) / dup
            init_time_lst[j] += model.get_init_time()/ dup
            total_time_lst[j] += model.get_total_time()/ dup


    with open("data/result.data", "a") as f:
        f.write(title + "\n")
        for j in range(len(name_lst)):
            f.write("\tmodel:{}\n".format(name_lst[j]))
            f.write("\t cost_init:{}, cost_final:{}, init_iter:{}, runtime_iter:{}, init_time:{}, total_time:{}\n".format(
                init_cost_lst[j], final_cost_lst[j], init_iter_lst[j], iter_lst[j], init_time_lst[j], total_time_lst[j]
            ))

    

# def benchmark(title, points, k, iter, dup):
#     basic_iter_avg, pp_iter_avg, par_iter_avg, par2_iter_avg = 0,0,0,0
#     pp_init_iter_avg, par_init_iter_avg, par2_init_iter_avg = 0,0,0,0
#     basic_cost_avg, pp_cost_avg, par_cost_avg, par2_cost_avg  = 0,0,0,0
#     basic_time_avg, pp_time_avg, par_time_avg, par2_time_avg = 0,0,0,0

#     for i in range(dup):
#         basic = kmeans_basic(points, k, iter=iter)
#         pp = kmeans_plusplus(points, k, iter)
#         par_weighted = kmeans_parallel(points, k,  k // 2 , 5, weighted=True )
#         par_weighted_2 = 
#         basic.train()
#         pp.train()
#         par_weighted.train()
#         par_weighted_2.train()

#         print("final cost\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}, kmeans||2_init:{}\
#             time\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}, kmeans||2_init:{} \
#         ".format(    
#             basic.get_cost(),
#             pp.get_cost(),
#             par_weighted.get_cost(),
#             par_weighted_2.get_cost(),
#             basic.get_time(),
#             pp.get_time(),
#             par_weighted.get_time(),
#             par_weighted_2.get_time()
#         ))
#         basic_iter_avg  += basic.get_runtime_iter() / dup
#         pp_iter_avg  += pp.get_runtime_iter() / dup
#         par_iter_avg  += par_weighted.get_runtime_iter() / dup
#         par2_iter_avg  += par_weighted_2.get_runtime_iter() / dup
#         pp_iter_avg  += pp.get_runtime_iter() / dup
#         par_iter_avg  += par_weighted.get_runtime_iter() / dup
#         par2_iter_avg  += par_weighted_2.get_runtime_iter() / dup
#         basic_cost_avg += basic.get_cost() / dup
#         pp_cost_avg += pp.get_cost() / dup
#         par_cost_avg += par_weighted.get_cost() / dup
#         par2_cost_avg += par_weighted_2.get_cost() / dup
#         basic_time_avg += basic.get_time() / dup
#         pp_time_avg += pp.get_time() / dup
#         par_time_avg += par_weighted.get_time() / dup
#         par2_time_avg += par_weighted_2.get_time() / dup

#     print("{}\n on average\n final cost\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}, kmeans||2_init:{} \
#             time\n random_init:{}, kmeans++_init:{}, kmeans||_init:{}, kmeans||2_init:{}  \
#         ".format(    
#             title, basic_cost_avg, pp_cost_avg, par_cost_avg, par2_cost_avg, basic_time_avg, pp_time_avg, par_time_avg, par2_time_avg
#     ))
#     with open("data/result.data", "a") as f:
#         f.write(title + "\n")
#         f.write("{},{},{},{},{},{},{},{}\n".format(basic_cost_avg, pp_cost_avg,
#          par_cost_avg, par2_cost_avg, basic_time_avg, pp_time_avg, par_time_avg, par2_time_avg))

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
            benchmark("guass_{}_{}".format(R,k), guass, k, iter, 10)