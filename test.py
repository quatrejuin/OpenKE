import config
import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CONDA_PREFIX']=''

def main():
    kb = 'RV15M' # FB15K
    model = "TransE" # DistMult
    clusters = "V2"

    con = config.Config()
    con.set_in_path("/u/wujieche/Projects/OpenKE/data/"+kb+"/")
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_dimension(100)
    con.set_import_files("models/{}-{}_model.vec.tf".format(model, kb))
    con.init()
    con.set_model(getattr(models, model)) # models.TransE
    con.test(clusters)
    log_path = "{}-{}_clusters_test_log.log".format(model, kb)
    os.rename("log.log", log_path)

    # rank_by_relID_plot()

def read_log(path):
    log = []
    with open(path) as f:
        for l in f:
            arg1, rel, arg2, head_rank, tail_rank = l.split()
            log.append([rel, head_rank, tail_rank])
    return log

def rank_by_relID_plot(log_path):
    log = read_log(log_path)
    import matplotlib.pyplot as plt
    head_ranks = [hr for r, hr, _ in log]
    tail_ranks = [tr for r, _, tr in log]
    ids = [r for r, _, __ in log]

    # FIXME Make a pandas dataframe directly out of the log data
    # (because pandas has nice data processing functions useful for graphing)
    
    # df = pd.read_csv("log_no_cluster.csv")
    # df = df.drop(columns=["id", "index", "h", "t"])
    # max_r = max(df['r'])
    # # print(max_r)

    # df['ranked_ids'] = df['r'].rank(method='first')
    # # df['x'] = pd.qcut(df['ranked_ids'], 1000)

    
    freq_head_MR, freq_tail_MR = 169512, 237961
    darkblue, darkgreen = "#3030AA", "#40AA40"

    # bins = 30
    # grp = df.groupby(by = pd.qcut(df['r'], bins))
    # df = grp.aggregate(np.average)

    
    plt.plot(ids, tail_ranks, "b", label = "arg1 mean rank")
    plt.plot(ids, head_ranks, "g", label = "arg2 mean rank")
    plt.hlines([freq_head_MR, freq_tail_MR], xmin=0, xmax = 10000, colors = [darkblue, darkgreen], label = "baseline")
    plt.legend()
    plt.savefig(log_path[:-4]+".MR_by_ID-graph.png")
    
main()
