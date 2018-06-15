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
    clusters = "V1"

    con = config.Config()
    con.set_in_path("/u/wujieche/Projects/OpenKE/data/"+kb+"/")
    con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_dimension(100)
    con.set_import_files("models/{}-{}_model.vec.tf".format(model, kb))
    con.init()
    con.set_model(getattr(models, model)) # models.TransE
    con.test()
    log_path = "{}-{}_clusters_test_log.log".format(model, kb)
    os.rename("log.log", log_path)

    rank_by_relID_plot(read_log(log_path))

def read_log(path):
    log = []
    with open(path) as f:
        for l in f:
            arg1, rel, arg2, head_rank, tail_rank = l.split()
            log.append([rel, head_rank, tail_rank])
    return log

def rank_by_relID_plot(log):
    import matplotlib.pyplot as plt
    head_ranks = [[r, hr] for r, hr, _ in log]
    tail_ranks = [[r, tr] for r, _, tr in log]
    plt.plot(zip(*head_ranks))
    plt.plot(zip(*tail_ranks))
    plt.display()
    # FIXME Smooth the graph by binning some values together (aggregate 1 to 10 rel values)
    
main()
