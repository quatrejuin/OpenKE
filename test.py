import config
import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['CONDA_PREFIX']=''

kb = 'RV15M' # FB15K
model = "TransE" # DistMult

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
