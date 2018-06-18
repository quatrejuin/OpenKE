import config
import models
import tensorflow as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.

kb = 'RV15M' # FB15K, DefIE
model = "TransE" # DistMult
clusters = "V2"

con = config.Config()
# Currently FB15K is in benchmarks/FB15K instead of data/FB15K
con.set_in_path("/u/wujieche/Projects/OpenKE/data/"+kb+"/")

con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("models/{}-{}-{}_model.vec.tf".format(model, kb, clusters), 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("models/{}-{}-{}_embeddings.vec.json".format(model, kb, clusters))
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(getattr(models, model)) # models.TransE
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
#con.test()

