import config
import models
import tensorflow as tf
import numpy as np
import os
import shutil

os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.

kb = 'FB15K' # FB15K, DefIE
model = "TransE" # DistMult
clusters = ""

xp ={}


#
##
###
####
#####
######
xp["title"] = "xp100"
######
#####
####
###
##
#


params = {"test_flag",
"in_path",
"out_path",
"bern",
"hidden_size",
"ent_size",
"rel_size",
"train_times",
"margin",
"nbatches",
"negative_ent",
"negative_rel",
"workThreads",
"alpha",
"lmbda",
"log_on",
"exportName",
"importName",
"export_steps",
"opt_method",
"optimizer",
"test_link_prediction",
"test_triple_classification",}

report_params = {"test_flag",
"in_path",
"out_path",
"bern",
"hidden_size",
"ent_size",
"rel_size",
"train_times",
"margin",
"nbatches",
"negative_ent",
"negative_rel",
"alpha",
"lmbda",
"log_on",
"exportName",
"importName",
"export_steps",
"opt_method",
"optimizer",}

def report(xp):
    report = ""
    report += ("Corpus is in {}\n".format(xp["parameters"]['in_path']))
    report += ("There are {} relations and {} entities.\n"
              .format(xp["total_rel"],xp["total_ent"]))
    report += ("There are  {} train/ {} test/ {} valid triples.\n"
              .format(xp['total_train'],xp['total_test'],xp['total_valid']))          
    report += ("Experiment {} achieved {:.2f}k average MR. (current best {:.2f}k)\n"
               .format(xp["title"], xp["average_mean_rank"]/1000, 180123/1000))
    if xp["model"] in ['TransE', 'DistMult']:
        report += ("Hyperparameters for the {} model were:\n{}\n"
                   .format(xp["model"], "\n".join([key+" : "+str(value) for key,value in xp["parameters"].items() if key in report_params])))
    else:
        report += "Some baseline was used in this experiment, no deep learning.\n"
    report += "\n"
    report += ("Learned embeddings are stored at {}\n".format(xp["embeddings_save_path"]))
    # I guess /u/wujieche/Projects/OpenKE/models/{xp["title"]}/embeddings.vec.json
    report += ("Detailed results:\nMR: head {:.0f} / tail {:.0f} / avg {:.1f}\n"
               .format(xp["head_MR"], xp["tail_MR"], xp["avg_MR"]))
    report += ("MRR: head {:.3f} / tail {:.3f} / avg {:.3f}\n"
               .format(xp["head_MRR"], xp["tail_MRR"], xp["avg_MRR"]))
    report += ("Config files train and test.py are saved alongside model in {}\n"
               .format(os.path.abspath(xp["embeddings_save_path"])))
    # same: /u/wujieche/Projects/OpenKE/models/{xp["title"]}/train.py
    print(report)
    save_via_email(xp["title"] + " {:.2f}k".format(xp["average_mean_rank"]/1000) , report)

def save_via_email(title, message):
    # runs 'echo "MESSAGE" | mail -s "TITLE" lechellw@iro.umontreal.ca'
    import os, socket
    title = socket.gethostname().upper()+": "+ title
    os.system('echo "Experiment log :\n{}" | '
              'mail -s "{}" william.lechelle@gmail.com'
              .format(message, title))  


ans= False
while not ans:
    print ("""
    1.Train + Test
    2.Train Only
    3.Test Only
    4.Exit/Quit
    """)
    ans=int(input("What would you like to do? "))
    if ans==1: 
      print("\n Train + Test") 
    elif ans==2:
      print("\n Train Only") 
    elif ans==3:
      print("\n Test Only") 
    elif ans==4:
      print("\n Goodbye")
      exit() 
    else:
      ans=1

menu = ["","train+test","train","test"]


con = config.Config()

# Save the default parameter value in config.py
default_params = {}
for x in params:
  default_params[x]=getattr(con, x)

# Currently FB15K is in benchmarks/FB15K instead of data/FB15K
con.set_in_path("/u/wujieche/Projects/OpenKE/data/"+kb+"/")

con.set_test_triple_classification(False)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(1000)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

##########################
con.set_train_times(1)
con.set_nbatches(100000)
con.set_alpha(0.01)
con.set_opt_method("Adagrad")
con.set_dimension(200)
##########################

# If it's train+ test or test only
if "test" in menu[ans]:
  con.set_test_link_prediction(True)
#####

xp["model"] = model
xp["embeddings_save_path"] = "models/{}".format(xp["title"])
import os
if not os.path.exists(xp["embeddings_save_path"]):
    os.makedirs(xp["embeddings_save_path"])
else:
  print("{} exists already!".format(xp["title"]))
  exit(1)

xp["parameters"] = {}
for x in params:
  if default_params[x]!=getattr(con, x):
    xp["parameters"][x]=getattr(con, x)



if "train" in menu[ans]:
  #Models will be exported via tf.Saver() automatically.
  con.set_export_files("{}/model.vec.tf".format(xp["embeddings_save_path"]), 0)
  #Model parameters will be exported to json files automatically.
  con.set_out_files("{}/embeddings.vec.json".format(xp["embeddings_save_path"]))
else:
  con.set_import_files("{}/model.vec.tf".format(xp["embeddings_save_path"]))

print(xp["parameters"])
 
#Initialize experimental settings.
con.init()

#Set the knowledge embedding model
con.set_model(getattr(models, model)) # models.TransE
#Train the model.
# Train+ test or Train only
if "train" in menu[ans]:
  con.run()
#To test models after training needs "set_test_flag(True)".
# Train+ test or Test only
if "test" in menu[ans]:
  con.test()

  # Get the performance
  xp["head_MR"] = con.get_head_mr()
  xp["tail_MR"] = con.get_tail_mr()
  xp["avg_MR"] = (xp["head_MR"] + xp["tail_MR"])/2
  xp["head_MRR"] = con.get_head_mrr()
  xp["tail_MRR"] = con.get_tail_mrr()
  xp["avg_MRR"] = (xp["head_MRR"] + xp["tail_MRR"])/2
  xp["average_mean_rank"] = xp["avg_MR"]

  #Get the corpus info (totals of relations and entities etc.)
  xp["total_ent"] = con.lib.getEntityTotal()
  xp["total_rel"] = con.lib.getRelationTotal()
  xp["total_train"] = con.lib.getTrainTotal()
  xp["total_test"] = con.lib.getTestTotal()
  xp["total_valid"] = con.lib.getValidTotal()


  shutil.copy2("log.log", "{}/".format(xp["embeddings_save_path"]))


  report(xp)



 