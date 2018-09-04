import config
import models
import tensorflow as tf
import numpy as np
import os
import shutil
import json
import pdb
import time
import socket
import datetime
import copy
import sys


os.environ['CUDA_VISIBLE_DEVICES']='0'
#Input training files from benchmarks/FB15K/ folder.
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
MENU = ["","train+test","train","test"]

PARAMS = {"test_flag",
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

REPORT_PARAMS = {"test_flag",
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

def obj_to_dict(obj):
  return {key:value for key, value in obj.__dict__.items() if not key.startswith('__') and not callable(key)}


class XpManager:

  def __init__(self, file = None):
    self._load(file)


  def filter_default_params(self, xp):
    for  pk,pv in xp['parameters'].items():
      try:
        if self.default_parameters[pk] == pv:
          xp['parameters'].pop(pk)
      except:
        continue


  def write(self, xp):
    # Reload the json file
    self._load(self.file)
    self.filter_default_params(xp)
    self.xp_list[xp["log"]["title"]] = xp
    # Save the host info
    xp["log"]["host"] = socket.gethostname()

    # Dump the XpManager object to json file
    self._dump()


  def _load(self, file= None):
    if file == None:
      file = self.file
    # Read experience json file
    self.file = file
    if os.path.exists(self.file):
      try:
        xp_man = json.load(open(self.file))
        for k, v in xp_man.items():
            setattr(self, k, v)
      except:
        pass
    else:
      self.size = 0
      self.xp_list = {}


  def _dump(self, file = None):
    if file == None:
      file = self.file
    # Recalculate the size
    xp_man.size = len(xp_man.xp_list)
    xp_man_dict = obj_to_dict(self)
    xp_man_dict.pop('file', None)
    xp_man_dict['last_modified']=time.strftime(TIME_FORMAT,time.localtime())
    json.dump(xp_man_dict, open(file,"w"), sort_keys=True, indent=4,)


# class ExpManager Ends



def report(xp):
    report = ""
    report += ("Corpus is  {}\n".format(xp["params_task"]["kb"]))
    report += ("There are {} relations and {} entities.\n"
              .format(xp["corpus"]["total_rel"],xp["corpus"]["total_ent"]))
    report += ("There are  {} train/ {} test/ {} valid triples.\n"
              .format(xp["corpus"]['total_train'],xp["corpus"]['total_test'],xp["corpus"]['total_valid']))          
    report += ("Experiment {} achieved {:.2f}k average MR. (current best {:.2f}k)\n"
               .format(xp["log"]["title"], xp["perf"]["avg_MR"]/1000, 182724/1000))
    if xp["params_task"]["model"] in ['TransE', 'DistMult']:
        report += ("Hyperparameters for the {} model were:\n{}\n"
                   .format(xp["params_task"]["model"], "\n".join([key+" : "+str(value) for key,value in xp["parameters"].items() if key in REPORT_PARAMS])))
    else:
        report += "Some baseline was used in this experiment, no deep learning.\n"
    report += "\n"
    report += ("Learned embeddings are stored at {}\n".format(xp["log"]["embeddings_path"]))
    report += ("Detailed results:\nMR: head {:.0f} / tail {:.0f} / avg {:.1f}\n"
               .format(xp["perf"]["head_MR"], xp["perf"]["tail_MR"], xp["perf"]["avg_MR"]))
    report += ("MRR: head {:.3f} / tail {:.3f} / avg {:.3f}\n"
               .format(xp["perf"]["head_MRR"], xp["perf"]["tail_MRR"], xp["perf"]["avg_MRR"]))
    report += ("Config files train and test.py are saved alongside model in {}\n"
               .format(os.path.abspath(xp["log"]["embeddings_path"])))
    report += ("Train Time: {}\nTest Time: {}\n".format(xp["log"]["time_train"],xp["log"]["time_test"]))
    print(report)
    save_via_email(xp["log"]["title"] + " {:.2f}k".format(xp["perf"]["avg_MR"]/1000) , report)

def save_via_email(title, message):
    # runs 'echo "MESSAGE" | mail -s "TITLE" lechellw@iro.umontreal.ca'
    import os, socket
    title = socket.gethostname().upper()+": "+ title
    os.system('echo "Experiment log :\n{}" | '
              'mail -s "{}" {}'
              .format(message, title, xp_man.mail_recepients))



xp_man = XpManager('./xps.json')
if len(sys.argv) < 2:
  print("# all experiences:\n#\n")
  #List all the xps
  nx = xp_man.xp_list.keys()
  nx.sort()
  for n in nx:
    if not n.startswith('_'):
      print(n)
  xp_man._dump()
  exit(1)
xp_title = sys.argv[1]

xp = {}
active_params = copy.deepcopy(xp_man.default_parameters)
try:
  xp = xp_man.xp_list[xp_title]
  xp["log"]={}
  xp["log"]["title"]=xp_title
  # Clean the params
  xp["params_task"]["task"] = xp["params_task"]["task"].lower().replace(" ", "")
  print("The {} for {} begins...".format(xp["params_task"]["task"],xp["log"]["title"]))
  xp["log"]["title"] = xp_title
  print(json.dumps(xp,sort_keys=True, indent=4,))
  # Set current xp parameter values
  active_params.update(xp['parameters'])
except:
  # In case the xp is not in json (Shouldn't happen!)
  print("{} is not in file {}.\n".format(xp_title, xp_man.file))
  if raw_input("Create new experience {}?[y/n]".format(xp_title)) == 'y':
    xp=xp_man.xp_list["_template"].copy()
    xp["log"]={}
    xp["log"]["title"]=xp_title
    xp_man.write(xp)
    print("The {} is created from the template. Please modify it in {}".format(xp_title, xp_man.file))
  exit(1)

try:
  ans = MENU.index(xp["params_task"]["task"])
except:
  print("Task \"{}\" is unknown {}".format(xp['task'],MENU))
  exit(1)

con = config.Config()

# Save the default parameter value from config.py
default_params = {}
for x in PARAMS:
  default_params[x]=getattr(con, x)


########
# Set active parameter values
# Pass the params to openke
for k,v in active_params.items():
  setattr(con, k, v)

# The in_path must ends in /(slash)
con.in_path = os.path.abspath("data/"+xp["params_task"]["kb"])+"/"
xp["log"]["in_path"] = con.in_path


# If it's train + test or test only
if "test" in MENU[ans]:
  con.set_test_link_prediction(True)


try:
  xp["params_task"]["xp_path_base"] = os.path.abspath(xp["params_task"]["xp_path_base"])
except:
  xp["params_task"]["xp_path_base"]= os.path.abspath("models".format())


xp["log"]["embeddings_path"] = os.path.abspath("{}/{}".format(xp["params_task"]["xp_path_base"],xp["log"]["title"]))



if not os.path.exists(xp["log"]["embeddings_path"]):
    os.makedirs(xp["log"]["embeddings_path"])
else:
  print("{} exists already!".format(xp["log"]["title"]))
  if raw_input("Input \"{}\" to confirm the override and to continue: ".format(xp["log"]["title"])) != xp["log"]["title"]:
    exit(1)


if "train" in MENU[ans]:
  #Models will be exported via tf.Saver() automatically.
  con.set_export_files("{}/model.vec.tf".format(xp["log"]["embeddings_path"]), 0)
  #Model parameters will be exported to json files automatically.
  con.set_out_files("{}/embeddings.vec.json".format(xp["log"]["embeddings_path"]))
else:
  con.set_import_files("{}/model.vec.tf".format(xp["log"]["embeddings_path"]))


print(active_params)


#Initialize experimental settings.
con.init()


#Get the corpus info (totals of relations and entities etc.)
xp["corpus"] = {}
xp["corpus"]["total_ent"] = con.lib.getEntityTotal()
xp["corpus"]["total_rel"] = con.lib.getRelationTotal()
xp["corpus"]["total_train"] = con.lib.getTrainTotal()
xp["corpus"]["total_test"] = con.lib.getTestTotal()
xp["corpus"]["total_valid"] = con.lib.getValidTotal()


#Set the knowledge embedding model
con.set_model(getattr(models, xp["params_task"]["model"])) # models.TransE

#Train the model.
# Train+ test or Train only
if "train" in MENU[ans]:
  t0 = time.time()
  con.run()
  xp["log"]["time_train"] = str(datetime.timedelta(seconds=time.time()-t0))
  # Save the train_times
  xp["log"]["train_times"] = con.train_times

#To test models after training needs "set_test_flag(True)".
# Train+ test or Test only
if "test" in MENU[ans]:
  t0 = time.time()
  con.test()
  xp["log"]["time_test"] = str(datetime.timedelta(seconds=time.time()-t0))

  # Get the performance
  xp["perf"] = {}
  xp["perf"]["head_MR"] = con.get_head_mr()
  xp["perf"]["tail_MR"] = con.get_tail_mr()
  xp["perf"]["avg_MR"] = (xp["perf"]["head_MR"] + xp["perf"]["tail_MR"])/2
  xp["perf"]["head_MRR"] = con.get_head_mrr()
  xp["perf"]["tail_MRR"] = con.get_tail_mrr()
  xp["perf"]["avg_MRR"] = (xp["perf"]["head_MRR"] + xp["perf"]["tail_MRR"])/2


  shutil.copy2("log.log", "{}/".format(xp["log"]["embeddings_path"]))


  report(xp)

xp_man.write(xp)

 
