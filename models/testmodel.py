#coding:utf-8
import numpy as np
import tensorflow as tf
from .Model import Model

class testmodel(Model):
    def forward(self,arg,rel):
        arg_e = tf.nn.embedding_lookup(self.ent_embeddings, arg)
        rel_e = tf.nn.embedding_lookup(self.rel_embeddings, rel)
        cat_e = tf.concat(arg_e,rel_e)
        pred = tf.nn.softmax(tf.add( tf.matmul(cat_e, self.nce_weight), self.nce_biases))
        prediction = pred
        return prediction
    
    # The outpout dimension of _calc , for the call from forward() it's 2, (batchsize,1), the call from predict_def() it's 1, (?,)
    def _calc(self, h, t, r, pred = False):
        config = self.get_config()
        # axis = h.shape[-1],  for the call from forward() it's 2, the call from predict_def() it's 1
        entropy_tail = -tf.reduce_sum(tf.one_hot(t,config.entTotal) * tf.log(self.forward(h,r)), axis = 1 if pred else 2)
        entropy_head = -tf.reduce_sum(tf.one_hot(h,config.entTotal) * tf.log(self.forward(t,r)), axis = 1 if pred else 2)
        print((tf.reduce_mean([entropy_tail,entropy_head], 0, keepdims = False)).shape)
        return tf.reduce_mean([entropy_tail,entropy_head], 0, keepdims = False)


    def embedding_def(self):
        #Obtaining the initial configuration of the model
        config = self.get_config()
        #Defining required parameters of the model, including embeddings of entities and relations
        self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [config.entTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [config.relTotal, config.hidden_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))      
        # Output weight and biases
        self.nce_weight = tf.get_variable(name = "nce_weight", shape = [config.hidden_size*2, config.entTotal], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        self.nce_biases = tf.get_variable(name="nce_biases", shape=[config.entTotal], initializer=tf.zeros_initializer())
        self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
                                "rel_embeddings":self.rel_embeddings, \
                                "nce_weight":self.nce_weight, \
                                "nce_biases":self.nce_biases    
                                }

    def loss_def(self):
        #Obtaining the initial configuration of the model
        config = self.get_config()
        #To get positive triples and negative triples for training
        #The shapes of pos_h, pos_t, pos_r are (batch_size, 1)
        #The shapes of neg_h, neg_t, neg_r are (batch_size, negative_ent + negative_rel)
        pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
        neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
        #Embedding entities and relations of triples, e.g. p_h, p_t and p_r are embeddings for positive triples
        p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
        p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
        p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
        n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
        n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
        n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
        #Calculating score functions for all positive triples and negative triples
        # The shape of new _p_score is (batch_size, 1)
        # The shape of new _n_score is (batch_size, negative_ent + negative_rel)
        _p_score = self._calc(pos_h, pos_t, pos_r)
        _n_score = self._calc(neg_h, neg_t, neg_r) 
        #The shape of p_score is (batch_size, 1)
        #The shape of n_score is (batch_size, 1)
        p_score =  tf.reduce_sum(_p_score, -1, keepdims = True)
        n_score =  tf.reduce_sum(_n_score, -1, keepdims = True)
        #Calculating loss to get what the framework will optimize
        self.loss = tf.reduce_mean(tf.maximum(p_score - n_score + config.margin, 0))

    def predict_def(self):
        predict_h, predict_t, predict_r = self.get_predict_instance()
        self.predict = self._calc(predict_h, predict_t, predict_r, pred = True)
