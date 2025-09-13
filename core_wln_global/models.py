import tensorflow as tf
from core_wln_global.mol_graph import max_nb
from core_wln_global.nn import *

def rcnn_wl_last(graph_inputs, batch_size, hidden_size, depth, training=True):
    '''This function performs the WLN embedding (local, no attention mechanism)'''
    input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = graph_inputs
    atom_features = tf.nn.relu(linearND(input_atom, hidden_size, "atom_embedding", init_bias=None))
    layers = []
    for i in range(depth):
        with tf.compat.v1.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)
            h_nei_atom = linearND(fatom_nei, hidden_size, "nei_atom", init_bias=None)
            h_nei_bond = linearND(fbond_nei, hidden_size, "nei_bond", init_bias=None)
            h_nei = h_nei_atom * h_nei_bond
            mask_nei = tf.reshape(tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32), [batch_size,-1,max_nb,1])
            f_nei = tf.reduce_sum(h_nei * mask_nei, -2)
            f_self = linearND(atom_features, hidden_size, "self_atom", init_bias=None)
            layers.append(f_nei * f_self * node_mask) # output
            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(linearND(l_nei, hidden_size, "label_U2"))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2) 
            new_label = tf.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            atom_features = tf.nn.relu(new_label) # updated atom features
    #kernels = tf.concat(1, layers)
    kernels = layers[-1] # atom FPs are the final output after "depth" convolutions
    fp = tf.reduce_sum(kernels, 1) # molecular FP is sum over atom FPs
    return kernels, fp



import tensorflow as tf
from core_wln_global.mol_graph import max_nb
import numpy as np


class RCNNWLModel(tf.keras.Model):
    def __init__(self, batch_size, hidden_size, depth):
        super(RCNNWLModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.depth = depth


        self.atom_embedding = tf.keras.layers.Dense(hidden_size, use_bias=None)
        self.nei_atom = tf.keras.layers.Dense(hidden_size, use_bias=None)
        self.nei_bond = tf.keras.layers.Dense(hidden_size, use_bias=None)
        self.self_atom = tf.keras.layers.Dense(hidden_size, use_bias=None)
        self.label_U2 = tf.keras.layers.Dense(hidden_size)
        self.label_U1 = tf.keras.layers.Dense(hidden_size)

    def call(self, graph_inputs):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = graph_inputs
        
        
        actual_batch_size = tf.shape(input_atom)[0]
        
        atom_features = tf.nn.relu(self.atom_embedding(input_atom))
        
        layers = []
        for i in range(self.depth):
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)
            
            h_nei_atom = self.nei_atom(fatom_nei)
            h_nei_bond = self.nei_bond(fbond_nei)
            h_nei = h_nei_atom * h_nei_bond
            
            mask_nei = tf.reshape(
                tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32),
                [actual_batch_size, -1, max_nb, 1]
            )
            
            f_nei = tf.reduce_sum(h_nei * mask_nei, -2)
            f_self = self.self_atom(atom_features)
            layers.append(f_nei * f_self * node_mask)
            
            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(self.label_U2(l_nei))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2)
            new_label = tf.concat([atom_features, nei_label], 2)
            atom_features = tf.nn.relu(self.label_U1(new_label))
        
        # 
        kernels = layers[-1]
        mol_hiddens = tf.reduce_sum(kernels, 1)
        
        return kernels, mol_hiddens

def linear(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.variable_scope(scope, reuse=reuse):
        W = tf.compat.v1.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    if init_bias is None:
        return tf.matmul(input_, W)
    with tf.variable_scope(scope, reuse=reuse):
        b = tf.compat.v1.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    return tf.matmul(input_, W) + b

def linearND(input_, output_size, scope, reuse=False, init_bias=0.0):
    shape = input_.get_shape().as_list()
    ndim = len(shape)
    stddev = min(1.0 / math.sqrt(shape[-1]), 0.1)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        W = tf.compat.v1.get_variable("Matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    X_shape = tf.gather(tf.shape(input_), list(range(ndim-1)))
    target_shape = tf.concat([X_shape, [output_size]], 0)
    exp_input = tf.reshape(input_, [-1, shape[-1]])
    if init_bias is None:
        res = tf.matmul(exp_input, W)
    else:
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            b = tf.compat.v1.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
        res = tf.matmul(exp_input, W) + b
    res = tf.reshape(res, target_shape)
    res.set_shape(shape[:-1] + [output_size])
    return res

