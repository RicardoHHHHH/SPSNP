import sys
sys.path.append("..")
import tensorflow as tf
from core_wln_global.nn import linearND, linear
from core_wln_global.models import RCNNWLModel
from core_wln_global.ioutils_direct import *
import math, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import os
from rdkit import Chem
from rdkit import RDLogger


RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings('ignore')

def count(s):
    return s.count(':')

class DirectWGModel(tf.keras.Model):
    def __init__(self, hidden_size, batch_size, depth):
        super(DirectWGModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.depth = depth

        # 
        self.rcnn_wl = RCNNWLModel(batch_size, hidden_size, depth)

        # 
        self.att_atom_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.att_bin_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.att_score_dense = tf.keras.layers.Dense(1)

        self.pair_atom_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.pair_bin_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.pair_ctx_dense = tf.keras.layers.Dense(hidden_size)
        self.score_dense = tf.keras.layers.Dense(5)  # 

    def call(self, inputs, training=False):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, binary = inputs
        node_mask = tf.expand_dims(node_mask, -1)

        # 
        atom_hiddens, _ = self.rcnn_wl((input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask))

        # 
        atom_hiddens1 = tf.expand_dims(atom_hiddens, axis=2)
        atom_hiddens2 = tf.expand_dims(atom_hiddens, axis=1)
        atom_pair = atom_hiddens1 + atom_hiddens2

        # 
        att_hidden = tf.nn.relu(self.att_atom_dense(atom_pair) + self.att_bin_dense(binary))
        att_score = tf.nn.sigmoid(self.att_score_dense(att_hidden))

        # 
        att_context = att_score * atom_hiddens1
        att_context = tf.reduce_sum(att_context, axis=2)

        # 
        att_context1 = tf.expand_dims(att_context, axis=2)
        att_context2 = tf.expand_dims(att_context, axis=1)
        att_pair = att_context1 + att_context2

        # 
        pair_hidden = tf.nn.relu(
            self.pair_atom_dense(atom_pair) +
            self.pair_bin_dense(binary) +
            self.pair_ctx_dense(att_pair)
        )
        score = self.score_dense(pair_hidden)

        return score

    def get_topk(self, scores, k):

        return tf.nn.top_k(scores, k=k)[1]

def read_data(path, batch_size, smiles2graph_batch):
    all_data = []
    with open(path, 'r') as f:
        for line in f:
            r, e = line.strip("\r\n ").split()
            all_data.append((r, e))

    data_len = len(all_data)
    for i in range(0, data_len, batch_size):
        end = min(i + batch_size, data_len)
        src_batch = [data[0].split('>')[0] for data in all_data[i:end]]
        edit_batch = [data[1] for data in all_data[i:end]]

        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))

        yield src_tuple, cur_bin, cur_label, sp_label, src_batch


def test_stage1():
    predfile = open('XXX.cbond', 'w')
    print(f"Prediction file path: {predfile.name}")
    NK3 = 80
    NK2 = 60
    NK1 = 40
    NK0 = 20
    NK = 10

    parser = OptionParser()
    parser.add_option("-t", "--test", dest="train_path", default="XX.txt.proc")
    parser.add_option("-m", "--model", dest="model_path", default="./XX")
    parser.add_option("-b", "--batch", dest="batch_size", default=20)
    parser.add_option("-w", "--hidden", dest="hidden_size", default=768)
    parser.add_option("-d", "--depth", dest="depth", default=3)
    parser.add_option("-r", "--rich", dest="rich_feat", default=False)
    parser.add_option("-v", "--verbose", dest="verbose", default=1)
    parser.add_option("--hard", dest="hard", default=False)
    parser.add_option("--detailed", dest="detailed", default=1)
    opts, args = parser.parse_args()

    batch_size = int(opts.batch_size)
    hidden_size = int(opts.hidden_size)
    depth = int(opts.depth)
    detailed = bool(opts.detailed)

    if opts.rich_feat:
        from core_wln_global.mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
    else:
        from core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g

    smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIntProp('molAtomMapNumber') - 1)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print(f"模型路径: {opts.model_path}")

    model = DirectWGModel(hidden_size, batch_size, depth)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        model=model,
        optimizer=optimizer
    )

    latest_checkpoint = tf.train.latest_checkpoint(opts.model_path)
    if latest_checkpoint is None:
        raise ValueError(f"未找到checkpoint文件: {opts.model_path}")

    print(f"加载checkpoint: {latest_checkpoint}")
    status = checkpoint.restore(latest_checkpoint)
    status.expect_partial()  
    print(f"成功加载模型权重，当前epoch: {int(checkpoint.epoch)}")

    it = 0
    n_cnt = 0 
    acc_counters = [0, 0, 0, 0, 0]  

    bo_to_index = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}
    bindex_to_o = {val: key for key, val in bo_to_index.items()}
    nbos = len(bo_to_index)

    data_generator = read_data(opts.train_path, batch_size, smiles2graph_batch)
    for src_tuple, cur_bin, cur_label, sp_label, src_batch in data_generator:
        if src_tuple is None or cur_bin is None or cur_label is None or sp_label is None:
            print("跳过无效batch")
            continue

        if (len(sp_label) < batch_size or
                len(cur_bin) != batch_size or
                len(cur_label) != batch_size):
            print("跳过大小不匹配的batch")
            continue

        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = src_tuple
        inputs = (
            tf.convert_to_tensor(input_atom, dtype=tf.float32),
            tf.convert_to_tensor(input_bond, dtype=tf.float32),
            tf.convert_to_tensor(atom_graph, dtype=tf.int32),
            tf.convert_to_tensor(bond_graph, dtype=tf.int32),
            tf.convert_to_tensor(num_nbs, dtype=tf.int32),
            tf.convert_to_tensor(node_mask, dtype=tf.float32),
            tf.convert_to_tensor(cur_bin, dtype=tf.float32)
        )

        scores = model(inputs, training=False)
        scores_reshaped = tf.reshape(scores, [batch_size, -1])

        cur_topk = model.get_topk(scores_reshaped, NK3)
        cur_dim = tf.cast(tf.sqrt(tf.cast(tf.shape(cur_label)[1], tf.float32) / 5), tf.int32)

        for i in range(batch_size):
            try:
                if i >= len(sp_label) or sp_label[i] is None:
                    continue

                for idx, k in enumerate([NK3, NK2, NK1, NK0, NK]):
                    pre = sum(1 for j in range(k)
                              if cur_topk[i, j].numpy() in sp_label[i])
                    if len(sp_label[i]) == pre:
                        acc_counters[idx] += 1
                n_cnt += 1

            except Exception as e:
                print(f"处理第{i}个样本时出错: {str(e)}")
                continue

            ratoms = set([atom.GetAtomMapNum() for atom in Chem.MolFromSmiles(src_batch[i]).GetAtoms()])
            rbonds = [
                tuple(sorted([b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()]) + [b.GetBondTypeAsDouble()])
                for b in Chem.MolFromSmiles(src_batch[i]).GetBonds()
            ]

            if opts.verbose:
                predfile.write("{}".format(src_batch[i]) + ' ')
                for j in range(NK3):
                    k = cur_topk[i, j].numpy()
                    bindex = k % nbos
                    y = int(((k - bindex) / nbos) % cur_dim + 1)
                    x = int((k - bindex - (y - 1) * nbos) / (cur_dim * nbos) + 1)

                    bindex = int(bindex)
                    bo = bindex_to_o[bindex]

                    if x < y and x in ratoms and y in ratoms and (x, y, bo) not in rbonds:
                        predfile.write("{}-{}-{:.1f}".format(x, y, bo) + ' ')
                        if detailed:
                            predfile.write("{:.3f}".format(scores_reshaped[i, j].numpy()) + ' ')
                predfile.write('\n')

        it += batch_size
        if it % 100 == 0:
            print("After seeing {}, acc@10: {:.3f}, acc@20: {:.3f}, acc@40: {:.3f}, acc@60: {:.3f}, acc@80: {:.3f}".format(
                it,
                acc_counters[4] / n_cnt if n_cnt > 0 else 0,  # NK
                acc_counters[3] / n_cnt if n_cnt > 0 else 0,  # NK0
                acc_counters[2] / n_cnt if n_cnt > 0 else 0,  # NK1
                acc_counters[1] / n_cnt if n_cnt > 0 else 0,  # NK2
                acc_counters[0] / n_cnt if n_cnt > 0 else 0  # NK3
            ))
    overall_acc_NK = acc_counters[4] / n_cnt if n_cnt > 0 else 0
    overall_acc_NK0 = acc_counters[3] / n_cnt if n_cnt > 0 else 0
    overall_acc_NK1 = acc_counters[2] / n_cnt if n_cnt > 0 else 0
    overall_acc_NK2 = acc_counters[1] / n_cnt if n_cnt > 0 else 0
    overall_acc_NK3 = acc_counters[0] / n_cnt if n_cnt > 0 else 0

    print("总体准确率:")
    print(f"acc@10: {overall_acc_NK:.3f}")
    print(f"acc@20: {overall_acc_NK0:.3f}")
    print(f"acc@40: {overall_acc_NK1:.3f}")
    print(f"acc@60: {overall_acc_NK2:.3f}")
    print(f"acc@80: {overall_acc_NK3:.3f}")

    predfile.close()


if __name__ == '__main__':
    test_stage1()