import tensorflow as tf
import sys
sys.path.append("..")
from core_wln_global.mol_graph import max_nb
from core_wln_global.nn import linearND, linear
from core_wln_global.models import RCNNWLModel
from core_wln_global.ioutils_direct import get_all_batch, INVALID_BOND
import math, random 
from collections import Counter
from optparse import OptionParser
from functools import partial
import numpy as np
from tqdm import tqdm
import time
import os
from rdkit import RDLogger, Chem
RDLogger.DisableLog('rdApp.*')

# GPU配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"找到 {len(gpus)} 个物理GPU, {len(logical_gpus)} 个逻辑GPU")
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("已设置使用GPU")
    except RuntimeError as e:
        print(e)
else:
    print("未找到GPU设备，将使用CPU")

# 配置命令行参数
parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default="XX.txt.proc")
parser.add_option("-m", "--save_dir", dest="save_path", default="./XX")
parser.add_option("-b", "--batch", dest="batch_size", default=25)
parser.add_option("-w", "--hidden", dest="hidden_size", default=768)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
parser.add_option("-r", "--rich", dest="rich_feat", default=False)
parser.add_option("--pretrained_ckpt", dest="pretrained_ckpt", default="./XX")
opts, args = parser.parse_args()

if opts.rich_feat:
    from core_wln_global.mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
else:
    from core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)
binary_fdim = 5  

smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

NK = 20  # 用于Acc@20
NK0 = 10  # 用于Acc@10

def count(s):
    return s.count(':')

def read_data(path):
    bucket_size = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]
    buckets = [[] for _ in range(len(bucket_size))]
    
    with open(path, 'r') as f:
        for line in f:
            r, e = line.strip("\r\n ").split()
            c = count(r)
            for i in range(len(bucket_size)):
                if c <= bucket_size[i]:
                    buckets[i].append((r, e))
                    break

    for bucket in buckets:
        random.shuffle(bucket)

    head = [0] * len(buckets)
    avil_buckets = [i for i in range(len(buckets)) if len(buckets[i])]

    while True:
        valid_samples = []
        bid = random.choice(avil_buckets)
        bucket = buckets[bid]
        it = head[bid]
        data_len = len(bucket)
        
        while len(valid_samples) < batch_size:
            react = bucket[it][0].split('>')[0]
            edit = bucket[it][1]
            it = (it + 1) % data_len
            
            try:
                mol = Chem.MolFromSmiles(react)
                if mol is not None:
                    valid_samples.append((react, edit))
            except:
                continue
                
            if it == head[bid] and len(valid_samples) < batch_size:
                bid = random.choice(avil_buckets)
                bucket = buckets[bid]
                it = head[bid]
                data_len = len(bucket)
        
        head[bid] = it
        
        src_batch, edit_batch = zip(*valid_samples)
        
        src_tuple = smiles2graph_batch(src_batch)
        if src_tuple is None:
            continue
            
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        if cur_bin is None:
            continue
            
        if len(src_batch) != batch_size:
            continue
            
        yield src_tuple, cur_bin, cur_label, sp_label

class DirectWGModel(tf.keras.Model):
    def __init__(self, hidden_size, batch_size, depth):
        super(DirectWGModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.depth = depth
        
        self.rcnn_wl = RCNNWLModel(batch_size, hidden_size, depth)
        
        self.att_atom_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.att_bin_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.att_score_dense = tf.keras.layers.Dense(1)
        
        self.pair_atom_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.pair_bin_dense = tf.keras.layers.Dense(hidden_size, use_bias=True)
        self.pair_ctx_dense = tf.keras.layers.Dense(hidden_size)
        self.score_dense = tf.keras.layers.Dense(5)  # 5种键类型的预测

    def call(self, inputs, training=True):
        input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, binary = inputs
        
        max_atoms = tf.shape(binary)[1]
        
        curr_bond_size = tf.shape(input_bond)[1]
        if curr_bond_size < max_atoms:
            input_bond = tf.pad(input_bond, [
                [0, 0],
                [0, max_atoms - curr_bond_size],
                [0, 0]
            ])
        else:
            input_bond = input_bond[:, :max_atoms, :]
        
        input_atom = tf.slice(input_atom, [0, 0, 0], [-1, max_atoms, -1])
        atom_graph = tf.slice(atom_graph, [0, 0, 0, 0], [-1, max_atoms, -1, -1])
        bond_graph = tf.slice(bond_graph, [0, 0, 0, 0], [-1, max_atoms, -1, -1])
        num_nbs = tf.slice(num_nbs, [0, 0], [-1, max_atoms])
        node_mask = tf.slice(node_mask, [0, 0], [-1, max_atoms])
        
        node_mask = tf.expand_dims(node_mask, -1)
        
        batch_size = tf.shape(input_atom)[0]
        
        atom_hiddens, _ = self.rcnn_wl((input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask))
        
        atom_hiddens1 = tf.expand_dims(atom_hiddens, axis=2)
        atom_hiddens2 = tf.expand_dims(atom_hiddens, axis=1)
        atom_pair = atom_hiddens1 + atom_hiddens2
        
        att_atom = self.att_atom_dense(atom_pair)
        att_bin = self.att_bin_dense(binary)
        att_hidden = tf.nn.relu(att_atom + att_bin)
        
        att_score = tf.nn.sigmoid(self.att_score_dense(att_hidden))
        
        att_context = att_score * atom_hiddens1
        att_context = tf.reduce_sum(att_context, axis=2)
        
        att_context1 = tf.expand_dims(att_context, axis=2)
        att_context2 = tf.expand_dims(att_context, axis=1)
        att_pair = att_context1 + att_context2
        
        pair_hidden = tf.nn.relu(
            self.pair_atom_dense(atom_pair) + 
            self.pair_bin_dense(binary) + 
            self.pair_ctx_dense(att_pair)
        )
        score = self.score_dense(pair_hidden)
        
        score = tf.reshape(score, [batch_size, max_atoms * max_atoms * 5])
        
        return score

    def get_topk(self, scores, k):
        return tf.nn.top_k(scores, k=k)[1]

def calculate_metrics(cur_topk, sp_label, batch_size, NK, NK0):
    sum_acc = 0.0
    sum_err = 0.0
    for i in range(batch_size):
        try:
            if i >= len(sp_label) or sp_label[i] is None:
                continue
                
            pre_20 = sum(1 for j in range(NK) if cur_topk[i,j].numpy() in sp_label[i])
            if len(sp_label[i]) == pre_20:
                sum_err += 1
            
            pre_10 = sum(1 for j in range(NK0) if cur_topk[i,j].numpy() in sp_label[i])
            if len(sp_label[i]) == pre_10:
                sum_acc += 1
        except Exception as e:
            print(f"Warning: Error processing batch {i}: {str(e)}")
            continue
    return sum_acc, sum_err

def main():
    model = DirectWGModel(hidden_size, batch_size, depth)
    
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0),
        model=model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    )
    
    latest_ckpt = tf.train.latest_checkpoint(opts.pretrained_ckpt)
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"成功加载预训练模型的权重: {latest_ckpt}")
    else:
        print("未找到预训练模型的检查点，将使用随机初始化的权重。")
    
    model.rcnn_wl.trainable = False
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=opts.save_path,
        max_to_keep=None  
    )
    
    num_epochs = 300  
    print(f"\n开始训练，总共 {num_epochs} 轮...")
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0
        sum_acc = 0.0
        sum_err = 0.0
        sum_gnorm = 0.0
        step = 0
        
        max_steps = 200  
        data_generator = read_data(opts.train_path)
        pbar = tqdm(total=max_steps, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        epoch_start_time = time.time()
        
        for src_tuple, cur_bin, cur_label, sp_label in data_generator:
            step += 1
            if step > max_steps:
                break
                
            if (src_tuple is None or cur_bin is None or 
                cur_label is None or sp_label is None or
                len(sp_label) < batch_size):
                print("跳过不完整的batch")
                continue
                
            with tf.GradientTape() as tape:
                input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = src_tuple
                
                if (len(input_atom) != batch_size or 
                    len(cur_bin) != batch_size or 
                    len(cur_label) != batch_size):
                    print("跳过维度不匹配的batch")
                    continue
                    
                inputs = (
                    tf.convert_to_tensor(input_atom, dtype=tf.float32),
                    tf.convert_to_tensor(input_bond, dtype=tf.float32),
                    tf.convert_to_tensor(atom_graph, dtype=tf.int32),
                    tf.convert_to_tensor(bond_graph, dtype=tf.int32),
                    tf.convert_to_tensor(num_nbs, dtype=tf.int32),
                    tf.convert_to_tensor(node_mask, dtype=tf.float32),
                    tf.convert_to_tensor(cur_bin, dtype=tf.float32)
                )
                
                max_atoms = tf.shape(cur_bin)[1]
                
                scores = model(inputs, training=True)
                
                cur_label = tf.slice(cur_label, [0, 0], [batch_size, max_atoms * max_atoms * 5])
                
                flat_scores = tf.reshape(scores, [-1])
                flat_labels = tf.reshape(cur_label, [-1])
                bond_mask = tf.cast(tf.not_equal(cur_label, INVALID_BOND), tf.float32)
                flat_labels = tf.maximum(0, flat_labels)
                
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.cast(flat_labels, tf.float32),
                        logits=flat_scores
                    )
                )
                loss = tf.reduce_sum(loss * bond_mask)
            
            trainable_variables = [var for var in model.trainable_variables if var.trainable]
            gradients = tape.gradient(loss, trainable_variables)
            gradients, gnorm = tf.clip_by_global_norm(gradients, max_norm)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            
            total_loss += loss
            sum_gnorm += gnorm

            scores_reshaped = scores
            cur_topk = model.get_topk(scores_reshaped, NK)
            
            sum_acc_batch, sum_err_batch = calculate_metrics(cur_topk, sp_label, batch_size, NK, NK0)
            sum_acc += sum_acc_batch
            sum_err += sum_err_batch
            
            pbar.update(1)
            
            if step % 50 == 0:
                acc_10 = sum_acc / (50 * batch_size)
                acc_20 = sum_err / (50 * batch_size)
                avg_gnorm = sum_gnorm / 50
                param_norm = tf.linalg.global_norm(model.trainable_variables)
                
                pbar.set_postfix({
                    'Loss': f'{total_loss/step:.4f}',
                    'Acc@10': f'{acc_10:.4f}',
                    'Acc@20': f'{acc_20:.4f}',
                    'GradNorm': f'{avg_gnorm:.2f}'
                })
                
                sum_acc = 0.0
                sum_err = 0.0
                sum_gnorm = 0.0

        pbar.close()
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} 完成 - 用时: {epoch_time:.2f}s")
        print(f"Average Loss: {total_loss/step:.4f}")
        
        if epoch % 2 == 0:
            save_path = manager.save(checkpoint_number=epoch)
            print(f"模型已保存到: {save_path}")
        
        if (epoch + 1) % 5 == 0:
            optimizer.learning_rate = optimizer.learning_rate * 0.9
            print(f"Learning rate adjusted to: {optimizer.learning_rate.numpy():.6f}")
        
        print("-" * 80)  

if __name__ == "__main__":
    main()