# -*- coding:utf-8 -*-
# Created by xjiao004 at 02/05/2020
import os
import sys
sys.path.append("./encoder-decoder-attention")
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据路径
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
# 测试数据路径
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
# 停用词路径
stop_word_path = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')

# 自定义切词表
user_dict = os.path.join(root, 'data', 'user_dict.txt')

# 0. 预处理
# 预处理后的训练数据
train_seg_path = os.path.join(root, 'data', 'train_seg_data.csv')
# 预处理后的测试数据
test_seg_path = os.path.join(root, 'data', 'test_seg_data.csv')
# 合并训练集测试集数据
merger_seg_path = os.path.join(root, 'data', 'merged_train_test_seg_data.csv')

# 1. 数据标签分离
train_x_seg_path = os.path.join(root, 'data', 'train_X_seg_data.csv')
train_y_seg_path = os.path.join(root, 'data', 'train_Y_seg_data.csv')
val_x_seg_path = os.path.join(root, 'data', 'val_X_seg_data.csv')
val_y_seg_path = os.path.join(root, 'data', 'val_Y_seg_data.csv')
test_x_seg_path = os.path.join(root, 'data', 'test_X_seg_data.csv')

# 2. pad oov处理后的数据
train_x_pad_path = os.path.join(root, 'data', 'train_X_pad_data.csv')
train_y_pad_path = os.path.join(root, 'data', 'train_Y_pad_data.csv')
test_x_pad_path = os.path.join(root, 'data', 'test_X_pad_data.csv')

# 3. numpy 转换后的数据
train_x_path = os.path.join(root, 'data', 'train_X')
train_y_path = os.path.join(root, 'data', 'train_Y')
test_x_path = os.path.join(root, 'data', 'test_X')

# 词向量路径
save_wv_model_path = os.path.join(root, 'data', 'wv', 'word2vec.model')

embedding_matrix_path = os.path.join(root, 'data', 'wv', 'embedding_matrix')
# 辞典保存路径
save_vocab_path = os.path.join(root, 'data', 'wv', 'vocab.txt')
# 辞典与index保存路径
reverse_vocab_path = os.path.join(root, 'data', 'wv', 'reversed_vocab.txt')

checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_pgn_cov_backed')

# checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')
seq2seq_checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, 'result')

# 词向量训练轮数
wv_train_epochs = 5

# 词向量维度
#embedding_dim = 256

# encoder-decoder 隐藏层单元数
#units = 300

# attention layer 的units数
#att_units = 300

# 每个batch的数量
#BATCH_SIZE = 32

# encoder-decoder-attention 模型迭代次数
#EPOCHS = 200

# 词向量维度
embedding_dim = 300

sample_total = 82871

batch_size = 50

epochs = 100

vocab_size = 50000

