#!/usr/bin/env python
# coding: utf-8
# Created by xjiao004 at 02/05/2020
from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings("ignore")
import sys
import time
import argparse

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print("tf version: ", tf.__version__)

from gensim.models.word2vec import LineSentence, Word2Vec

import numpy as np
import pandas as pd

from utils.config import *
from utils.file_utils import save_dict, submit_proc, get_result_filename
from utils.multi_proc_utils import parallelize
from utils.data_loader import build_dataset, transform_data
from utils.models import Encoder, BahdanauAttention, Decoder, Seq2Seq
from utils.losses import loss_function
from utils.trainer import train_step
from utils.evaluate import evaluate, plot_attention, model_predict


# Train encoder-decoder-attention models through word2vec pretrained embedding matrix
# 1) filter words
# 2)
def train_models(checkpoint_dir, test_sentence, vocab_path, reverse_vocab_path, test=False):

  # 生成训练集和测试集
  train_df_X, train_df_Y, test_df_X, wv_model, X_max_len, train_y_max_len = build_dataset(train_data_path, test_data_path, save_wv_model_path, testOnly=test)

  

  # 词表大小
  vocab_size=len(vocab)
  params = {}
  params['vocab_size'] = vocab_size
  params['input_length'] = train_data_X.shape[1]

  vocab_inp_size = vocab_size
  vocab_tar_size = vocab_size
  input_length = train_data_X.shape[1]
  output_length = train_data_Y.shape[1]

  BUFFER_SIZE = len(train_data_X)
  steps_per_epoch = len(train_data_X)//BATCH_SIZE
  start_index = train_ids_y[0][0]

  # Dataset generator
  dataset = tf.data.Dataset.from_tensor_slices((train_data_X, train_data_Y)).shuffle(BUFFER_SIZE)
  dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

  # create sample input and target
  # example_input_batch, example_target_batch = next(iter(dataset))

  # # create encoder model
  encoder = Encoder(vocab_inp_size, embedding_dim, embedding_matrix, input_length, units, BATCH_SIZE)

  # create decoder model
  decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

  # model = Seq2Seq(params)

  # Define the optimizer and the loss function
  # optimizer = tf.keras.optimizers.Adam(1e-3)
  optimizer = tf.keras.optimizers.Adagrad(1e-3)

  # Checkpoints (Object-based saving)
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                  encoder=encoder,
                                  decoder=decoder)

  if test:
    # test only and plot results
    # 
    # * The evaluate function is similar to the training loop, except we don't use *teacher forcing* here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
    # * Stop predicting when the model predicts the *end token*.
    # * And store the *attention weights for every time step*.
    # 
    # Note: The encoder output is calculated only once for one input.

    # restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    # result, sentence, attention_plot = evaluate(encoder, decoder, test_sentence, vocab, reverse_vocab, units, input_length, train_y_max_len, start_index)
    results = model_predict(encoder, decoder, test_data_X, BATCH_SIZE, vocab, reverse_vocab, train_y_max_len, start_index)

    # print(results[1005])

    # 读入提交数据
    test_df=pd.read_csv(test_data_path)
    test_df.head()

    for idx,result in enumerate(results):
        if result=='':print(idx)

    # 赋值结果
    test_df['Prediction']=results
    #　提取ID和预测结果两列
    test_df=test_df[['QID','Prediction']]

    test_df.head()

    # 判断是否有空值
    # for predic in test_df['Prediction']:
    #     if type(predic) != str:
    #         print(predic)

    test_df['Prediction']=test_df['Prediction'].apply(submit_proc)

    test_df.head()

    # 获取结果存储路径
    result_save_path = get_result_filename(BATCH_SIZE, EPOCHS, X_max_len, embedding_dim, commit='_4_1_submit_seq2seq_code')

    # 保存结果.
    test_df.to_csv(result_save_path,index=None,sep=',')

    # 读取结果
    test_df=pd.read_csv(result_save_path)
    # 查看格式
    test_df.head(10)

    # print('Input: %s' % (sentence))
    # print('Predicted report: {}'.format(result))

    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

  else:
    # Training
    # 
    # 1. Pass the *input* through the *encoder* which return *encoder output* and the *encoder hidden state*.
    # 2. The encoder output, encoder hidden state and the decoder input (which is the *start token*) is passed to the decoder.
    # 3. The decoder returns the *predictions* and the *decoder hidden state*.
    # 4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
    # 5. Use *teacher forcing* to decide the next input to the decoder.
    # 6. *Teacher forcing* is the technique where the *target word* is passed as the *next input* to the decoder.
    # 7. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
    for epoch in range(EPOCHS):
      start = time.time()
      total_loss = 0

      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(encoder, decoder, inp, targ, optimizer, start_index)
        total_loss += batch_loss

        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                      batch,
                                                      batch_loss.numpy()))
      # saving (checkpoint) the model every epoch
      checkpoint.save(file_prefix = checkpoint_prefix)

      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Training Encoder-Decoder with attention")
  parser.add_argument("--models", default="./training_attention", help="checkpoint saving path", type=str)
  parser.add_argument("--vocab", default="./data/wv/vocab.txt", help="checkpoint saving path", type=str)
  parser.add_argument("--reversedVocab", default="./data/wv/reversed_vocab.txt", help="checkpoint saving path", type=str)
  parser.add_argument("--evaluate", default=False, help="test only or not", type=bool)
  args = parser.parse_args()
  
  # sentences predicted
  sentence1 = u'2010款宝马X1，2011年出厂，2.0排量，通用6L45变速箱，原地换挡位PRND车辆闯动，行驶升降档正常，4轮离地换挡无冲击感，更换变速箱油12L无改变。试过一辆2014年进口X1原地换挡位也有冲击感，这是什么情况，哪里的问题'
  sentence2 = u'车辆起步的时候汽车一顿熄火了。 这样会不会打坏或者打断发动机齿轮？'

  # Training encoder-decoder models
  train_models(args.models, sentence1, args.vocab, args.reversedVocab, test=args.evaluate)

  

  

  


