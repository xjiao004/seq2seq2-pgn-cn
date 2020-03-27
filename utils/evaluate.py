# Created by xjiao004 at 02/05/2020
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .data_loader import sentence_proc, pad_proc, transform_data
from .losses import loss_function
from .config import BATCH_SIZE, units

from .beam_search_helper import beam_decode

def evaluate(encoder, decoder, sentence, vocab, reverse_vocab, units, input_length, output_length, start_index):
  attention_plot = np.zeros((output_length, input_length))

  x_max_len = input_length - 4
  sentence = sentence_proc(sentence)
  sentence = pad_proc(sentence, x_max_len, vocab)

  inputs = transform_data(sentence, vocab)
  inputs = tf.convert_to_tensor([inputs])

  preidicts=[''] * BATCH_SIZE

  hidden = [tf.zeros((1 * BATCH_SIZE, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([start_index] * BATCH_SIZE, 1)

  for t in range(output_length):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    # greedy search, how about beam search?
    predicted_ids = tf.argmax(predictions[0]).numpy()

    # result += reverse_vocab[predicted_ids] + ' '

    # if reverse_vocab[predicted_id] == '<STOP>':
    #   return result, sentence, attention_plot
    for index,predicted_id in enumerate(predicted_ids):
      preidicts[index]+= reverse_vocab[predicted_id] + ' '

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  results = []
  for preidict in preidicts:
        # 去掉句子前后空格
        preidict=preidict.strip()
        # 句子小于max len就结束了 截断
        if '<STOP>' in preidict:
            # 截断stop
            preidict=preidict[:preidict.index('<STOP>')]
        # 保存结果
        results.append(preidict)
  return results

  # return result, sentence, attention_plot


# In[28]:


# # function for plotting the attention weights
# def plot_attention(attention, sentence, predicted_sentence):
#   fig = plt.figure(figsize=(300,300))
#   ax = fig.add_subplot(1, 1, 1)
#   ax.matshow(attention, cmap='viridis')

#   fontdict = {'fontsize': 10}

#   ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
#   ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

#   ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#   ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#   plt.show()


  # function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 12, 'fontproperties': font}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def batch_predict(inps, encoder, decoder, vocab, reverse_vocab, train_y_max_len, start_index):
  # max_length_targ = dec_targets.shape[1]
  # 判断输入长度
  batch_size=len(inps)
  # 开辟结果存储list
  preidicts=[''] * batch_size
  
  inps = tf.convert_to_tensor(inps)
  # 0. 初始化隐藏层输入
  hidden = [tf.zeros((1 * batch_size, units))]
  # hidden = encoder.initialize_hidden_state()
  # 1. 构建encoder
  # print("1>>>>>>>>>>>>", inps.shape)
  enc_output, enc_hidden = encoder(inps, hidden)
  # enc_output, enc_hidden = model.call_encoder(inps)

  # 2. 复制
  dec_hidden = enc_hidden

  # 3. <START> * BATCH_SIZE 
  dec_input = tf.expand_dims([start_index] * batch_size, 1)
  
  # context_vector, _ = attention(dec_hidden, enc_output)
  # Teacher forcing - feeding the target as the next input
  for t in range(train_y_max_len): # 输出序列的最大长度
    # 计算上下文
    # context_vector, attention_weights = model.attention(dec_hidden, enc_output)
    # 单步预测
    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
    # preidicts, dec_hidden = model.call_decoder_onestep(dec_input, dec_hidden, enc_output)
    
    # id转换 贪婪搜索, greedy search, 试想：换成beam search？
    predicted_ids = tf.argmax(predictions, axis=1).numpy()
    
    for index,predicted_id in enumerate(predicted_ids):
        preidicts[index]+= reverse_vocab[predicted_id] + ' '
    
    # using teacher forcing
    dec_input = tf.expand_dims(predicted_ids, 1)

  results=[]
  for preidict in preidicts:
    # 去掉句子前后空格
    preidict=preidict.strip()
    # 句子小于max len就结束了 截断
    if '<STOP>' in preidict:
        # 截断stop
        preidict=preidict[:preidict.index('<STOP>')]
    # 保存结果
    results.append(preidict)
  return results



from tqdm import tqdm
import math

def model_predict(encoder, decoder, data_X, batch_size, vocab, reverse_vocab, train_y_max_len, start_index):
  # 存储结果
  results=[]
  # 样本数量
  sample_size = len(data_X)
  # batch 操作轮数 math.ceil向上取整 小数 +1
  # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算  
  steps_epoch = math.ceil(sample_size / batch_size)
  # [0,steps_epoch)
  for i in tqdm(range(steps_epoch)):
    batch_data = data_X[i*batch_size:(i+1)*batch_size]
    results += batch_predict(batch_data, encoder, decoder, vocab, reverse_vocab, train_y_max_len, start_index)
  return results

def beam_search_predict(model, data_X, batch_size, vocab, reverse_vocab, params):
  # 存储结果
  results=[]
  # 样本数量
  sample_size = len(data_X)
  # batch 操作轮数 math.ceil向上取整 小数 +1
  # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算  
  steps_epoch = math.ceil(sample_size / batch_size)
  # [0,steps_epoch)
  for i in tqdm(range(steps_epoch)):
    batch_data = data_X[i*batch_size:(i+1)*batch_size]
    # results += batch_predict(batch_data, encoder, decoder, vocab, reverse_vocab, train_y_max_len, start_index)
    results += beam_decode(model, batch_data, vocab, reverse_vocab, params)
  return results

  