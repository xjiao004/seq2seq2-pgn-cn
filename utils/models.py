import tensorflow as tf

from .config import *
from .data_loader import load_embedding_matrix

# Write the encoder and decoder model
# 
# Implement an encoder-decoder model with attention which you can read about in the TensorFlow [Neural Machine Translation (seq2seq) tutorial](https://github.com/tensorflow/nmt). This example uses a more recent set of APIs. This notebook implements the [attention equations](https://github.com/tensorflow/nmt#background-on-the-attention-mechanism) from the seq2seq tutorial. The following diagram shows that each input words is assigned a weight by the attention mechanism which is then used by the decoder to predict the next word in the sentence. The below picture and formulas are an example of attention mechanism from [Luong's paper](https://arxiv.org/abs/1508.04025v5). 
# 
# <img src="https://www.tensorflow.org/images/seq2seq/attention_mechanism.jpg" width="500" alt="attention mechanism">
# 
# The input is put through an encoder model which gives us the encoder output of shape *(batch_size, max_length, hidden_size)* and the encoder hidden state of shape *(batch_size, hidden_size)*.
# 
# Here are the equations that are implemented:
# 
# <img src="https://www.tensorflow.org/images/seq2seq/attention_equation_0.jpg" alt="attention equation 0" width="800">
# <img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">
# 
# This tutorial uses [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf) for the encoder. Let's decide on notation before writing the simplified form:
# 
# * FC = Fully connected (dense) layer
# * EO = Encoder output
# * H = hidden state
# * X = input to the decoder
# 
# And the pseudo-code:
# 
# * `score = FC(tanh(FC(EO) + FC(H)))`
# * `attention weights = softmax(score, axis = 1)`. Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*, since the shape of score is *(batch_size, max_length, hidden_size)*. `Max_length` is the length of our input. Since we are trying to assign a weight to each input, softmax should be applied on that axis.
# * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
# * `embedding output` = The input to the decoder X is passed through an embedding layer.
# * `merged vector = concat(embedding output, context vector)`
# * This merged vector is then given to the GRU
# 
# The shapes of all the vectors at each step have been specified in the comments in the code:

class Seq2Seq(tf.keras.Model):
  def __init__(self, params):
    super(Seq2Seq, self).__init__()
    self.embedding_matrix = load_embedding_matrix()
    self.encoder = Encoder(params['vocab_size'], 
                           embedding_dim,
                           self.embedding_matrix,
                           params['input_length'],
                           units,
                           BATCH_SIZE)
    self.attention = BahdanauAttention(att_units)
    self.decoder = Decoder(params['vocab_size'], 
                           embedding_dim, 
                           units, 
                           BATCH_SIZE)


  def call_encoder(self, enc_batch):
    # batch_size = enc_batch.shape[0]
    # hidden = [tf.zeros((1 * batch_size, units))]
    enc_hidden = self.encoder.initialize_hidden_state()

    enc_output, enc_hidden = self.encoder(enc_batch, enc_hidden)

    return enc_output, enc_hidden

  def call_decoder_onestep(self, dec_batch, dec_hidden, enc_output):
    context_vector, _ = self.attention(enc_hidden, enc_output)
    pred, state = self.decoder(dec_batch,
                                    dec_hidden, 
                                    enc_output,
                                    context_vector)
                                        
    return pred, state

  def call(self, enc_batch, dec_batch, dec_targets):
    preds = []
    states = []
    enc_output, enc_hidden = self.call_encoder(enc_batch)
    context_vector, _ = self.attention(enc_hidden, enc_output)
    dec_hidden = enc_hidden
    for i in range(1, dec_targets.shape[1]):
      
      pred, state = self.call_decoder_onestep(tf.expand_dims(dec_targets[:, i], 1), 
                                                 dec_hidden, 
                                                enc_output,
                                                context_vector)
      # using teacher forcing
      # dec_batch = tf.expand_dims(dec_targets[:, i], 1)
      preds.append(pred)
      states.append(state)
                                        
    return tf.stacks(preds, 1), state

    # def call_onestep(self, enc_batch, dec_batch, dec_targets):
    # preds = []
    # states = []
    # enc_output, enc_hidden = self.call_encoder(enc_batch)

    # for i in range(1, dec_targets.shape[1]):

    #   pred, state, _ = self.decoder(dec_batch,
    #                                 enc_hidden, 
    #                                 enc_output
    #                                 context_vector)
                                
    #   # context_vector, _ = self.attention(enc_hidden, enc_hidden)
    #   # using teacher forcing
    #   dec_batch = tf.expand_dims(dec_targets[:, i], 1)
    #   preds.append(pred)
    #   states.append(state)
                                        
    # return tf.stacks(preds, 1), state


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size,embedding_dim,embedding_matrix,input_length,enc_units,batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix],trainable=False,input_length=input_length)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    # print("Encoder: ", x.shape, hidden.shape)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))



class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)
    # print(values.shape, hidden_with_time_axis.shape)
    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    # print("2>>>>>>>", hidden.shape, enc_output.shape) #2>>>>>>> (32, 300) (32, 261, 300)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights
  
  def call_onestep(self, x, hidden, enc_output, context_vector):
    # enc_output shape == (batch_size, max_length, hidden_size)
    # print("2>>>>>>>", hidden.shape, enc_output.shape) #2>>>>>>> (32, 300) (32, 261, 300)
    # context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state


