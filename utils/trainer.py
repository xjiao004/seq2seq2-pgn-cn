import tensorflow as tf
from .config import BATCH_SIZE
from .losses import loss_function

@tf.function
def train_step(encoder, decoder, inp, targ, optimizer, start_index):
  loss = 0
  enc_hidden = encoder.initialize_hidden_state()
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([start_index] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


@tf.function
def train_step_pgn(model, inp, targ, optimizer, start_index, params):
  loss = 0
  predictions = []
  attentions = []
  p_gens = []
  coverages = []

  # encoder hidden
  enc_hidden = model.encoder.initialize_hidden_state()

  with tf.GradientTape() as tape:
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(params["batch_size"], params["max_enc_len"]), dtype=tf.int32)

    # encoder
    enc_output, enc_hidden = model.encoder(enc_inp, enc_hidden)

    # context_vector, attention_weights = model.attention(enc_hidden, enc_output, enc_pad_mask)
    context_vector, _, coverage_ret = model.attention(dec_hidden,
                                                     enc_output,
                                                     enc_pad_mask,
                                                     use_coverage,
                                                     prev_coverage)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([start_index] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(dec_input.shape[1]):
      # passing enc_output to the decoder
      # predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      dec_out, dec_hidden = model.decoder(dec_input,
                                          dec_hidden,
                                          enc_output,
                                          context_vector)

      # context_vector, attn, coverage_ret = model.attention(dec_hidden,
      #                                                           enc_output,
      #                                                           enc_pad_mask,
      #                                                           use_coverage,
      #                                                           coverage_ret)
      final_dists, _, attentions, p_gen, coverages = model(enc_inp,
                                                          dec_input,
                                                          extended_enc_input,
                                                          max_oov_len,
                                                          enc_pad_mask=enc_pad_mask,
                                                          use_coverage=params['use_coverage'],
                                                          prev_coverage=None)

      p_gen = model.pointer(context_vector, dec_hidden, tf.squeeze(dec_x, axis=1))
      coverages.append(coverage_ret)
      attentions.append(attn)
      predictions.append(dec_pred)
      p_gens.append(p_gen)
    if self.params["pointer_gen"]:
      final_dists = _calc_final_dist(enc_extended_inp,
                                          predictions,
                                          attentions,
                                          p_gens,
                                          batch_oov_len,
                                          self.params["vocab_size"],
                                          self.params["batch_size"])
      if self.params["mode"] == "train":
        return tf.stack(final_dists, 1), dec_hidden, attentions, tf.stack(p_gens, 1), None
      else:
        return tf.stack(final_dists, 1), dec_hidden, attentions, tf.stack(p_gens, 1), None
    else:
      return tf.stack(predictions, 1), dec_hidden, attentions, None, None

    # p_gen = model.pointer(context_vector, dec_hidden, dec_input)
    loss += loss_function(targ[:, t], predictions)

    # using teacher forcing
    dec_input = tf.expand_dims(targ[:, t], 1)
    
  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss