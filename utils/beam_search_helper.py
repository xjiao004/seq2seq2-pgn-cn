#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: beam_search_helper.py 
@time: 2020-02-16 11:33
@description:
"""
import tensorflow as tf


def merge_batch_beam(t: tf.Tensor):
    # 输入: tensor of shape [batch_size, beam_size ...]
    # 输出: tensor of shape [batch_size * beam_size, ...]
    batch_size, beam_size = t.shape[0], t.shape[1]
    return tf.reshape(t, shape=[batch_size * beam_size] + list(t.shape[2:]))


def split_batch_beam(t: tf.Tensor, beam_size: int):
    # 输入: tensor of shape [batch_size * beam_size ...]
    # 输出: tensor of shape [batch_size, beam_size, ...]
    return tf.reshape(t, shape=[-1, beam_size] + list(t.shape[1:]))


def tile_beam(t: tf.Tensor, beam_size: int):
    # 输入: tensor of shape [batch_size, ...]
    # 输出: tensor of shape [batch_size, beam_size, ...]
    multipliers = [1, beam_size] + [1] * (t.shape.ndims - 1)
    return tf.tile(tf.expand_dims(t, axis=1), multipliers)


class Hypothesis(object):
    def __init__(self, tokens, log_probs, hidden):
        # all tokens from time step 0 to the current time step t
        self.tokens = tokens
        # log probabilities of all tokens
        self.log_probs = log_probs
        # decoder hidden state after the last token decoding
        self.hidden = hidden

    def extend(self, token, log_prob, hidden):
        """
        Method to extend the current hypothesis by adding the next decoded token and
        all information associated with it
        """
        tokens = self.tokens + [token]
        log_probs = self.log_probs + [log_prob]
        return Hypothesis(tokens, log_probs, hidden)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_probs(self):
        return sum(self.log_probs)

    @property
    def avg_log_probs(self):
        return self.tot_log_probs / len(self.log_probs)


def decode_one_topk(decoder_onestep, dec_input, dec_hidden,
                    enc_output,
                    k: int = 1):
    # 单步解码
    pred, dec_hidden = decoder_onestep(dec_input, dec_hidden, enc_output)
    # 计算top-K概率/logits和对应的indice
    top_k_probs, top_k_ids = tf.nn.top_k(pred, k, sorted=True)
    # 计算log概率/logits
    top_k_log_probs = tf.math.log(top_k_probs)
    # 返回结果
    return top_k_log_probs, top_k_ids, dec_hidden



def beam_decode(model, test_batch, vocab, reverse_vocab, params):
    # 初始化mask
    start_index = vocab['<START>']
    stop_index = vocab['<STOP>']
    batch_size = len(test_batch)
    # print("test batch size: ", batch_size)

    enc_input = tf.convert_to_tensor(test_batch)
    dec_input = tf.expand_dims([start_index] * batch_size, 1)
    min_steps = params['min_dec_steps']
    max_steps = params['max_dec_steps']
    beam_size = params['beam_size']
    end_token = stop_index

    # 编码器输出
    # enc_output: [batch_size, sequence_length, enc_units]
    # enc_hidden: [batch_size, enc_units]
    enc_output, enc_hidden = model.call_encoder(enc_input)

    # 将编码器输出复制beam_size份
    # 并调整维度为[beam_size*batch_size, ...]
    enc_output = merge_batch_beam(tile_beam(enc_output, beam_size))

    # 复制隐层状态
    dec_hidden = enc_hidden

    # 初始化[batch_size, beam_size]个Hypothesis对象
    hyps = [[Hypothesis(tokens=list(dec_input[i].numpy()), log_probs=[0.], hidden=dec_hidden[i])
             for _ in range(beam_size)] for i in range(batch_size)]

    # 进行搜索
    for step in range(max_steps):
        # 获得上一步的输出: [batch_size, beam_size]
        latest_tokens = tf.stack(
            [tf.stack([h.latest_token for h in beam], axis=0) for beam in hyps],
            axis=0
        )
        # 构建解码器单步输入: [batch_size*beam_size, 1]
        dec_input = tf.expand_dims(merge_batch_beam(latest_tokens), axis=1)

        # 获得上一步的隐层: [batch_size, beam_size, dec_units]
        hiddens = tf.stack(
            [tf.stack([h.hidden for h in beam], axis=0) for beam in hyps],
            axis=0
        )

        # 构建解码器隐层[batch_size*beam_size, dec_units]
        dec_hidden = merge_batch_beam(hiddens)

        # 单步解码
        top_k_log_probs, top_k_ids, dec_hidden = \
            decode_one_topk(model.call_decoder_onestep, dec_input, dec_hidden, enc_output, k=beam_size)

        # 将上述结果形状变为[batch_size, beam_size, ...]
        top_k_log_probs = split_batch_beam(top_k_log_probs, beam_size)
        top_k_ids = split_batch_beam(top_k_ids, beam_size)
        dec_hidden = split_batch_beam(dec_hidden, beam_size)

        # 遍历batch中所有句子:
        for bc in range(batch_size):
            # 当前句子对应的变量
            bc_hyps = hyps[bc] # bc=1
            bc_top_k_log_probs = top_k_log_probs[bc]
            bc_top_k_ids = top_k_ids[bc]
            bc_dec_hidden = dec_hidden[bc]

            # 遍历上一步中所有的假设情况: beam_size个
            # 获得当前步骤的最大概率假设: beam_size * k个 (k = beam_size)
            bc_all_hyps = []
            num_prev_bc_hyps = 1 if step == 0 else len(bc_hyps)
            for i in range(num_prev_bc_hyps):
                hyp, new_hidden = bc_hyps[i], bc_dec_hidden[i]
                # 分裂，增加当前步中的beam_size * k个可能假设 (k = beam_size)
                for j in range(beam_size):
                    new_hyp = hyp.extend(token=bc_top_k_ids[i, j].numpy(),
                                         log_prob=bc_top_k_log_probs[i, j].numpy(),
                                         hidden=new_hidden)
                    bc_all_hyps.append(new_hyp)

            # 重置当前句子对应的Hypothesis对象列表
            bc_hyps = []

            # 按照概率排序
            sorted_bc_hyps = sorted(bc_all_hyps, key=lambda h: h.avg_log_probs, reverse=True)

            # 筛选top-'beam_size'句话
            for h in sorted_bc_hyps:
                bc_hyps.append(h)
                if len(bc_hyps) == beam_size:
                    # 假设句子数目达到beam_size, 则不再添加
                    break

            # 更新hyps
            hyps[bc] = bc_hyps

    # print(len(hyps))
    # print(len(hyps[0]))


    # 从获得的假设集中取出最终结果
    results = [[]] * batch_size
    #
    # 遍历所有句子
    for bc in range(batch_size):
        # 当前句子对应的变量
        bc_hyps = hyps[bc]
        # 优先选取有结束符的结果
        for i in range(beam_size):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if end_token in tokens:
                tokens = tokens[1:tokens.index(end_token)]
                # 有结束符且满足最小长度要求
                if len(tokens) > min_steps:
                    results[bc] = tokens
                    break
        # 如果找到了满足要求的结果，则直接处理下一句
        if results[bc]:
            continue
        # 若在上述条件下未找到合适结果，则只找没有结束符的结果
        for i in range(beam_size):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if end_token in tokens:
                continue
            results[bc] = tokens[1:]

    def get_abstarct(tokens):
        return " ".join([reverse_vocab[index] for index in tokens])
    # 返回结果
    return [get_abstarct(res) for res in results]
    # return results