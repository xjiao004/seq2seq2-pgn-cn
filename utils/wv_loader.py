# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import codecs
# 引入日志配置
import logging

from utils.config import embedding_matrix_path, save_vocab_path, save_wv_model_path, embedding_dim


# SENTENCE_START = '<s>'
# SENTENCE_END = '</s>'

# PAD_TOKEN = '<PAD>'
# UNKNOWN_TOKEN = '<UNK>'
# START_DECODING = '<START>'
# STOP_DECODING = '<STOP>'


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'
    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]
    MASK_COUNT = len(MASKS)

    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)
    START_DECODING_INDEX = MASKS.index(START_DECODING)
    STOP_DECODING_INDEX = MASKS.index(STOP_DECODING)

    def __init__(self, vocab_file=save_vocab_path, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        # self.PAD_TOKEN_INDEX = None
        # self.UNKNOWN_TOKEN_INDEX = None
        # self.START_DECODING_INDEX = None
        # self.STOP_DECODING_INDEX = None
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, file_path, vocab_max_size=None):
        """
        读取字典
        :param file_path: 文件路径
        :return: 返回读取后的字典
        """
        vocab = {mask: index
                 for index, mask in enumerate(Vocab.MASKS)}

        reverse_vocab = {index: mask
                         for index, mask in enumerate(Vocab.MASKS)}

        for line in open(file_path, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index > vocab_max_size - Vocab.MASK_COUNT:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index + Vocab.MASK_COUNT
            reverse_vocab[index + Vocab.MASK_COUNT] = word

        # self.PAD_TOKEN_INDEX = vocab[self.PAD_TOKEN]
        # self.UNKNOWN_TOKEN_INDEX = vocab[self.UNKNOWN_TOKEN]
        # self.START_DECODING_INDEX = vocab[self.START_DECODING]
        # self.STOP_DECODING_INDEX = vocab[self.STOP_DECODING]
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


# def get_vocab_size():


def load_embedding_matrix(filepath=embedding_matrix_path, max_vocab_size=50000):
    """
    加载 embedding_matrix_path
    """
    embedding_matrix = np.load(filepath + '.npy')
    flag_matrix = np.zeros_like(embedding_matrix[:Vocab.MASK_COUNT])
    return np.concatenate([flag_matrix, embedding_matrix])[:max_vocab_size]


def load_word2vec_file():
    # 保存词向量模型
    return Word2Vec.load(save_wv_model_path)


if __name__ == '__main__':
    # vocab 对象
    # vocab = Vocab(vocab_path)
    # print(vocab.count)
    print(load_embedding_matrix(max_vocab_size=300))
