# -*- coding:utf-8 -*-
# Created by LuoJie at 12/7/19
from pgn_tf2.eval import evaluate
from pgn_tf2.test import test_and_save, predict_result
from pgn_tf2.train import train
import argparse

from utils.config import *
from utils.file_utils import get_result_filename
import os


def main():
    # 获得参数
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default='train', help="run mode", type=str)
    parser.add_argument("--max_enc_len", default=400, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=100, help="Decoder input max sequence length", type=int)
    parser.add_argument("--batch_size", default=batch_size, help="batch size", type=int)
    parser.add_argument("--epochs", default=epochs, help="train epochs", type=int)
    parser.add_argument("--vocab_path", default=save_vocab_path, help="vocab path", type=str)
    parser.add_argument("--learning_rate", default=0.1, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument('--rand_unif_init_mag', default=0.02,
                        help='magnitude for lstm cells random uniform inititalization', type=float)
    parser.add_argument('--eps', default=1e-12, help='eps',
                        type=float)

    parser.add_argument('--trunc_norm_init_std', default=1e-4, help='std of trunc norm init, '
                                                                    'used for initializing everything else',
                        type=float)

    parser.add_argument('--cov_loss_wt', default=1.0, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)

    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)

    parser.add_argument("--vocab_size", default=50000, help="max vocab size , None-> Max ", type=int)
    parser.add_argument("--max_vocab_size", default=50000, help="max vocab size , None-> Max ", type=int)

    parser.add_argument("--beam_size", default=batch_size,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--embed_size", default=300, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=128, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward \
                            result dimension - this result is used to compute the attention weights",
                        type=int)

    parser.add_argument("--train_seg_x_dir", default=train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=train_y_seg_path, help="train_seg_y_dir", type=str)

    parser.add_argument("--val_seg_x_dir", default=val_x_seg_path, help="val_x_seg_path", type=str)
    parser.add_argument("--val_seg_y_dir", default=val_y_seg_path, help="val_y_seg_path", type=str)

    parser.add_argument("--test_seg_x_dir", default=test_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--checkpoint_dir", default=checkpoint_dir,
                        help="checkpoint_dir",
                        type=str)
    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)

    parser.add_argument("--max_train_steps", default=500000 / (batch_size / 8), help="max_train_steps", type=int)
    # parser.add_argument("--max_train_steps", default=50, help="max_train_steps", type=int)
    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle",
                        type=bool)
    parser.add_argument("--test_save_dir", default=save_result_dir, help="test_save_dir", type=str)
    parser.add_argument("--pointer_gen", default=True, help="training, eval or test options", type=bool)
    parser.add_argument("--use_coverage", default=True, help="test_save_dir", type=bool)

    parser.add_argument("--greedy_decode", default=False, help="greedy_decode", type=bool)
    parser.add_argument("--result_save_path", default=get_result_filename(batch_size, epochs, 200, 300),
                        help='result_save_path', type=str)

    parser.add_argument("--max_num_to_eval", default=5, help="max_num_to_eval", type=int)
    parser.add_argument("--num_to_test", default=20000, help="num_to_test", type=int)
    parser.add_argument("--gpu_memory", default=6, help="gpu_memory GB", type=int)

    args = parser.parse_args()
    params = vars(args)
    # print(params)
    if params["mode"] == "train":
        train(params)
    elif params["mode"] == "test":
        params['beam_size'] = 2
        params['batch_size'] = 2
        result_save_path = params['result_save_path']
        predict_result(params, result_save_path)
        # test_and_save(params)
    elif params["mode"] == "eval":
        evaluate(params)
    elif params['mode'] == 'auto':
        # PGN training
        params['mode'] = 'train'

        # params['use_coverage'] = False
        # params['epochs'] = 30
        params['use_coverage'] = True
        params['epochs'] = 30

        train(params)
        # predict result
        params['mode'] = 'test'
        params['beam_size'] = 2
        params['batch_size'] = 2
        result_save_path = params['result_save_path']
        predict_result(params, result_save_path)
        # evaluate
        params['mode'] = 'eval'
        evaluate(params)


if __name__ == '__main__':
    main()
