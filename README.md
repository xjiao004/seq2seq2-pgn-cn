# Seq2Seq-PGN-cn

## Installation 

conda create -n tfTrain python=3.6

pip install tf-nightly-gpu

pip install -r requirements.txt

## Training

set args.evaluate=False, run:

python train_attention.py 

## Test

set args.evaluate=True, run:

python train_attention.py

## PGN

模型改进加入PGN后，训练：python train_pgn.py

测试即将train_pgn.py中的mode设置为test。

