# coding: utf-8

# linux环境使用
# import sys
# sys.path.append('根目录')

from ner_config import *


def gen_item(file_path):
    words = []
    label = []
    word_classes = []
    word_nums = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for word_couple in f:
            if word_couple:
                word, label = tuple(word_couple.strip().split(' '))
                x = 1





def parse(src_path, train_path, eval_path, c2n_path):
    f_train = open(train_path, 'w', encoding='utf-8')
    f_eval = open(eval_path, 'w', encoding='utf-8')

    for file in os.listdir(src_path):
        gen_item(os.path.join(src_path, file))


if __name__ == '__main__':
    parse(SourcePath, TrainPath, EvalPath, C2NPicklePath)
