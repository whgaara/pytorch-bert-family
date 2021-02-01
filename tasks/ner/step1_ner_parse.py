# coding: utf-8

# linux环境使用
# import sys
# sys.path.append('根目录')
import random

from ner_config import *
from bert.common.tokenizers import Tokenizer


tokenizer = Tokenizer(VocabPath)


def parse(src_path, train_path, eval_path, c2n_path):
    wordlabel2num = {
        'ptzf': 0
    }
    f_train = open(train_path, 'w', encoding='utf-8')
    f_eval = open(eval_path, 'w', encoding='utf-8')
    all_items = []

    for file in os.listdir(src_path):
        words = []
        wordnums = []
        labels = []
        labelnums = []
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as f:
            for word_couple in f:
                if word_couple:
                    word, label = tuple(word_couple.strip().split(' '))
                    words.append(word)
                    wordnums.append(tokenizer.token_to_id(word))
                    if label == 'O':
                        label = 'ptzf'
                    if label == 'ptzf':
                        labels.append(label)
                        labelnums.append(wordlabel2num[label])
                    else:
                        labels.append(label)
                        if label not in wordlabel2num:
                            wordlabel2num[label] = len(wordlabel2num)
                        labelnums.append(wordlabel2num[label])
                    all_items.append([words, wordnums, labels, labelnums])
    random.shuffle(all_items)
    train_line = int(len(all_items) * TrainRate)
    train_items = all_items[:train_line]
    eval_items = all_items[train_line:]

    # 写入到文件中



if __name__ == '__main__':
    parse(SourcePath, TrainPath, EvalPath, C2NPicklePath)
