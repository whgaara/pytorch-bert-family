# coding: utf-8

# linux环境使用
# import sys
# sys.path.append('根目录')

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
        words = ['[CLS]']
        wordnums = ['101']
        labels = ['ptzf']
        labelnums = ['0']
        with open(os.path.join(src_path, file), 'r', encoding='utf-8') as f:
            for word_couple in f:
                if word_couple:
                    try:
                        char, label = tuple(word_couple.strip().split(' '))
                    except:
                        continue
                    if char == 'END':
                        continue
                    words.append(char)
                    wordnums.append(str(tokenizer.token_to_id(char)))
                    if label == 'O':
                        label = 'ptzf'
                    if label == 'ptzf':
                        labels.append(label)
                        labelnums.append(str(wordlabel2num[label]))
                    else:
                        labels.append(label)
                        if label not in wordlabel2num:
                            wordlabel2num[label] = len(wordlabel2num)
                        labelnums.append(str(wordlabel2num[label]))
        # 过滤过长语句
        if len(words) > SentenceLength:
            print('当前语句过长！语句内容为：%s!' % ''.join(words))
            continue
        all_items.append([words, wordnums, labels, labelnums])
    random.shuffle(all_items)
    train_line = int(len(all_items) * TrainRate)
    train_items = all_items[:train_line]
    eval_items = all_items[train_line:]

    # 写入到文件中
    with open(c2n_path, 'wb') as f:
        pickle.dump(wordlabel2num, f)
    for item in train_items:
        sentence = ''.join(item[0])
        wordnums = ' '.join(item[1])
        labels = ' '.join(item[2])
        labelnums = ' '.join(item[3])
        f_train.write(sentence + SegmentChar + wordnums + SegmentChar + labels + SegmentChar + labelnums + '\n')
    for item in eval_items:
        sentence = ''.join(item[0])
        wordnums = ' '.join(item[1])
        labels = ' '.join(item[2])
        labelnums = ' '.join(item[3])
        f_eval.write(sentence + SegmentChar + wordnums + SegmentChar + labels + SegmentChar + labelnums + '\n')


if __name__ == '__main__':
    parse(SourcePath, TrainPath, EvalPath, C2NPicklePath)
