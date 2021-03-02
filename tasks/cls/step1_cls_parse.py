# coding: utf-8

# linux环境使用
# import sys
# sys.path.append('根目录')

from cls_config import *
from bert.common.tokenizers import Tokenizer


tokenizer = Tokenizer(VocabPath)


def parse_data(src_path, train_path, eval_path, c2n_path, cls_count_path):
    src_data = []
    class2num = {}
    class_count = {}
    ft = open(train_path, 'w', encoding='utf-8')
    fe = open(eval_path, 'w', encoding='utf-8')

    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                chars2nums = ['101']

                line = line.strip().split(SegmentChar)
                input = line[1].lower()
                for char in input:
                    chars2nums.append(str(tokenizer.token_to_id(char)))

                label = line[0]
                if label not in class2num:
                    class2num[label] = len(class2num)
                # 统计所有的类别的分部
                if class2num[label] in class_count:
                    class_count[class2num[label]] += 1
                else:
                    class_count[class2num[label]] = 1

                label2num = str(class2num[label])
                input = '[CLS]' + input
                chars2nums = ' '.join(chars2nums)
                src_data.append((label, label2num, input, chars2nums))

    random.shuffle(src_data)
    train_count = int(len(src_data) * TrainRate)
    train_data = src_data[:train_count]
    eval_data = src_data[train_count:]

    with open(c2n_path, 'wb') as f:
        pickle.dump(class2num, f)
    with open(cls_count_path, 'wb') as f:
        pickle.dump(class_count, f)
    for item in train_data:
        ft.write(item[0] + SegmentChar + item[1] + SegmentChar + item[2] + SegmentChar + item[3] + '\n')
    for item in eval_data:
        fe.write(item[0] + SegmentChar + item[1] + SegmentChar + item[2] + SegmentChar + item[3] + '\n')


if __name__ == '__main__':
    parse_data(SourcePath, TrainPath, EvalPath, C2NPicklePath, ClassCountPath)
