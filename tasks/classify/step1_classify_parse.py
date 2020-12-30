from classify_config import *


def parse_data(src_path, train_path, eval_path, c2n_path):
    src_data = []
    class2num = {}
    ft = open(train_path, 'w', encoding='utf-8')
    fe = open(eval_path, 'w', encoding='utf-8')

    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line:
                line = line.strip().split(SegmentChar)
                label = line[0]
                input = line[1]
                if label not in class2num:
                    class2num[label] = len(class2num)
                src_data.append((label, input))

    random.shuffle(src_data)
    train_count = int(len(src_data) * TrainRate)
    train_data = src_data[:train_count]
    eval_data = src_data[train_count:]

    with open(c2n_path, 'wb') as f:
        pickle.dump(class2num, f)
    for item in train_data:
        ft.write(item[0] + SegmentChar + item[1] + '\n')
    for item in eval_data:
        fe.write(item[0] + SegmentChar + item[1] + '\n')


if __name__ == '__main__':
    parse_data(SourcePath, TrainPath, EvalPath, C2NPicklePath)
