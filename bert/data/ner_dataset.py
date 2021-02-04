from tasks.ner.ner_config import *
from torch.utils.data import Dataset
from bert.common.tokenizers import Tokenizer


class NerDataSet(Dataset):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.src_lines = []
        self.tar_lines = []

        # 读取训练数据，并按batch分组
        batch_group = []
        with open(TrainPath, 'r', encoding='utf-8') as f:
            for line in f:
                if len(batch_group) == BatchSize:
                    self.src_lines.append(batch_group)
                    batch_group = []
                if line:
                    line = line.strip()
                    items = line.split('\t')
                    input_tokens, input_tokens_id, input_tokens_label, input_tokens_label_id = items
                    if not input_tokens:
                        continue
                    input_tokens_id = [int(x) for x in input_tokens_id.split(' ')]
                    input_tokens_label = input_tokens_label.split(' ')
                    input_tokens_label_id = [int(x) for x in input_tokens_label_id.split(' ')]
                    batch_group.append([input_tokens, input_tokens_id, input_tokens_label, input_tokens_label_id])
            if len(batch_group) > 0:
                self.src_lines.append(batch_group)

        # 格式化所有batch数据
        for batch_group in self.src_lines:
            dict_group = []
            group_max_len = max([len(x[1]) for x in batch_group])
            for batch_item in batch_group:
                batch_item[1] = batch_item[1] + [0] * (group_max_len - len(batch_item[1]))
                batch_item[2] = batch_item[2] + ['ptzf'] * (group_max_len - len(batch_item[2]))
                batch_item[3] = batch_item[3] + [0] * (group_max_len - len(batch_item[3]))
                input_segments_id = [1 if x else 0 for x in batch_item[1]]
                input_positions_id = [x for x in range(len(batch_item[1]))]
                tmp = {
                    'input_tokens_id': batch_item[1],
                    'input_segments_id': input_segments_id,
                    'input_positions_id': input_positions_id,
                    'input_tokens_label_id': batch_item[3]
                }
                tmp = {k: torch.tensor(v, dtype=torch.long) for k, v in tmp.items()}
                dict_group.append(tmp)
            self.tar_lines.append(dict_group)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]


class NerEvalSet(Dataset):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.tar_lines = []

        # 读取训练数据
        with open(TrainPath, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    items = line.split('\t')
                    input_tokens, input_tokens_id, input_tokens_label, input_tokens_label_id = items
                    if not input_tokens:
                        continue
                    input_tokens_id = [int(x) for x in input_tokens_id.split(' ')]
                    input_segments_id = [1 if x else 0 for x in input_tokens_id]
                    input_tokens_label_id = [int(x) for x in input_tokens_label_id.split(' ')]
                    input_positions_id = [x for x in range(len(input_tokens_id))]
                    tmp = {
                        'input_tokens_id': input_tokens_id,
                        'input_segments_id': input_segments_id,
                        'input_positions_id': input_positions_id,
                        'input_tokens_label_id': input_tokens_label_id
                    }
                    tmp = {k: torch.tensor(v, dtype=torch.long) for k, v in tmp.items()}
                    self.tar_lines.append(tmp)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]


if __name__ == '__main__':
    # dataset = NerDataSet()
    # for i, data in enumerate(dataset):
    #     print(data)
    evalset = NerEvalSet()
    for i, data in enumerate(evalset):
        print(data)
