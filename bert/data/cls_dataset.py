from torch.utils.data import Dataset
from bert.common.tokenizers import Tokenizer
from tasks.classify.classify_config import *


class BertClsDataSet(Dataset):
    def __init__(self):
        # 属性初始化
        self.src_lines = []
        self.tar_lines = []

        with open(TrainPath, 'r', encoding='utf-8') as f:
            batch_tmp = []
            for line in f:
                if line:
                    batch_tmp.append(line.strip())
                    if len(batch_tmp) == BatchSize:
                        self.src_lines.append(batch_tmp)
                        batch_tmp = []
            if len(batch_tmp) > 0:
                self.src_lines.append(batch_tmp)

        # 格式化所有batch数据
        for batch_group in self.src_lines:
            tmp = {
                'batch_inputs': [],
                'batch_labels': [],
                'batch_segments': [],
                'batch_positions': []
            }
            group_max_len = max([len(x[1]) for x in batch_group])
            for batch_item in batch_group:
                batch_item[1] = batch_item[1] + [0] * (group_max_len - len(batch_item[1]))
                batch_item[2] = batch_item[2] + ['ptzf'] * (group_max_len - len(batch_item[2]))
                batch_item[3] = batch_item[3] + [0] * (group_max_len - len(batch_item[3]))
                input_segments_id = [1 if x else 0 for x in batch_item[1]]
                input_positions_id = [x for x in range(len(batch_item[1]))]
                tmp['batch_inputs'].append(batch_item[1])
                tmp['batch_labels'].append(batch_item[3])
                tmp['batch_segments'].append(input_segments_id)
                tmp['batch_positions'].append(input_positions_id)
            tmp = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in tmp.items()}
            self.tar_lines.append(tmp)


    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, item):
        instance = {}

        # 获取一个batch
        batch_max = 0
        batch_data = self.src_lines[item]
        batch_labels = []
        batch_inputs = []
        batch_segments = []
        batch_positions = []
        for line in batch_data:
            line = line.split(SegmentChar)
            label = self.classes2num[line[0]]
            input_tokens = self.tokenizer.tokens_to_ids(['[CLS]'] + list(line[1]))
            if len(input_tokens) > batch_max:
                batch_max = len(input_tokens)
            batch_labels.append(label)
            batch_inputs.append(input_tokens)

        # padding
        for input in batch_inputs:
            if len(input) < batch_max:
                for i in range(batch_max - len(input)):
                    input.append(0)
            segment = [1 if x else 0 for x in input]
            batch_segments.append(segment)
            position = [x for x in range(len(segment))]
            batch_positions.append(position)

        instance['batch_inputs'] = batch_inputs
        instance['batch_labels'] = batch_labels
        instance['batch_segments'] = batch_segments
        instance['batch_positions'] = batch_positions
        instance = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in instance.items()}
        return instance


class BertClsEvalSet(Dataset):
    def __init__(self, eval_path, vocab_path=VocabPath, c2n_path=C2NPicklePath):
        # 属性初始化
        self.eval_path = eval_path
        self.vocab_path = vocab_path
        self.c2n_path = c2n_path
        self.eval_data = []
        self.tokenizer = Tokenizer(self.vocab_path)
        with open(self.c2n_path, 'rb') as f:
            self.classes2num = pickle.load(f)

        # 方法初始化
        self.__load_train_data(self.eval_path)

    def __load_train_data(self, eval_path):
        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    self.eval_data.append(line.strip())

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, item):
        instance = {}

        line = self.eval_data[item]
        line = line.split(SegmentChar)
        label = self.classes2num[line[0]]
        input_tokens = self.tokenizer.tokens_to_ids(['[CLS]'] + list(line[1]))
        segment = [1 for x in input_tokens]
        position = [x for x in range(len(segment))]

        instance['eval_input'] = input_tokens
        instance['eval_label'] = label
        instance['eval_segment'] = segment
        instance['eval_position'] = position
        instance = {k: torch.tensor(v, dtype=torch.long).to(device) for k, v in instance.items()}
        return instance


if __name__ == '__main__':
    tt = BertClsDataSet()
    for x in tt:
        x = 1
    hh = BertClsEvalSet()
    for x in hh:
        y = 1
