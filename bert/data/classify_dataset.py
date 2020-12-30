from tasks.classify.classify_config import *
from torch.utils.data import Dataset
from bert.common.tokenizers import Tokenizer


class BertClsDataSet(Dataset):
    def __init__(self, train_path, vocab_path=VocabPath, c2n_path=C2NPicklePath):
        # 属性初始化
        self.train_path = train_path
        self.vocab_path = vocab_path
        self.c2n_path = c2n_path
        self.train_data = []
        self.tokenizer = Tokenizer(self.vocab_path)
        with open(self.c2n_path, 'rb') as f:
            self.classes2num = pickle.load(f)

        # 方法初始化
        self.__load_train_data(self.train_path)

    def __load_train_data(self, train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            batch_tmp = []
            for line in f:
                if line:
                    batch_tmp.append(line.strip())
                    if len(batch_tmp) == BatchSize:
                        self.train_data.append(batch_tmp)
                        batch_tmp = []
            if len(batch_tmp) > 0:
                self.train_data.append(batch_tmp)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        instance = {}

        # 获取一个batch
        batch_max = 0
        batch_data = self.train_data[item]
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
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in instance.items()}
        return instance


if __name__ == '__main__':
    tt = BertClsDataSet(train_path='../../data/classify/train_data.txt',
                        vocab_path='../../data/vocab.txt',
                        c2n_path='../../data/classify/classes2num.pickle')
    for x in tt:
        print(x)
