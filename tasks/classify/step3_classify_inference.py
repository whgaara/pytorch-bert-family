# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tqdm import tqdm
from tasks.classify.classify_config import *
from bert.common.tokenizers import Tokenizer


class Inference(object):
    def __init__(self, model_path):
        print('开始加载模型！')
        self.model = torch.load(model_path).to(device).eval()
        print('完成加载模型！')
        self.tokenizer = Tokenizer(VocabPath)
        with open(C2NPicklePath, 'rb') as f:
            self.classes2num = pickle.load(f)
            self.num2classes = {}
            for x, y in self.classes2num.items():
                self.num2classes[y] = x

    def inference_single(self, input):
        if input:
            current_words = list(input.lower())
            current_words = ['[CLS]'] + current_words
            tokens_id = self.tokenizer.tokens_to_ids(current_words)
            input_token = torch.tensor(tokens_id, dtype=torch.long).unsqueeze(0).to(device)
            position_ids = [i for i in range(len(tokens_id))]
            position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0).to(device)
            segment_ids = torch.tensor([1 if x else 0 for x in tokens_id], dtype=torch.long).unsqueeze(0).to(device)
            output = self.model(input_token, position_ids, segment_ids, AttentionMask)
            output = torch.nn.Softmax(dim=-1)(output)
            output_prob = round(torch.topk(output, 1).values.squeeze(0).tolist()[0], 4)
            output_num = torch.topk(output, 1).indices.squeeze(0).tolist()[0]
            current_label = self.num2classes[output_num]
            print('"%s"的疾病类别是：%s' % (input, current_label))
            return current_label, output_prob
        else:
            print('您的输入有异常！')
            return None, None

    def inference_group(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line:
                    label, prob = self.inference_single(line.strip())


if __name__ == '__main__':
    bert_infer = Inference(FinetunePath)
    bert_infer.inference_single('血压很高')
