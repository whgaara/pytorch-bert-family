# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import json
import torch
import pickle

from tqdm import tqdm
from classify_config import *
from bert.common.tokenizers import Tokenizer


class Inference(object):
    def __init__(self):
        self.fr_oce = open(OceTestPath, 'r', encoding='utf-8')
        self.fr_ocn = open(OcnTestPath, 'r', encoding='utf-8')
        self.fr_tnews = open(TnewsTestPath, 'r', encoding='utf-8')
        self.fw_oce = open('submit/ocemotion_predict.json', 'w', encoding='utf-8')
        self.fw_ocn = open('submit/ocnli_predict.json', 'w', encoding='utf-8')
        self.fw_tnews = open('submit/tnews_predict.json', 'w', encoding='utf-8')
        self.model = torch.load(FinetunePath).to(device).eval()
        print('加载模型完成！')

        self.tokenizer = Tokenizer(CharsVocabPath)

        with open(C2NPicklePath, 'rb') as f:
            self.classes2num = pickle.load(f)
            self.num2classes = {}
            for x, y in self.classes2num.items():
                self.num2classes[y] = x

        with open(EmojiPicklePath, 'rb') as f:
            self.emoji_dict = pickle.load(f)

    def inference_submit(self):
        # 预测oce部分
        for line in tqdm(self.fr_oce):
            if line:
                line = line.strip().split('\t')
                num = line[0]
                input = line[1]
                # current_words = input.split(' ')
                current_words = list(input)
                current_words = ['[CLS]'] + current_words + ['[SEP]']
                tokens_id = self.tokenizer.tokens_to_ids(current_words)
                input_token = torch.tensor(tokens_id, dtype=torch.long).unsqueeze(0).to(device)
                position_ids = [i for i in range(len(tokens_id))]
                position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0).to(device)
                segment_ids = torch.tensor([1 if x else 0 for x in tokens_id], dtype=torch.long).unsqueeze(0).to(device)
                oce_output, _, _ = self.model(
                    input_token,
                    position_ids,
                    segment_ids,
                    1,
                    0,
                    0
                )
                oce_output = torch.nn.Softmax(dim=-1)(oce_output)
                oce_topk = torch.topk(oce_output, 1).indices.squeeze(0).tolist()[0]
                current_label = self.num2classes[oce_topk]
                submit = {"id": str(num), "label": str(current_label)}
                self.fw_oce.write(json.dumps(submit) + '\n')

        # 预测ocn部分
        for line in tqdm(self.fr_ocn):
            if line:
                line = line.strip().split('\t')
                num = line[0]
                input1 = line[1]
                input2 = line[2]
                # current_words = input.split(' ')
                input1 = list(input1)
                input2 = list(input2)
                current_words = ['[CLS]'] + input1 + ['[SEP]'] + input2 + ['[SEP]']
                tokens_id = self.tokenizer.tokens_to_ids(current_words)
                input_token = torch.tensor(tokens_id, dtype=torch.long).unsqueeze(0).to(device)
                position_ids = [i for i in range(len(tokens_id))]
                position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0).to(device)
                segment_ids = torch.tensor([1 if x else 0 for x in tokens_id], dtype=torch.long).unsqueeze(0).to(device)
                _, ocn_output, _ = self.model(
                    input_token,
                    position_ids,
                    segment_ids,
                    0,
                    1,
                    0
                )
                ocn_output = torch.nn.Softmax(dim=-1)(ocn_output)
                ocn_topk = torch.topk(ocn_output, 1).indices.squeeze(0).tolist()[0]
                current_label = self.num2classes[ocn_topk + 7]
                submit = {"id": str(num), "label": str(current_label)}
                self.fw_ocn.write(json.dumps(submit) + '\n')

        # 预测tnews部分
        for line in tqdm(self.fr_tnews):
            if line:
                line = line.strip().split('\t')
                num = line[0]
                input = line[1]
                current_words = list(input)
                # current_words = input.split(' ')
                current_words = ['[CLS]'] + current_words + ['[SEP]']
                tokens_id = self.tokenizer.tokens_to_ids(current_words)
                input_token = torch.tensor(tokens_id, dtype=torch.long).unsqueeze(0).to(device)
                position_ids = [i for i in range(len(tokens_id))]
                position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0).to(device)
                segment_ids = torch.tensor([1 if x else 0 for x in tokens_id], dtype=torch.long).unsqueeze(
                    0).to(device)
                _, _, tnews_output = self.model(
                    input_token,
                    position_ids,
                    segment_ids,
                    0,
                    0,
                    1
                )
                tnews_output = torch.nn.Softmax(dim=-1)(tnews_output)
                tnews_topk = torch.topk(tnews_output, 1).indices.squeeze(0).tolist()[0]
                current_label = self.num2classes[tnews_topk + 10]
                submit = {"id": str(num), "label": str(current_label)}
                self.fw_tnews.write(json.dumps(submit) + '\n')


if __name__ == '__main__':
    bert_infer = Inference()
    bert_infer.inference_submit()
