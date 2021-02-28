# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# linux环境使用
# import sys
# sys.path.append('根目录')

from tasks.ner.ner_config import *
from bert.common.tokenizers import Tokenizer


class NerInference(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        with open(C2NPicklePath, 'rb') as f:
            self.class_to_num = pickle.load(f)
        self.num_to_class = {}
        for k, v in self.class_to_num.items():
            self.num_to_class[v] = k
        try:
            self.model = torch.load(FinetunePath).to(device).eval()
        except:
            self.model = torch.load(FinetunePath, map_location='cpu').to(device).eval()
        print('加载模型完成！')

    def parse_inference_text(self, ori_line):
        input_token_ids = [101]
        ori_line = ori_line.strip().replace(' ', '')
        if len(list(ori_line)) > SentenceLength - 1:
            print('文本过长，将截取前%s个字符进行识别！' % (SentenceLength - 1))
            ori_line = ori_line[: SentenceLength - 1]
        for token in list(ori_line):
            id = self.tokenizer.token_to_id(token)
            input_token_ids.append(id)
        position_ids = [n for n, x in enumerate(input_token_ids)]
        segment_ids = [1] * len(input_token_ids)
        return input_token_ids, position_ids, segment_ids

    def inference_single(self, text):
        input_tokens_id, position_ids, segment_ids = self.parse_inference_text(text)
        input_tokens_id = torch.tensor(input_tokens_id).unsqueeze(0).to(device)
        position_ids = torch.tensor(position_ids).unsqueeze(0).to(device)
        segment_ids = torch.tensor(segment_ids).unsqueeze(0).to(device)

        ner_output = self.model(input_tokens_id, position_ids, segment_ids, AttentionMask)
        if IsCrf:
            inference_topk = self.model.crf.decode(ner_output, segment_ids.to(torch.uint8))[0]
        else:
            output_tensor = torch.nn.Softmax(dim=-1)(ner_output)
            inference_topk = torch.topk(output_tensor, 1).indices.squeeze(0).squeeze(-1).tolist()

        word_entities = []
        for num in inference_topk:
            entity = self.num_to_class[num]
            word_entities.append(entity)

        # 实体的第一个字是cls，因此去掉
        return text, word_entities[1:]


if __name__ == '__main__':
    ner = NerInference()
    text, word_entities = ner.inference_single('工艺行业协会要加强政策宣传和技术引导，'
                                               '对已售出的红木家具产品，有严重质量问题的，'
                                               '要主动召回服务，由林达集团投资开发的海渔广场'
                                               '就位于东四环内四方桥南，首期推出精装公寓海洋公'
                                               '寓——但深发展已经成为国内第一个实际盈利的信用卡部门。')
    for i in range(len(text)):
        print(text[i], word_entities[i])
