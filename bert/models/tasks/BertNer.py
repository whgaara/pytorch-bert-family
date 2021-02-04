import torch.nn as nn

from torchcrf import CRF
from tasks.ner.ner_config import *
from bert.models.layers.Bert import Bert
from bert.models.layers.BiGRU import BiGRU
from bert.models.layers.Classify import Classify


class BertNerBiRnnCrf(nn.Module):
    def __init__(self,
                 number_of_categories,
                 hidden=HiddenSize,
                 vocab_size=VocabSize,
                 dropout_prob=DropOut,
                 attention_heads=AttentionHeadNum,
                 num_hidden_layers=HiddenLayerNum,
                 intermediate_size=IntermediateSize
                 ):
        super(BertNerBiRnnCrf, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.dropout_prob = dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.intermediate_size = intermediate_size
        self.number_of_categories = number_of_categories

        # 申明网络
        self.bert = Bert(
            device,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            dropout_prob=self.dropout_prob,
            attention_heads=self.attention_head_num,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size
        )
        self.classify = Classify(self.hidden_size, self.number_of_categories)
        self.bi_gru = BiGRU(self.number_of_categories, self.number_of_categories)
        self.crf = CRF(self.number_of_categories, batch_first=True)

    def load_finetune(self, path=FinetunePath):
        pretrain_model_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(pretrain_model_dict.state_dict())

    def load_pretrain(self, path):
        pretrain_model_dict = torch.load(path)
        finetune_model_dict = self.state_dict()
        new_parameter_dict = {}

        # 加载embedding层参数
        for key in local2target_emb:
            local = key
            target = local2target_emb[key]
            new_parameter_dict[local] = pretrain_model_dict[target]

        # 加载transformerblock层参数
        for i in range(self.num_hidden_layers):
            for key in local2target_transformer:
                local = key % i
                target = local2target_transformer[key] % i
                new_parameter_dict[local] = pretrain_model_dict[target]

        for key, value in new_parameter_dict.items():
            if key in finetune_model_dict:
                finetune_model_dict[key] = new_parameter_dict[key]

        self.load_state_dict(finetune_model_dict)

    def forward(self, input_token, position_ids, segment_ids, attention_mask):
        # bert_output是最后一层经过全连接和激活函数后的结果
        # transformer_outputs是所有transformer层的结果
        bert_output, transformer_outputs = self.bert(input_token, position_ids, segment_ids, attention_mask)

        # 将每个字768维度的特征数转换为实体类别特征数
        output = self.classify(transformer_outputs[-1])
        output = self.bi_gru(output)

        return output
