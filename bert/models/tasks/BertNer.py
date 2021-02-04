import torch.nn as nn

from tasks.classify.classify_config import *
from torchcrf import CRF
from bert.models.layers.Bert import Bert

from layers.layers.BiGRU import BiGRU
from layers.layers.Classify import Classify
from layers.layers.bert.Transformer import Transformer
from layers.layers.bert.BertEmbeddings import BertEmbeddings


class BertNerBiRnnCrf(nn.Module):
    def __init__(self,
                 number_of_categories,
                 vocab_size=VocabSize,
                 hidden=HiddenSize,
                 num_hidden_layers=HiddenLayerNum,
                 attention_heads=AttentionHeadNum,
                 dropout_prob=DropOut,
                 intermediate_size=IntermediateSize
                 ):
        super(BertNerBiRnnCrf, self).__init__()
        self.number_of_categories = number_of_categories
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden // attention_heads
        self.intermediate_size = intermediate_size
        self.batch_size = OceBatchSize + OcnBatchSize + TnewsBatchSize

        # 申明网络
        self.bert_emb = BertEmbeddings()
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(device)
            for _ in range(self.num_hidden_layers)
        )
        self.pooler = nn.Linear(self.hidden_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.classify = Classify(self.hidden_size, self.vocab_size)
        self.bi_gru = BiGRU(self.number_of_categories, self.number_of_categories)
        self.crf = CRF(self.number_of_categories, batch_first=True)

    @staticmethod
    def gen_attention_masks(segment_ids):
        return segment_ids[:, None, None, :]

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
                if key == 'bert_emb.token_embeddings.weight':
                    finetune_model_dict[key] = torch.cat(
                        [new_parameter_dict[key][:21128].to(device), finetune_model_dict[key][21128:].to(device)])
                else:
                    finetune_model_dict[key] = new_parameter_dict[key]
        self.load_state_dict(finetune_model_dict)

    def forward(self, input_token, position_ids, segment_ids, oce_end_id, ocn_end_id, tnews_end_id):
        # embedding
        embedding_x = self.bert_emb(input_token, position_ids)
        if AttentionMask:
            attention_mask = self.gen_attention_masks(segment_ids).to(device)
        else:
            attention_mask = None
        feedforward_x = None

        # transformer
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.transformer_blocks[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.transformer_blocks[i](feedforward_x, attention_mask)

        # classify
        feedforward_pooler = self.tanh(self.pooler(feedforward_x))

        # ner
        output = self.mlm(feedforward_pooler)

        # bi-rnn
        output = self.bi_gru(output)

        return output
