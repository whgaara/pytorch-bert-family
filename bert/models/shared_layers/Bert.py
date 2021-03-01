from torch import nn
from bert.models.shared_layers.BertEmbeddings import BertEmbeddings
from bert.models.shared_layers.Transformer import Transformer


class Bert(nn.Module):
    def __init__(self,
                 device,
                 vocab_size,
                 hidden_size,
                 dropout_prob,
                 attention_heads,
                 num_hidden_layers,
                 intermediate_size
                 ):
        super(Bert, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attention_head_num = attention_heads
        self.dropout_prob = dropout_prob
        self.attention_head_size = hidden_size // attention_heads
        self.intermediate_size = intermediate_size

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

    @staticmethod
    def gen_attention_masks(segment_ids):
        return segment_ids[:, None, None, :]

    def forward(self, device, input_token, position_ids, segment_ids, AttentionMask):
        # embedding
        embedding_x = self.bert_emb(input_token, position_ids)
        if AttentionMask:
            attention_mask = self.gen_attention_masks(segment_ids).to(device)
        else:
            attention_mask = None
        transformer_outputs = []

        # transformer
        for i in range(self.num_hidden_layers):
            if i == 0:
                transformer_outputs.append(self.transformer_blocks[i](embedding_x, attention_mask))
            else:
                transformer_outputs.append(self.transformer_blocks[i](transformer_outputs[-1], attention_mask))

        # pool
        output = self.tanh(self.pooler(transformer_outputs[-1][:, 0, :]))

        return output, transformer_outputs
