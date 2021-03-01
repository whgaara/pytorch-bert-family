import torch.nn as nn

from tasks.cls.cls_config import HiddenSize, SentenceLength, VocabSize


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size=VocabSize, max_len=SentenceLength, hidden_size=HiddenSize, dropout_prob=0.1):
        super(BertEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token, position_ids):
        token_embeddings = self.token_embeddings(input_token)
        position_embeddings = self.position_embeddings(position_ids)
        embedding_x = token_embeddings + position_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x
