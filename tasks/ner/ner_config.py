import os
import time
import torch
import pickle
import random

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# 保存最大句长，字符数，类别数
Assistant = 'data/ner/assistant.txt'

# ## 训练调试参数开始 ## #
IsCrf = True
Epochs = 16
DropOut = 0.1
BatchSize = 8
TrainRate = 0.9
HiddenSize = 768
SegmentChar = '	'
LearningRate = 1e-5
AttentionMask = True
HiddenLayerNum = 12
SentenceLength = 512
IntermediateSize = 3072
AttentionHeadNum = 12
# ## 训练调试参数结束 ## #

# ## 模型文件路径 ## #
root = '/'.join(os.getcwd().split('/')[:-2])
UserDict = os.path.join(root, 'data/key.txt')
StopDict = os.path.join(root, 'data/stop.txt')
VocabPath = os.path.join(root, 'data/vocab.txt')
SourcePath = os.path.join(root, 'data/ner/source_data')
TrainPath = os.path.join(root, 'data/ner/train_data.txt')
EvalPath = os.path.join(root, 'data/ner/eval_data.txt')
C2NPicklePath = os.path.join(root, 'data/ner/classes2num.pickle')
FinetunePath = os.path.join(root, 'checkpoint/finetune/ner/bert_cls_%s_%s.model' % (SentenceLength, HiddenLayerNum))
PretrainPath = os.path.join(root, 'checkpoint/pretrain/pytorch_model.bin')

try:
    VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
except:
    VocabSize = 0

local2target_emb = {
    'bert.bert_emb.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'bert.bert_emb.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'bert.bert_emb.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
    'bert.bert_emb.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
}

local2target_transformer = {
    'bert.transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'bert.transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'bert.transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'bert.transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'bert.transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'bert.transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'bert.transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'bert.transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'bert.transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
    'bert.transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
    'bert.transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'bert.transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'bert.transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'bert.transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'bert.transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
    'bert.transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
}


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
