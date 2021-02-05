# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# linux环境使用
# import sys
# sys.path.append('根目录')

import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score
from tasks.ner.ner_config import *
from bert.models.tasks.BertNer import BertNerBiRnnCrf
from bert.data.ner_dataset import NerDataSet, NerEvalSet


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score


if __name__ == '__main__':
    best_eval_f1 = 0
    dataset = NerDataSet()
    evalset = NerEvalSet()

    # 加载类别映射表
    with open(C2NPicklePath, 'rb') as f:
        class_to_num = pickle.load(f)
    num_to_class = {}
    for k, v in class_to_num.items():
        num_to_class[v] = k

    number_of_categories = len(class_to_num)
    bert_ner = BertNerBiRnnCrf(number_of_categories).to(device)

    if os.path.exists(FinetunePath):
        print('开始加载本地预训练模型！')
        bert_ner.load_finetune(FinetunePath)
        print('完成加载本地预训练模型！\n')
    else:
        print('开始加载外部预训练模型！')
        bert_ner.load_pretrain(PretrainPath)
        print('完成加载外部预训练模型！\n')
    bert_ner = bert_ner.to(device)

    optim = Adam(bert_ner.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        pre_list = []
        gt_list = []

        # train
        bert_ner.train()
        data_iter = tqdm(enumerate(dataset),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataset),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            batch_inputs = data['batch_inputs']
            batch_labels = data['batch_labels']
            batch_segments = data['batch_segments']
            batch_positions = data['batch_positions']

            if IsCrf:
                ner_output = bert_ner(batch_inputs, batch_positions, batch_segments, AttentionMask)
                mask_loss = -1 * bert_ner.crf(emissions=ner_output, tags=batch_labels, mask=batch_segments.to(torch.uint8))
                batch_output = torch.nn.Softmax(dim=-1)(ner_output)
                batch_topk = torch.topk(batch_output, 1).indices.squeeze(-1).tolist()
            else:
                ner_output = bert_ner(batch_inputs, batch_positions, batch_segments, AttentionMask).permute(0, 2, 1)
                mask_loss = criterion(ner_output, batch_labels)
                batch_output = torch.nn.Softmax(dim=-1)(ner_output.permute(0, 2, 1))
                batch_topk = torch.topk(batch_output, 1).indices.squeeze(-1).tolist()

            print_loss = mask_loss.item()
            mask_loss.backward()
            optim.step()
            optim.zero_grad()

            # 收集结果
            gt_list.extend(sum(batch_labels.tolist(), []))
            pre_list.extend(sum(batch_topk, []))

        print('EP_%d mask loss:%s' % (epoch, print_loss))
        cls_f1 = get_f1(gt_list, pre_list)
        print(epoch, 'train-ner f1 is:%s' % cls_f1)

        # eval
        with torch.no_grad():
            bert_ner.eval()
            correct = 0
            total = 0
            pred_list = []
            label_list = []

            for eval_data in evalset:
                total += 1
                eval_label = eval_data['eval_label'].tolist()
                eval_input = eval_data['eval_input'].unsqueeze(0).to(device)
                eval_segment = eval_data['eval_segment'].unsqueeze(0).to(device)
                eval_position = eval_data['eval_position'].unsqueeze(0).to(device)
                eval_output = bert_ner(eval_input, eval_position, eval_segment, AttentionMask)

                if IsCrf:
                    eval_topk = bert_ner.crf.decode(eval_output, eval_segment.to(torch.uint8))[0]
                else:
                    output_tensor = torch.nn.Softmax(dim=-1)(eval_output)
                    eval_topk = torch.topk(output_tensor, 1).indices.squeeze(0).squeeze(-1).tolist()

                pred_list.extend(eval_topk)
                label_list.extend(eval_label)
                # 累计数值
                for x, y in zip(eval_topk, eval_label):
                    if x == y:
                        correct += 1

            acc_rate = float(correct) / float(total)
            acc_rate = round(acc_rate, 2)
            print('验证集正确率：%s' % acc_rate)

            eval_f1 = get_f1(label_list, pred_list)
            print(epoch, 'eval-ner f1 is:%s' % eval_f1)

            # save
            if eval_f1 > best_eval_f1:
                best_eval_f1 = eval_f1
                torch.save(bert_ner.cpu(), FinetunePath)
                bert_ner.to(device)
                print('EP:%d Model Saved on:%s' % (epoch, FinetunePath))

            print('\n')
