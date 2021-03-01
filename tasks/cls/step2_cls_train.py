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
from bert.data.cls_dataset import BertClsDataSet, BertClsEvalSet
from tasks.cls.cls_config import *
from bert.models.tasks.BertCls import BertClassify


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score


if __name__ == '__main__':
    best_eval_f1 = 0
    dataset = BertClsDataSet()
    evalset = BertClsEvalSet()

    # 加载类别映射表
    with open(C2NPicklePath, 'rb') as f:
        class_to_num = pickle.load(f)
    num_to_class = {}
    for k, v in class_to_num.items():
        num_to_class[v] = k

    number_of_categories = len(class_to_num)
    bert_classify = BertClassify(number_of_categories).to(device)

    if os.path.exists(FinetunePath):
        print('开始加载本地预训练模型！')
        bert_classify.load_finetune(FinetunePath)
        print('完成加载本地预训练模型！\n')
    else:
        print('开始加载外部预训练模型！')
        bert_classify.load_pretrain(PretrainPath)
        print('完成加载外部预训练模型！\n')
    bert_classify = bert_classify.to(device)

    optim = Adam(bert_classify.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        pre_list = []
        gt_list = []

        # train
        bert_classify.train()
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

            batch_output = bert_classify(batch_inputs, batch_positions, batch_segments, AttentionMask)
            mask_loss = criterion(batch_output, batch_labels)
            print_loss = mask_loss.item()

            mask_loss.backward()
            optim.step()
            optim.zero_grad()

            batch_output = torch.nn.Softmax(dim=-1)(batch_output)
            output_topk = torch.topk(batch_output, 1).indices.squeeze(-1).tolist()

            # 收集结果
            gt_list.extend(batch_labels.tolist())
            pre_list.extend(output_topk)

        print('EP_%d mask loss:%s' % (epoch, print_loss))
        cls_f1 = get_f1(gt_list, pre_list)
        print(epoch, 'train-cls f1 is:%s' % cls_f1)

        # eval
        with torch.no_grad():
            bert_classify.eval()
            correct = 0
            total = 0
            pred_list = []
            label_list = []

            for eval_data in evalset:
                total += 1
                label = eval_data['eval_label'].tolist()
                input_token = eval_data['eval_input'].unsqueeze(0).to(device)
                segment_ids = eval_data['eval_segment'].unsqueeze(0).to(device)
                position_ids = eval_data['eval_position'].unsqueeze(0).to(device)
                output = bert_classify(input_token, position_ids, segment_ids, AttentionMask)
                output = torch.nn.Softmax(dim=-1)(output)
                topk = torch.topk(output, 1).indices.squeeze(0).tolist()[0]
                pred_list.append(topk)
                label_list.append(label)
                # 累计数值
                if label == topk:
                    correct += 1

            acc_rate = float(correct) / float(total)
            acc_rate = round(acc_rate, 2)
            print('验证集正确率：%s' % acc_rate)

            eval_f1 = get_f1(label_list, pred_list)
            print(epoch, 'eval-cls f1 is:%s' % eval_f1)

            # save
            if eval_f1 > best_eval_f1:
                best_eval_f1 = eval_f1
                torch.save(bert_classify.cpu(), FinetunePath)
                bert_classify.to(device)
                print('EP:%d Model Saved on:%s' % (epoch, FinetunePath))

            print('\n')
