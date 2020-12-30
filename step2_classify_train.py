import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from bert.data.classify_dataset import *
from bert.models.tasks.BertClassify import BertClassify


def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score


if __name__ == '__main__':
    best_eval_f1 = 0
    dataset = BertClsDataSet(TrainPath, VocabPath, C2NPicklePath)
    # dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=False)

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
            output_topk = torch.topk(batch_output, 1).indices.squeeze(0).tolist()

            # 收集结果
            gt_list.extend(batch_labels.tolist())
            pre_list.extend([x[0] for x in output_topk])

        print('EP_%d mask loss:%s' % (epoch, print_loss))
        cls_f1 = get_f1(gt_list, pre_list)
        print(epoch, 'classify f1 is:%s' % cls_f1)

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(bert_classify.cpu(), output_path)
        bert_classify.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # # test
        # with torch.no_grad():
        #     bert_classify.eval()
        #     accuracy = 0
        #     recall = 0
        #     entities_count = 0
        #
        #     for test_data in testset:
        #         label2class = []
        #         output2class = []
        #
        #         input_token = test_data['input_tokens_id'].unsqueeze(0).to(device)
        #         segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
        #         input_token_list = input_token.tolist()
        #         input_len = len([x for x in input_token_list[0] if x])
        #         label_list = test_data['input_tokens_class_id'].tolist()[:input_len]
        #         batch_output = bert_classify(input_token, segment_ids)
        #
        #         if IsCrf:
        #             output_topk = bert_classify.crf.decode(batch_output, segment_ids.to(torch.uint8))[0]
        #         else:
        #             batch_output = batch_output[:, :input_len, :]
        #             output_tensor = torch.nn.Softmax(dim=-1)(batch_output)
        #             output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()
        #             output_topk = [x[0] for x in output_topk]
        #
        #         # 累计数值
        #         for i, output in enumerate(output_topk):
        #             output2class.append(num_to_class[output])
        #             label2class.append(num_to_class[label_list[i]])
        #         output_entities = extract_output_entities(output2class)
        #         label_entities = extract_label_entities(label2class)
        #
        #         # 核算结果
        #         entities_count += len(label_entities.keys())
        #         recall_list = []
        #         for out_num in output_entities.keys():
        #             if out_num in label_entities.keys():
        #                 recall_list.append(out_num)
        #         recall += len(recall_list)
        #         for num in recall_list:
        #             if output_entities[num] == label_entities[num]:
        #                 accuracy += 1
        #     if entities_count:
        #         recall_rate = float(recall) / float(entities_count)
        #         recall_rate = round(recall_rate, 4)
        #         print('实体召回率为：%s' % recall_rate)
        #         accuracy_rate = float(accuracy) / float(recall)
        #         accuracy_rate = round(accuracy_rate, 4)
        #         print('实体正确率为：%s\n' % accuracy_rate)
