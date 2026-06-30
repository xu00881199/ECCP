import json
import os.path
import re
from collections import Counter
import pickle
import torch

with open('./datasets/strip_list.pkl', 'rb') as file:
    strip = pickle.load(file)


class Tokenizer(object):
    def __init__(self, ann_path, threshold, dataset_name, max_length=128):
        self.ann_path = ann_path
        self.threshold = threshold
        self.dataset_name = dataset_name
        self.vocabulary_path = os.path.join("datasets", self.dataset_name + "_vocabulary.pkl")
        self.max_length = max_length

        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        with open(self.ann_path, 'r', encoding='utf-8') as ann_file:
            self.ann = json.load(ann_file)
        if os.path.exists(self.vocabulary_path):
            with open(self.vocabulary_path, "rb") as f:
                self.token2idx, self.idx2token = pickle.load(f)
        else:
            self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        if isinstance(self.ann, list):
            iterator = self.ann
            report_key = 'Finding-English' if self.ann and 'Finding-English' in self.ann[0] else 'report'
        else:
            iterator = self.ann.get('train', [])
            report_key = 'report'

        for example in iterator:
            report = example.get(report_key, '')
            if not isinstance(report, str) or not report.strip():
                continue
            tokens = self.clean_report(report).split()
            for token in tokens:
                total_tokens.append(token)

        total_tokens = [item for item in total_tokens if item not in strip]

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']

        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 3
            idx2token[idx + 3] = token
        with open(self.vocabulary_path, "wb") as f:
            pickle.dump([token2idx, idx2token], f)

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [1] + ids + [2]
        return ids

    # def decode(self, ids):
    #     txt = ''
    #     for i, idx in enumerate(ids):
    #         if idx > 0:
    #             if i >= 1:
    #                 txt += ' '
    #             try:
    #                 txt += self.idx2token[idx]
    #             except:
    #                 txt += self.idx2token[idx.cpu().item()]
    #         else:
    #             break
    #     return txt

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            # 1. 安全地提取数值 (兼容 Tensor 和 int)
            if isinstance(idx, torch.Tensor):
                idx_val = idx.item()
            else:
                idx_val = idx
                
            # 2. 过滤逻辑
            # 假设 0 是 PAD，通常我们需要过滤掉它
            if idx_val > 0:
                # 如果不是第一个词，加空格分隔
                if i >= 1:
                    txt += ' '
                
                # 3. 安全查找字典 (解决 KeyError: 2)
                # 如果 idx2token 里没有这个 id，默认返回 '<unk>'
                # 这样程序就不会因为一个词找不到而崩溃
                word = self.idx2token.get(idx_val, '<unk>')
                txt += word
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def encode(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [1] + ids + [2]
        return ids

    def encode_batch(self, report_batch):
        out = []
        for ids in report_batch:
            out.append(self.encode(ids)[:self.max_length])
        return out


# ffair

# import os
# import json
# import re
# import pickle
# from collections import Counter
# import torch

# # 尝试加载停用词表，如果文件不存在则设为空列表（增强代码健壮性）
# try:
#     with open('./datasets/strip_list.pkl', 'rb') as file:
#         strip = pickle.load(file)
# except FileNotFoundError:
#     print("Warning: ./datasets/strip_list.pkl not found, proceeding without stopword removal.")
#     strip = []


# class Tokenizer(object):
#     def __init__(self, ann_path, threshold, dataset_name, max_length=128):
#         self.ann_path = ann_path
#         self.threshold = threshold
#         self.dataset_name = dataset_name
#         self.vocabulary_path = os.path.join("datasets", self.dataset_name + "_vocabulary.pkl")
#         self.max_length = max_length

#         # 1. 【修改】根据数据集选择对应的清洗函数
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         elif self.dataset_name == 'mimic_cxr':
#             self.clean_report = self.clean_report_mimic_cxr
#         else:
#             # 默认为 FFA-IR 或其他
#             self.clean_report = self.clean_report_ffa_ir

#         # 2. 加载标注文件
#         with open(self.ann_path, 'r', encoding='utf-8') as f:
#             self.ann = json.load(f)

#         # 3. 加载或创建词表
#         if os.path.exists(self.vocabulary_path):
#             with open(self.vocabulary_path, "rb") as f:
#                 self.token2idx, self.idx2token = pickle.load(f)
#         else:
#             self.token2idx, self.idx2token = self.create_vocabulary()

#     def create_vocabulary(self):
#         total_tokens = []

#         # 4. 【修改】根据数据集结构选择迭代方式
#         if self.dataset_name == 'ffa_ir' or isinstance(self.ann, list):
#             # Case A: FFA-IR (Flat List)
#             iterator = self.ann
#             key_name = 'Finding-English' if 'Finding-English' in self.ann[0] else 'report'
#         else:
#             # Case B: MIMIC/IU (Dictionary with 'train')
#             iterator = self.ann.get('train', [])
#             key_name = 'report'

#         print(f"Creating vocabulary from {len(iterator)} samples (dataset: {self.dataset_name})...")

#         for example in iterator:
#             if key_name in example:
#                 tokens = self.clean_report(example[key_name]).split()
#                 for token in tokens:
#                     total_tokens.append(token)

#         # 过滤停用词
#         total_tokens = [item for item in total_tokens if item not in strip]

#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']

#         vocab.sort()
#         token2idx, idx2token = {}, {}
        
#         # 预留特殊 token id (0: pad, 1: bos, 2: eos)
#         # 注意：这里我们显式定义它们，确保 id 对齐
#         token2idx['<pad>'] = 0; idx2token[0] = '<pad>'
#         token2idx['<bos>'] = 1; idx2token[1] = '<bos>'
#         token2idx['<eos>'] = 2; idx2token[2] = '<eos>'

#         # 从 3 开始分配普通词汇
#         current_idx = 3
#         for token in vocab:
#             if token not in token2idx: # 避免覆盖特殊 token
#                 token2idx[token] = current_idx
#                 idx2token[current_idx] = token
#                 current_idx += 1
                
#         with open(self.vocabulary_path, "wb") as f:
#             pickle.dump([token2idx, idx2token], f)

#         return token2idx, idx2token

#     # --- 保持原有的清洗函数 ---
#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     # 5. 【新增】FFA-IR 清洗函数 (通用版)
#     def clean_report_ffa_ir(self, report):
#         if not isinstance(report, str): return ""
        
#         # 转小写
#         report = report.lower()
        
#         # 简单替换常见标点
#         report = report.replace('.', ' . ').replace(',', ' , ').replace('?', ' ? ').replace('!', ' ! ').replace('"', ' " ')
        
#         # 移除非法字符
#         report = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', report)
        
#         # 移除多余空格
#         report = re.sub(r'\s+', ' ', report).strip()
        
#         return report

#     def get_token_by_id(self, id):
#         return self.idx2token[id]

#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]

#     def get_vocab_size(self):
#         return len(self.token2idx)

#     def __call__(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         ids = [1] + ids + [2]
#         return ids

#     def decode(self, ids):
#         txt = ''
#         for i, idx in enumerate(ids):
#             # 处理 Tensor 类型转 int
#             if isinstance(idx, torch.Tensor):
#                 idx = idx.item()
            
#             if idx > 0:
#                 if i >= 1:
#                     txt += ' '
#                 if idx in self.idx2token:
#                     txt += self.idx2token[idx]
#                 else:
#                     txt += '<unk>'
#             else:
#                 break
#         return txt

#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out

#     def encode(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         ids = [1] + ids + [2]
#         return ids

#     def encode_batch(self, report_batch):
#         out = []
#         for ids in report_batch:
#             out.append(self.encode(ids)[:self.max_length])
#         return out
