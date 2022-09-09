"""
Japanese Retrieval System
Author: Zhao Yuqi
Date: 2022.6.17
"""

import re
from collections import Counter, defaultdict
from itertools import chain

import numpy as np
from fugashi import Tagger
from nltk import RegexpTokenizer


class JpnSearchEngine:
    def __init__(self, file_path, length=3600, allow_single_char=False, length_norm=False):
        self.docs = []
        self.load_data(file_path, length)
        
        self.tagger = Tagger()
        self.rules = [
            r'[a-zA-Z0-9¥"¥.¥,¥@]+',
            r'[!"“#$%&()\*\+\-\.,\/:;<=>?@\[\\\]^_`{|}~]',
            r'[\n|\r|\t|年|月|日]'
        ]

        self.re_rule = u'([ぁ-んー]+|[ァ-ンー]+|[\u4e00-\u9FFF]+|[ぁ-んァ-ンー\u4e00-\u9FFF]+)'
        self.re_tokenizer = RegexpTokenizer(self.re_rule)
        self.ignored = ['非自立', '接尾', '数']

        self.length_norm = length_norm
        self.allow_single_char = allow_single_char
        self.docs_words, self.inverted_idx = self.parse_and_build()

        self.vocab = set(chain(*self.docs_words))
        self.vocab_to_num = {v: i for i, v in enumerate(self.vocab)}
        self.num_to_vocab = {i: v for v, i in self.vocab_to_num.items()}

        self.tf = self.__cal_tf()
        self.idf = self.__cal_idf()
        self.tf_idf = self.tf * self.idf

    def load_data(self, file_path, length):
        lines = []

        with open(file_path, 'r', encoding='utf-8') as f:
            if length == 0:
                lines = f.readlines()
                lines = [''.join([lines[i], lines[i + 1], lines[i + 2]])
                         for i in range(0, len(lines) // 3 * 3, 3)]
            else:
                sentence = ''
                for i in range(length // 3 * 3):
                    sentence += f.readline()
                    if (i + 1) % 3 == 0:
                        lines.append(sentence)
                        sentence = ''

        self.docs = lines

    def filter(self, text):
        txt = text
        for rule in self.rules:
            txt = re.sub(rule, '', txt)

        txt = self.re_tokenizer.tokenize(txt)
        return ''.join(txt)

    def parse(self, text: str):
        word_list = []
        for node in self.tagger.parseToNodeList(text):
            if '名詞' not in node.pos:
                continue

            for ignored in self.ignored:
                if ignored in node.pos:
                    continue

            if not self.allow_single_char:
                if len(node.surface) == 1:
                    continue

            word_list.append(node.surface)
        return word_list

    def parse_and_build(self):
        words_list = []
        inverted_idx = defaultdict(set)

        for i, doc in enumerate(self.docs):
            words = self.parse(self.filter(doc))
            words_list.append(words)
            for word in words:
                inverted_idx[word].add(i)

        return words_list, inverted_idx

    def __cal_tf(self) -> np.ndarray:
        tf = np.zeros((len(self.vocab), len(self.docs)))

        for i, words in enumerate(self.docs_words):
            counter = Counter(words)
            for word in counter.keys():
                tf[self.vocab_to_num[word], i] = counter[word] / len(words)

        return tf

    def __cal_idf(self) -> np.ndarray:
        freq = np.zeros((len(self.num_to_vocab), 1))

        for i in range(len(self.num_to_vocab)):
            count = 0
            for doc_words in self.docs_words:
                if self.num_to_vocab[i] in doc_words:
                    count += 1
            freq[i, 0] = count

        idf = np.log(len(self.docs) / (freq + 1))
        return idf

    def cal_tf_idf(self) -> np.ndarray:
        return self.__cal_tf() * self.__cal_idf()

    @staticmethod
    def cos_sim(docs_tf_idf, text_tf_idf):
        a = text_tf_idf / np.sqrt(np.sum(np.square(text_tf_idf), axis=0, keepdims=True))
        b = docs_tf_idf / np.sqrt(np.sum(np.square(docs_tf_idf), axis=0, keepdims=True))
        return b.T.dot(a).ravel()

    def get_score(self, text):
        text_words = self.parse(self.filter(text))

        unk_word_num = 0
        target_doc_idxes = set()
        target_word_idxes = set()
        for word in set(text_words):
            target_doc_idxes = target_doc_idxes | self.inverted_idx[word]

            if word not in self.vocab_to_num:
                self.vocab_to_num[word] = len(self.vocab_to_num)
                self.num_to_vocab[len(self.vocab_to_num) - 1] = word
                unk_word_num += 1

            target_word_idxes.add(self.vocab_to_num[word])
        if len(target_doc_idxes) == 0:
            return None

        xs, ys = list(target_word_idxes), list(target_doc_idxes)
        idf, tf_idf = self.idf, self.tf_idf
        if unk_word_num > 0:
            idf = np.concatenate((idf, np.zeros((unk_word_num, 1), dtype=np.float64)), axis=0)
            tf_idf = np.concatenate((tf_idf, np.zeros((unk_word_num, self.tf_idf.shape[1]), dtype=np.float64)), axis=0)

        counter = Counter(text_words)
        tf = np.zeros((len(idf), 1))
        for word in counter.keys():
            tf[self.vocab_to_num[word], 0] = counter[word]

        text_tf = tf[xs, :]
        text_idf = idf[xs, :]
        doc_tf_idf = tf_idf[:, ys][xs, :]

        # text_tf_idf = tf * idf
        # scores = self.cos_sim(tf_idf, text_tf_idf)

        text_tf_idf = text_tf * text_idf
        scores = self.cos_sim(doc_tf_idf, text_tf_idf)

        if self.length_norm:
            doc_lens = np.array([len(doc) for doc in self.docs_words])[ys]
            scores = scores / doc_lens

        return list(zip(ys, scores))

    def get_top_n(self, text, n=3):
        scores = self.get_score(text)

        if scores is None:
            print('Oops! No matches.')
            return None

        n = np.minimum(n, len(scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n]

        print('Top {} docs for "{}": \n'.format(n, text))
        top_docs = []
        for i, _ in scores:
            print(self.docs[i])
            top_docs.append(self.docs[i])

        return top_docs


if __name__ == '__main__':
    searcher = JpnSearchEngine('dataset/leads.org.txt')

    target = '宇野浩二'
    searcher.get_top_n(target)
