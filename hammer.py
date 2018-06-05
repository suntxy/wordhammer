# -*- coding: utf-8 -*-

from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
import jieba
import pickle
from stop_words import chinese_stop_words


class WordHammer:
    def __init__(self):
        self.bi_dict = defaultdict(int)  # bigram dict for two words
        self.tri_dict = defaultdict(int)  # trigram dict for three words
        self.quad_dict = defaultdict(int)  # quadgram dict for four words
        self.vocab_dict = defaultdict(int)  # fir storing different words with their frequencies
        self.bi_prob_dict = OrderedDict()
        self.tri_prob_dict = OrderedDict()
        self.quad_prob_dict = OrderedDict()

    @staticmethod
    def word_cut(sen):
        """
        split the string into word tokens
        :param sen:
        :return: string
        """
        r = jieba.cut(sen)
        temp = []
        for i in r:
            if i not in chinese_stop_words:
                temp.append(i)
        content = " ".join(temp)
        return content

    def loadCorpus(self, corpus):
        content = self.word_cut(corpus)
        token = content.split()
        temp0 = list(ngrams(token, 2)) # token for bigrams
        temp1 = list(ngrams(token, 3)) # tokens for trigrams
        temp2 = list(ngrams(token, 4)) # tokens for quadgrams

        # add new unique words to the vocaulary set if available
        for word in token:
            if word not in self.vocab_dict:
                self.vocab_dict[word] = 1
            else:
                self.vocab_dict[word] += 1

        # count the frequency of the bigram sentences
        for t in temp0:
            sen = ' '.join(t)
            self.bi_dict[sen] += 1

        # count the frequency of the trigram sentences
        for t in temp1:
            sen = ' '.join(t)
            self.tri_dict[sen] += 1

        # count the frequency of the quadgram sentences
        for t in temp2:
            sen = ' '.join(t)
            self.quad_dict[sen] += 1

    def find_Quadgram_Prob(self):
        V = len(self.vocab_dict)

        for quad_sen in self.quad_dict:
            quad_token = quad_sen.split()

            # trigram sentence for key
            tri_sen = ' '.join(quad_token[:3])

            # find the probability, add i smoothing has been used
            prob = (self.quad_dict[quad_sen] + 1) / (self.tri_dict[tri_sen] + V)

            if tri_sen not in self.quad_prob_dict:
                self.quad_prob_dict[tri_sen] = []
                self.quad_prob_dict[tri_sen].append([prob, quad_token[-1]])
            else:
                self.quad_prob_dict[tri_sen].append([prob, quad_token[-1]])
        print('Quad_Prb_dict len:', len(self.quad_prob_dict))

        # sort prob of quad dict.
        self.sortProbWordDict(self.quad_prob_dict)

    def find_Trigram_Prob(self):
        V = len(self.vocab_dict)

        for tri in self.tri_dict:
            # for tri in file:
            tri_token = tri.split()
            # bigram sentence for key
            bi_sen = ' '.join(tri_token[:2])
            # find the probability

            # add i smoothing has been used
            prob = (self.tri_dict[tri] + 1) / (self.bi_dict[bi_sen] + V)

            # tri_prob_dict is a dict of list
            if bi_sen not in self.tri_prob_dict:
                self.tri_prob_dict[bi_sen] = []
                self.tri_prob_dict[bi_sen].append([prob, tri_token[-1]])
            else:
                self.tri_prob_dict[bi_sen].append([prob, tri_token[-1]])
        # print('Tri_Prb_dict len:', len(tri_prob_dict))
        self.sortProbWordDict(self.tri_prob_dict)

    def find_Bigram_Prob(self):
        V = len(self.vocab_dict)
        for bi in self.bi_dict:
            bi_token = bi.split()
            unigram = bi_token[0]

            # add i smoothing has been used
            prob = (self.bi_dict[bi] + 1) / (self.vocab_dict[unigram] + V)

            # bi_prob_dict is a dict of list
            if unigram not in self.bi_prob_dict:
                self.bi_prob_dict[unigram] = []
                self.bi_prob_dict[unigram].append([prob, bi_token[-1]])
            else:
                self.bi_prob_dict[unigram].append([prob, bi_token[-1]])
        self.sortProbWordDict(self.bi_prob_dict)

    @staticmethod
    def sortProbWordDict(prob_dict):
        for key in prob_dict:
            if len(prob_dict[key]) > 1:
                # only at most top 2 most probable words have been taken
                prob_dict[key] = sorted(prob_dict[key], reverse=True)[:2]

    @staticmethod
    def __predict__(sen, prob_dict, rank=5):
        if sen in prob_dict:
            if rank <= len(prob_dict[sen]):
                return prob_dict[sen][:(rank-1)]
                # return prob_dict[sen]
            else:
                return prob_dict[sen]
        else:
            return []

    def doPrediction(self, sen, num=5):
        content = self.word_cut(sen)
        print("content: ", content)
        words = content.split()
        print("words", words)
        k = len(words)
        if k < 1:
            return {}
        if k == 1:
            ret = self.__predict__(content, prob_dict=self.bi_prob_dict, rank=num)
            return drop_duplicate(ret)
        if k == 2:
            # 当输入两次词时，先用trigram, 再用bigram预测.
            ret = self.__predict__(content, prob_dict=self.tri_prob_dict, rank=num)
            j = len(ret)
            if j < num:
                sc = " ".join(words[-1])
                sr = self.__predict__(sc, prob_dict=self.bi_prob_dict, rank=(num-j))
                if len(sr) > 0:
                    ret = ret + sr
            return drop_duplicate(ret)
        if k >= 3:
            # 当输入三个以上词时, 先用quadgram，再用trigram, 再用bigram。
            con = " ".join(words[(k-3):])
            ret = self.__predict__(con, prob_dict=self.quad_prob_dict, rank=num)
            j = len(ret)
            if j < num:
                sc = " ".join(words[(k-2):])
                sr = self.__predict__(sc, prob_dict=self.tri_prob_dict, rank=(num-j))
                if len(sr) > 0:
                    ret = ret + sr
                t = len(ret)
                if t < num:
                    tc = " ".join(words[-1])
                    tr = self.__predict__(tc, prob_dict=self.bi_prob_dict, rank=(num-t))
                    if len(tr) > 0:
                        ret = ret + tr
            return drop_duplicate(ret)

    def save(self):
        with open("gen/bi_dict.pkl", "wb") as f:
            pickle.dump(self.bi_dict, f)
        with open("gen/tri_dict.pkl", "wb") as f:
            pickle.dump(self.tri_dict, f)
        with open("gen/quad_dict.pkl", "wb") as f:
            pickle.dump(self.quad_dict, f)
        with open("gen/vocab_dict.pkl", "wb") as f:
            pickle.dump(self.vocab_dict, f)
        with open("gen/bi_prob_dict.pkl", "wb") as f:
            pickle.dump(self.bi_prob_dict, f)
        with open("gen/tri_prob_dict.pkl", "wb") as f:
            pickle.dump(self.tri_prob_dict, f)
        with open("gen/quad_prob_dict.pkl", "wb") as f:
            pickle.dump(self.quad_prob_dict, f)
        print("Successful save cache.")

    def load(self):
        with open("gen/bi_dict.pkl", "rb") as f:
            self.bi_dict = pickle.load(f)
        with open("gen/tri_dict.pkl", "rb") as f:
            self.tri_dict = pickle.load(f)
        with open("gen/quad_dict.pkl", "rb") as f:
            self.quad_dict = pickle.load(f)
        with open("gen/vocab_dict.pkl", "rb") as f:
            self.vocab_dict = pickle.load(f)
        with open("gen/bi_prob_dict.pkl", "rb") as f:
            self.bi_prob_dict = pickle.load(f)
        with open("gen/tri_prob_dict.pkl", "rb") as f:
            self.tri_prob_dict = pickle.load(f)
        with open("gen/quad_prob_dict.pkl", "rb") as f:
            self.quad_prob_dict = pickle.load(f)
        print("Successful load cache.")


def drop_duplicate(keys):
    """
    对返回的预测词去重，概率取最大值
    :param keys: lists in list
    :return: dict
    """
    ret = {}
    for word in keys:
        if word[1] in ret:
            ret[word[1]] = max(word[0], ret[word[1]])
        else:
            ret[word[1]] = word[0]
    return ret
