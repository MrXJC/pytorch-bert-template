# -*- coding:utf-8 -*-


import os

from collections import defaultdict


class Vocab:

    def __init__(self, infile, seg, emb=None, vocab_file='', add_be=True):
        self.infile = infile
        self.seg = seg
        self.word2idx = defaultdict()
        self.idx2word = []

        if emb is not None:
            print('Vocabulary from embedding ')
            self.idx2word = emb.w2v.index2word
            if add_be:
                self.add_BE()
            self.word2idx = dict(zip(self.idx2word, range(len(self.idx2word))))

        else:
            if not os.path.exists(vocab_file):
                print('Vocabulary file not found. Building vocabulary...')
                self.build_vocab()
            else:
                self.idx2word = open(vocab_file).read().strip().split('\n')
                self.word2idx = dict(zip(self.idx2word, range(len(self.idx2word))))

#    @staticmethod
#    def _clean_text(text):
#        """Text filter for Chinese corpus, only keep CN character."""
#        re_non_ch = re.compile(ur'[^\u4e00-\u9fa5]+')
#        text = text.decode('utf-8').strip(' ')
#        text = re_non_ch.sub('', text)
#        return text
    def add_BE(self):
        # add pad word
        self.idx2word.insert(0, '<PAD>')
        self.add_word('UNK')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_vocab(self):
        self.add_word('<PAD>')  # pad index: 0
        for line in open(self.infile, 'r'):
            _, s1, s2, label = line.strip().split('\t')
            # s1, s2 = map(self._clean_text, [s1, s2])
            s1 = self.seg(s1, ifremove=False)['tokens']
            s2 = self.seg(s2, ifremove=False)['tokens']
            for token in s1 + s2:
                # build vocabulary
                self.add_word(token)
        self.add_word('UNK')  # unk index: len(word2idx)-1

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.idx2word[i])
        return tokens

    def __len__(self):
        return len(self.idx2word)


if __name__ == '__main__':
    pass
