from base import BaseBertProcessor


class ATECProcessor(BaseBertProcessor):
    def __init__(self, logger, config, data_name, data_path, bert_vocab_file, max_len=50, query_max_len=20,
                 target_max_len=20, do_lower_case=True, test_split=0.0, training=True):
        self.skip_row = 0
        super().__init__(logger, config, data_name, data_path, bert_vocab_file, max_len, query_max_len,
                 target_max_len, do_lower_case, test_split, training)

    def get_labels(self):
        """See base class."""
        return [u'0', u'1']

    def split_line(self, line):
        line = line.strip().split('\t')
        q, t, label = line[1], line[2], line[-1]
        return q, t, label


class CnewsProcessor(BaseBertProcessor):
    def __init__(self, logger, config, data_name, data_path, bert_vocab_file, max_len=50, query_max_len=20,
                 target_max_len=20, do_lower_case=True, test_split=0.0, training=True):
        self.skip_row = 1
        super().__init__(logger, config, data_name, data_path, bert_vocab_file, max_len, query_max_len,
                 target_max_len, do_lower_case, test_split, training)

    def get_labels(self):
        """See base class."""
        return [u'房产', u'科技', u'财经', u'游戏', u'娱乐', u'时尚', u'时政', u'家居', u'教育', u'体育']

    def split_line(self, line):
        line = line.strip().split('\t')
        q, label = line[0], line[-1]
        t = None
        return q, t, label