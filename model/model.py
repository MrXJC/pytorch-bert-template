from base import BaseModel
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn


class BertOrigin(BaseModel, BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertOrigin, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        output = logits.view(-1, self.num_labels)
        return output

