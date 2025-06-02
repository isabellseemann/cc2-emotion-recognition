from torch import nn
from transformers import BertModel

from src.models.components.classifier import Classifier


class BERTEmotionClassifier(nn.Module):

    def __init__(self, num_labels=7, dropout_value=0.3, hidden_size=128):
        super(BERTEmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.requires_grad_(False)
        self.classifier = Classifier(self.bert.config.hidden_size, num_labels, dropout_value, hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
