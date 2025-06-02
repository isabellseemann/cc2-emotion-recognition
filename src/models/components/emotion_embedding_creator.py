from torch import nn

from src.models.components.classifier import Classifier


class EmotionClassifier(nn.Module):
    def __init__(self, embedding_creator: nn.Module, classifier: Classifier):
        super().__init__()
        self.embedding_creator = embedding_creator
        self.classifier = classifier

    def forward(self, *args, **kwargs):
        embeddings = self.embedding_creator(**kwargs)
        return self.classifier(embeddings)
