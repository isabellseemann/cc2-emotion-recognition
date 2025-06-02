import torch
import torch.nn as nn


class GloveEmotionEmbeddingCreator(nn.Module):
    def __init__(self, embedding_dim=300, kernel_sizes=None, num_filters=50):
        super(GloveEmotionEmbeddingCreator, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in kernel_sizes
        ])


    def forward(self, features):
        features = features.unsqueeze(1)
        conv_results = [torch.relu(conv(features)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(result, dim=2)[0] for result in conv_results]
        feature_vector = torch.cat(pooled, dim=1)
        return feature_vector
