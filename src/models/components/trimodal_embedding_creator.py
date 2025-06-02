import torch
import torch.nn as nn

from src.models.components.glove_embedding_creator import GloveEmotionEmbeddingCreator
from src.models.components.video_emotion_embedding_creator import AudioVideoEmotionEmbeddingCreator


class TrimodalEmotionEmbeddingCreator(nn.Module):
    def __init__(self, text_model: GloveEmotionEmbeddingCreator, video_model: AudioVideoEmotionEmbeddingCreator,
                 audio_model: AudioVideoEmotionEmbeddingCreator):
        super(TrimodalEmotionEmbeddingCreator, self).__init__()
        self.text_model = text_model
        self.video_model = video_model
        self.audio_model = audio_model
        # self.linear_layer = LazyLinear(out_features=64)

    def forward(self, text, video, audio):
        text_emb = self.text_model(**text)
        visual_emb = self.video_model(**video)
        audio_emb = self.audio_model(**audio)

        #we use the mean to make the vector that is returned smaller
        # early_fusion_concat = torch.cat(
        #     [text["features"].mean(dim=1), video["features"].mean(dim=1), audio["features"].mean(dim=1)], dim=-1)
        # early_fusion_concat = self.linear_layer(early_fusion_concat)
        # fused_embeddings = torch.cat((early_fusion_concat, late_fusion_concat), dim=-1)
        # return torch.cat((early_fusion_concat, late_fusion_concat), dim=-1)

        late_fusion_concat = torch.cat([text_emb, visual_emb, audio_emb], dim=-1)
        return late_fusion_concat
