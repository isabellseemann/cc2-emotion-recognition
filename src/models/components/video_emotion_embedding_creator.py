from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class AudioVideoEmotionEmbeddingCreator(nn.Module):
    def __init__(self, cnn_out_channels=64, lstm_hidden_size=32, lstm_layers=1, dropout=0.3):
        super(AudioVideoEmotionEmbeddingCreator, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

    def forward(self, features, lengths):
        batch_size, seq_len, feature_dim = features.size()
        features = features.view(batch_size * seq_len, 1, feature_dim)
        features = self.cnn(features)
        features = features.squeeze(-1)
        features = features.view(batch_size, seq_len, -1)
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return hn[-1]
