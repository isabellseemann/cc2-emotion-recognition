import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TrimodalDataset(Dataset):
    def __init__(self, text_features, audio_features, video_features, labels):
        self.text_data = text_features
        self.audio_data = audio_features
        self.video_data = video_features
        self.labels = labels

        assert len(text_features) == len(audio_features) == len(video_features) == len(labels), \
            "All modalities must have the same number of samples"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        audio = self.audio_data[idx]
        padded_audio_features = pad_sequence(audio, batch_first=True, padding_value=0)
        video = self.video_data[idx]
        padded_video_features = pad_sequence(video, batch_first=True, padding_value=0)
        label = self.labels[idx]
        class_index = int(label.argmax())
        class_index = torch.tensor(class_index, dtype=torch.long)

        return {
            'text': {
                "features": text,
            },
            'audio': {
                "features": padded_audio_features,
                "lengths": audio.size(0)
            },
            'video': {
                "features": padded_video_features,
                "lengths": video.size(0)
            },
            'labels': class_index
        }
