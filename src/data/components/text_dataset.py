import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, glove_data, labels, max_len=50, embedding_dim=300):
        self.glove_data = glove_data
        self.labels = labels
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieve GloVe word vectors for the sentence
        sentence_embedding = self.glove_data[idx]  # Shape: (sentence_length, embedding_dim)

        # Pad or truncate the sentence embedding to max_len
        if len(sentence_embedding) > self.max_len:
            sentence_embedding = sentence_embedding[:self.max_len]  # Truncate
        else:
            # Pad with zeros if the sequence is shorter than max_len
            padding = torch.zeros((self.max_len - len(sentence_embedding), self.embedding_dim), dtype=torch.float32)
            sentence_embedding = torch.cat((torch.tensor(sentence_embedding, dtype=torch.float32), padding,), dim=0)

        # Retrieve the label corresponding to the sentence
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)
        class_index = int(label.argmax())
        class_index = torch.tensor(class_index, dtype=torch.long)

        return {
            'features': sentence_embedding,
            'labels': class_index,
        }
