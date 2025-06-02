import torch
from torch.utils.data import Dataset


class CMUMoseiRawDataset(Dataset):
    def __init__(self, tokenized_data, labels):
        self.tokenized_data = tokenized_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #get the tokenized input and the corresponding label
        input_ids = self.tokenized_data['input_ids'][idx]
        attention_mask = self.tokenized_data['attention_mask'][idx]
        label = self.labels[idx]
        label_index = torch.argmax(torch.tensor(label)).item()

        return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(label_index, dtype=torch.long)
            }
