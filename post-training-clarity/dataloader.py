import tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd

tokenizer = tiktoken.get_encoding("gpt2")

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length, pad_id):
        self.data = pd.read_csv(csv_file)
        self.encoded_text = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        self.encoded_text = [
            encoded_text[:self.max_length] for encoded_text in self.encoded_text
        ]
        self.encoded_text = [
            encoded_text + [pad_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_text
        ]
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_text:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
    def __getitem__(self, index):
        encoded = self.encoded_text[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    def __len__(self):
        return len(self.data)