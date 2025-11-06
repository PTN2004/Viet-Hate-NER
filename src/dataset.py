import torch
from torch.utils.data import Dataset


class HateDataset(Dataset):
    def __init__(self, input_texts, input_labels, tokenizer, label2id, max_len=64):
        super().__init__()
        self.tokens = input_texts
        self.labels = input_labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        words = self.tokens[idx]
        labels = self.labels[idx]

        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )

        label_ids = [self.label2id.get(l, 0) for l in labels]
        label_ids = self.pad_and_truncate(label_ids, pad_id=-100)

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

    def pad_and_truncate(self, seq, pad_id):
        """Hàm tiện ích để pad/truncate về max_len"""
        if len(seq) < self.max_len:
            return seq + [pad_id] * (self.max_len - len(seq))
        else:
            return seq[:self.max_len]
