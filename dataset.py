import torch
from torch.utils.data import Dataset, DataLoader
import random
class DNABERTDataset(Dataset):
    """
    A PyTorch Dataset class for preparing DNA sequences for BERT-style pretraining.
    This dataset reads DNA sequences from a file, tokenizes them, and applies masking
    to a portion of the sequence for masked language modeling (MLM) training.
    Attributes:
        sequences (list): A list of DNA sequences read from the input file.
        tokenizer (object): A tokenizer object used to encode DNA sequences.
        max_length (int): The maximum sequence length for tokenization.
    Args:
        corpus_path (str): Path to the file containing DNA sequences, one per line.
        tokenizer (object): Tokenizer object with `encode` and `vocab` attributes.
        max_length (int, optional): Maximum sequence length for tokenization. Default is 512.
    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Retrieves the tokenized and masked sequence at the given index.
    Returns:
        dict: A dictionary containing:
            - 'input_ids' (torch.Tensor): Tokenized input IDs with masking applied.
            - 'attention_mask' (torch.Tensor): Attention mask for the input IDs.
            - 'labels' (torch.Tensor): Original tokenized input IDs (used as labels for MLM).
    """
    def __init__(self, corpus_path, tokenizer, max_length=512):
        self.sequences = open(corpus_path).read().strip().split('\n')
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids, attention_mask = self.tokenizer.encode(seq, self.max_length)
        labels = input_ids.copy()
        seq_len = len(labels)

        num_to_mask = max(1, int(0.15 * seq_len))
        start_idx = random.randint(1, seq_len - num_to_mask - 1)
        end_idx = start_idx + num_to_mask
        for i in range(start_idx, end_idx):
            input_ids[i] = self.tokenizer.vocab[self.tokenizer.mask_token]

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': torch.tensor(labels),
        }


