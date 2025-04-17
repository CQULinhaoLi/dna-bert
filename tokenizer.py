
def kmer_tokenize(sequence, k=6, stride=1):
    return [sequence[i:i+k] for i in range(0,len(sequence) - k + 1, stride)]


def build_kmer_vocab(k=6):
    from itertools import product
    vocab = [''.join(p) for p in product('ACGT', repeat=k)]
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + vocab
    return {kmer: i for i, kmer in enumerate(vocab)}


class KmerTokenizer:
    """
    A tokenizer class for converting DNA sequences into k-mer tokens and encoding them into input IDs and attention masks.
    Attributes:
        vocab (dict): A dictionary mapping k-mer tokens to their corresponding IDs.
        inv_vocab (dict): A reverse mapping from IDs to k-mer tokens.
        k (int): The k-mer size (default is 6).
        pad_token (str): The token used for padding sequences ('[PAD]').
        unk_token (str): The token used for unknown k-mers ('[UNK]').
        cls_token (str): The token used to represent the start of a sequence ('[CLS]').
        sep_token (str): The token used to represent the end of a sequence ('[SEP]').
        mask_token (str): The token used for masked k-mers ('[MASK]').
    Methods:
        encode(sequence, max_length=512):
            Converts a DNA sequence into k-mer tokens, encodes them into input IDs, and generates an attention mask.
            - sequence (str): The DNA sequence to tokenize and encode.
            - max_length (int): The maximum length of the encoded sequence, including special tokens (default is 512).
            - Returns: A tuple (input_ids, attention_mask), where:
                - input_ids (list[int]): The list of token IDs for the sequence.
                - attention_mask (list[int]): The list indicating which tokens are actual data (1) and which are padding (0).
    """
    def __init__(self, vocab_dict, k=6):
        self.vocab = vocab_dict
        self.inv_vocab = {i: kmer for kmer, i in vocab_dict.items()}
        self.k = k
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'

    def encode(self, sequence, max_length=512):
        """
        Encodes a DNA sequence into input IDs and attention masks for model processing.

        Args:
            sequence (str): The DNA sequence to be encoded.
            max_length (int, optional): The maximum length of the encoded sequence, 
                including special tokens. Defaults to 512.

        Returns:
            tuple: A tuple containing:
                - input_ids (list of int): The tokenized representation of the sequence, 
                  including special tokens and padding.
                - attention_mask (list of int): A binary mask indicating which tokens 
                  are actual input (1) and which are padding (0).

        Notes:
            - The sequence is tokenized into k-mers using the `kmer_tokenize` function.
            - Special tokens (`cls_token` and `sep_token`) are added at the beginning 
              and end of the sequence, respectively.
            - If the sequence length exceeds `max_length - 2`, it is truncated to fit 
              within the limit, accounting for the special tokens.
            - Padding is added to ensure the output length matches `max_length`.
            - Unknown tokens are replaced with the ID for `unk_token`.
        """
        kmers = kmer_tokenize(sequence, self.k)
        tokens = [self.cls_token] + kmers[:max_length - 2] + [self.sep_token]
        input_ids = [self.vocab.get(kmer, self.vocab[self.unk_token]) for kmer in tokens]
        padding_length = max_length - len(input_ids)
        input_ids += [self.vocab[self.pad_token]] * padding_length
        attention_mask = [1] * len(tokens) + [0] * padding_length
        return input_ids, attention_mask