from transformers import BertConfig, BertForMaskedLM

def get_dnabert_model(vocab_dict, CFG):
    """
    Creates and returns a DNA-BERT model configured for masked language modeling.

    Args:
        vocab_dict (dict): A dictionary representing the vocabulary, where keys are tokens
            and values are their corresponding indices.
        CFG (object): A configuration object containing the following attributes:
            - hidden_size (int): The size of the hidden layers in the BERT model.
            - num_hidden_layers (int): The number of hidden layers in the BERT model.
            - num_attention_heads (int): The number of attention heads in the BERT model.
            - intermediate_size (int): The size of the intermediate (feed-forward) layer.
            - type_vocab_size (int): The size of the token type vocabulary.
            - device (str): The device to which the model should be moved (e.g., 'cpu' or 'cuda').

    Returns:
        BertForMaskedLM: A BERT model configured for masked language modeling, moved to the specified device.

    Note:
        This function uses the Hugging Face Transformers library to create the BERT model.
    """
    model_config = BertConfig(
        vocab_size=len(vocab_dict),
        hidden_size=CFG.hidden_size,
        num_hidden_layers=CFG.num_hidden_layers,
        num_attention_heads=CFG.num_attention_heads,
        intermediate_size=CFG.intermediate_size,
        type_vocab_size=CFG.type_vocab_size
    )
    return BertForMaskedLM(config=model_config)
