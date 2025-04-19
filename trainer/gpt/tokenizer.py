from pathlib import Path
from ...minbpe_tokenizer import BasicTokenizer


def load_tokenizer():
    model_path = Path(__file__).resolve().parents[2] / "resources" / "darija_tokenizer.model"
    tokenizer = BasicTokenizer()
    tokenizer.load(model_file=str(model_path))

    max_vocab_id = list(tokenizer.vocab.keys())[-1]
    tokenizer.special_tokens = {
        "<|startoftext|>": max_vocab_id + 1,
        "<|separator|>": max_vocab_id + 2,
        "<|endoftext|>": max_vocab_id + 3,
        "<|unk|>": max_vocab_id + 4
    }
    return tokenizer


tokenizer = load_tokenizer()
