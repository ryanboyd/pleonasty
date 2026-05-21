import csv
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

csv.field_size_limit(sys.maxsize)

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")


class Pleonast:
    """The main class for having an LLM generate a response to individual texts that you pass to it in a systematic fashion."""

    def __init__(self,
                 model: str,
                 tokenizer: str = None,
                 quantize_model: bool = True,
                 hf_token: str = None,
                 **model_kwargs):
        self.model_name_str = model
        self.tokenizer_name_str = tokenizer if tokenizer else model
        self.hf_token = hf_token

        tok_kwargs = {}
        if hf_token:
            tok_kwargs["token"] = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_str, **tok_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hf_token:
            model_kwargs["token"] = hf_token

        if quantize_model:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs.setdefault("quantization_config", BitsAndBytesConfig(load_in_4bit=True))
            except ImportError:
                print("bitsandbytes not installed; skipping quantization. "
                      "Install with: pip install bitsandbytes  or  pip install pleonasty[quantization]")

        model_kwargs.setdefault("device_map", "auto")
        model_kwargs.setdefault("torch_dtype", "auto")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name_str, **model_kwargs)
        self.model.eval()

        print("Pleonast is initialized.")

    def chunk_by_tokens(self, text: str, chunk_size: int):
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids     = enc["input_ids"]
        offsets = enc["offset_mapping"]

        out = []
        for i in range(0, len(ids), chunk_size):
            j = min(i + chunk_size, len(ids))
            start_char = offsets[i][0]
            end_char   = offsets[j - 1][1]
            substring  = text[start_char:end_char]
            out.append(substring.strip())
        return out
