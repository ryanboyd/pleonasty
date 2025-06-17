import csv
import sys
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

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
                 **vllm_kwargs):
        self.model_name_str = model
        self.tokenizer_name_str = tokenizer if tokenizer else model
        self.hf_token = hf_token

        if quantize_model:
            # in-flight 4-bit via bitsandbytes
            vllm_kwargs.setdefault("quantization", "bitsandbytes")

        if hf_token:
            vllm_kwargs.setdefault("hf_overrides", {})["use_auth_token"] = hf_token

        llm_init_kwargs = {
            "model": self.model_name_str,
            "tokenizer": self.tokenizer_name_str,
            **vllm_kwargs,               # <-- all other engine args
        }

        self.llm = LLM(**llm_init_kwargs)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.tokenizer_name_str,
                use_fast=True,
                token=self.hf_token
            )


        print("Pleonast is initialized.")
