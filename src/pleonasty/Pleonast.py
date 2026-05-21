import csv
import sys
import warnings
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# bitsandbytes internal FutureWarning about PyTorch guard APIs — not actionable by users
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")

csv.field_size_limit(sys.maxsize)

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")


def _print_load_plan(model_name_str, tokenizer_name_str, quantize_model, model_kwargs, hf_token):
    """Print a human-readable summary of what is about to be loaded."""
    sep = "─" * 56
    print(sep)
    print("Pleonast: preparing to load model")
    print(f"  Model:       {model_name_str}")
    if tokenizer_name_str != model_name_str:
        print(f"  Tokenizer:   {tokenizer_name_str}")

    # Architecture from config.json — fast, no weights loaded
    try:
        from transformers import AutoConfig
        cfg_kwargs = {"token": hf_token} if hf_token else {}
        cfg = AutoConfig.from_pretrained(model_name_str, **cfg_kwargs)
        archs = getattr(cfg, "architectures", None)
        if archs:
            print(f"  Architecture: {archs[0]}")
    except Exception:
        pass

    # Disk footprint from weight files
    p = Path(model_name_str)
    if p.is_dir():
        st_files = list(p.rglob("*.safetensors"))
        bin_files = list(p.rglob("*.bin"))
        if st_files:
            size_gb = sum(f.stat().st_size for f in st_files) / 1024 ** 3
            print(f"  Weights on disk: {size_gb:.1f} GB (safetensors)")
        elif bin_files:
            size_gb = sum(f.stat().st_size for f in bin_files) / 1024 ** 3
            print(f"  Weights on disk: {size_gb:.1f} GB (.bin)")
            print("  Note: .bin weights load slower than safetensors.")

    # Quantization
    if "quantization_config" in model_kwargs:
        qcfg = model_kwargs["quantization_config"]
        if getattr(qcfg, "load_in_4bit", False):
            bits = "4-bit"
        elif getattr(qcfg, "load_in_8bit", False):
            bits = "8-bit"
        else:
            bits = "custom"
        print(f"  Quantization: {bits} bitsandbytes (on-the-fly — weights quantized at load time)")
    else:
        print("  Quantization: none")

    dtype = model_kwargs.get("torch_dtype", "auto")
    print(f"  torch_dtype:  {dtype}")
    print(f"  device_map:   {model_kwargs.get('device_map', 'auto')}")
    print(sep)
    print("Loading weights — this may take several minutes for large models...")


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
        # BPE tokenizers warn about this option being designed for WordPiece; suppress it
        self.tokenizer.clean_up_tokenization_spaces = False

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

        _print_load_plan(
            model_name_str=self.model_name_str,
            tokenizer_name_str=self.tokenizer_name_str,
            quantize_model=quantize_model,
            model_kwargs=model_kwargs,
            hf_token=hf_token,
        )

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
