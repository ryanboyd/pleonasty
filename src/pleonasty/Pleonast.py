import torch
import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

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
                 quantize_model: bool = False,
                 offload_folder: str = None,
                 hf_token: str = None):
        self.model_name_str = model
        self.tokenizer_name_str = tokenizer if tokenizer else model
        self.hf_token = hf_token
        self.offload_folder = offload_folder

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_device("cuda")
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = self._load_model(quantize_model)
        self.tokenizer = self._load_tokenizer()
        self._pipeline = None  # Pipeline will be lazily loaded

        print("Pleonast is initialized.")

    def _load_model(self, quantize_model: bool):
        try:
            kwargs = {
                "pretrained_model_name_or_path": self.model_name_str,
                "device_map": "auto",
                "offload_folder": self.offload_folder if self.offload_folder else None,
                "torch_dtype": torch.float16,
                "trust_remote_code": True,  # Needed for many new/large models
                "token": self.hf_token,
            }
            if quantize_model:
                if self.device != "cuda":
                    raise ValueError("No GPU found. A GPU is needed for quantization.")
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0)
            model = AutoModelForCausalLM.from_pretrained(**kwargs)
            print("Device map:", model.hf_device_map)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.tokenizer_name_str,
                use_fast=True,
                token=self.hf_token
            )
            return tokenizer
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                self._pipeline = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    return_full_text=False
                )
            except Exception as e:
                print(f"Error initializing pipeline: {e}")
                raise
        return self._pipeline

    def generate(self, input_text, **kwargs):
        try:
            return self.pipeline(input_text, **kwargs)
        except Exception as e:
            print(f"Error during text generation: {e}")
            raise
