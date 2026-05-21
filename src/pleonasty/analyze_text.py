from .LLM_Result import LLM_Result
from time import time
import torch

def analyze_text(self,
                 input_texts: list,
                 **generation_kwargs) -> list:

    # vllm used "max_tokens"; transformers uses "max_new_tokens"
    if "max_tokens" in generation_kwargs:
        generation_kwargs.setdefault("max_new_tokens", generation_kwargs.pop("max_tokens"))

    generation_kwargs.setdefault("max_new_tokens", 512)

    # Sampling must be explicitly enabled when temperature / top_k / top_p are set
    if {"temperature", "top_k", "top_p"} & set(generation_kwargs):
        generation_kwargs.setdefault("do_sample", True)

    llm_results = []

    for input_text in input_texts:
        start_time = time()

        conversation = self._buildPrompt(input_text)

        formatted = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        new_tokens = output_ids[0][input_len:]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        stop_time = time()

        llm_results.append(
            LLM_Result(
                input_text=input_text,
                response_text=reply,
                model_output=output_ids,
                start_time=start_time,
                stop_time=stop_time
            )
        )

    self.result = llm_results
    return llm_results
