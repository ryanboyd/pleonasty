from .LLM_Result import LLM_Result
from time import time


def _to_api_kwargs(kwargs: dict) -> dict:
    """Translate transformers generation kwargs to OpenAI API equivalents."""
    out = {}
    for k, v in kwargs.items():
        if k in ("max_new_tokens", "max_tokens"):
            out["max_tokens"] = v
        elif k in ("do_sample", "pad_token_id", "top_k"):
            pass  # transformers-only, no OpenAI API equivalent
        else:
            out[k] = v
    out.setdefault("max_tokens", 512)
    return out


def analyze_text(self,
                 input_texts: list,
                 **generation_kwargs) -> list:

    llm_results = []

    if self._backend == "api":
        api_kwargs = _to_api_kwargs(generation_kwargs)

        for input_text in input_texts:
            start_time = time()
            conversation = self._buildPrompt(input_text)
            reply = self._api_generate(conversation, **api_kwargs)
            stop_time = time()
            llm_results.append(
                LLM_Result(
                    input_text=input_text,
                    response_text=reply,
                    model_output=None,
                    start_time=start_time,
                    stop_time=stop_time,
                )
            )

    else:
        import torch

        # vllm used "max_tokens"; transformers uses "max_new_tokens"
        if "max_tokens" in generation_kwargs:
            generation_kwargs.setdefault("max_new_tokens", generation_kwargs.pop("max_tokens"))

        generation_kwargs.setdefault("max_new_tokens", 512)

        # Sampling must be explicitly enabled when temperature / top_k / top_p are set
        if {"temperature", "top_k", "top_p"} & set(generation_kwargs):
            generation_kwargs.setdefault("do_sample", True)

        # Suppress the per-call transformers log about pad_token_id
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        for input_text in input_texts:
            start_time = time()

            conversation = self._buildPrompt(input_text)
            formatted = self._format_conversation(conversation)

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
                    stop_time=stop_time,
                )
            )

    self.result = llm_results
    return llm_results
