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
                 batch_size: int = 1,
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

        if "max_tokens" in generation_kwargs:
            generation_kwargs.setdefault("max_new_tokens", generation_kwargs.pop("max_tokens"))
        generation_kwargs.setdefault("max_new_tokens", 512)
        if {"temperature", "top_k", "top_p"} & set(generation_kwargs):
            generation_kwargs.setdefault("do_sample", True)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        current_batch_size = max(1, batch_size)
        i = 0

        while i < len(input_texts):
            batch = input_texts[i : i + current_batch_size]

            conversations = [self._buildPrompt(t) for t in batch]
            formatted    = [self._format_conversation(c) for c in conversations]

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            input_len  = inputs["input_ids"].shape[1]
            start_time = time()

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)

                stop_time = time()

                for input_text, out in zip(batch, output_ids):
                    new_tokens = out[input_len:]
                    reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    llm_results.append(
                        LLM_Result(
                            input_text=input_text,
                            response_text=reply,
                            model_output=out.unsqueeze(0),
                            start_time=start_time,
                            stop_time=stop_time,
                        )
                    )
                i += current_batch_size

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                new_size = current_batch_size // 2
                if new_size < 1:
                    raise RuntimeError(
                        "CUDA out of memory even with batch_size=1. "
                        "Try a smaller model, enable quantization, or reduce max_new_tokens."
                    )
                print(
                    f"OOM at batch_size={current_batch_size} — "
                    f"reducing to {new_size} for the rest of this job."
                )
                current_batch_size = new_size
                # retry the same batch at the smaller size (i not advanced)

    self.result = llm_results
    return llm_results
