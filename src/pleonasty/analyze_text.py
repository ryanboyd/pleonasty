import threading
from time import time

from .LLM_Result import LLM_Result


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


def _make_token_counter():
    """Return a StoppingCriteria that counts generation steps without stopping."""
    from transformers import StoppingCriteria

    class _TokenCounter(StoppingCriteria):
        __slots__ = ("count",)
        def __init__(self):       self.count = 0
        def __call__(self, *_, **__): self.count += 1; return False

    return _TokenCounter()


def _run_status_thread(status_bar, counter, batch_label, batch_size, start):
    """Daemon thread: updates status_bar every second while generation runs."""
    stop = threading.Event()

    def _worker():
        while not stop.wait(timeout=1.0):
            elapsed  = time() - start
            tokens   = counter.count * batch_size
            tok_s    = tokens / elapsed if elapsed > 0 else 0
            status_bar.set_description(
                f"{batch_label} | Tokens: {tokens} | "
                f"{elapsed:.0f}s | {tok_s:.1f} tok/s"
            )

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return stop, t


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
        from transformers import StoppingCriteriaList

        if "max_tokens" in generation_kwargs:
            generation_kwargs.setdefault("max_new_tokens", generation_kwargs.pop("max_tokens"))
        generation_kwargs.setdefault("max_new_tokens", 512)
        if {"temperature", "top_k", "top_p"} & set(generation_kwargs):
            generation_kwargs.setdefault("do_sample", True)
        generation_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)

        # Attach our token counter to any existing stopping criteria
        counter = _make_token_counter()
        existing = generation_kwargs.pop("stopping_criteria", StoppingCriteriaList())
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            list(existing) + [counter]
        )

        status_bar  = getattr(self, "_status_bar",  None)
        batch_label = getattr(self, "_batch_label", "Generating")

        current_batch_size = max(1, batch_size)
        i = 0
        batch_num = 0
        n_batches = -(-len(input_texts) // current_batch_size)  # ceil division

        while i < len(input_texts):
            batch = input_texts[i : i + current_batch_size]
            batch_num += 1

            # Show chunk progress when a single call spans multiple batches
            if n_batches > 1:
                label = f"{batch_label} | Batch {batch_num}/{n_batches}"
            else:
                label = batch_label

            conversations = [self._buildPrompt(t) for t in batch]
            formatted     = [self._format_conversation(c) for c in conversations]

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            input_len  = inputs["input_ids"].shape[1]
            counter.count = 0          # reset for this batch
            start_time = time()

            stop_event = t = None
            if status_bar is not None:
                stop_event, t = _run_status_thread(
                    status_bar, counter, label,
                    len(batch), start_time
                )

            try:
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)

                stop_time = time()

                if stop_event is not None:
                    stop_event.set()
                    t.join(timeout=2)
                    elapsed  = stop_time - start_time
                    tokens   = counter.count * len(batch)
                    tok_s    = tokens / elapsed if elapsed > 0 else 0
                    status_bar.set_description(
                        f"{label} | Tokens: {tokens} | "
                        f"{elapsed:.1f}s | {tok_s:.1f} tok/s"
                    )

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
                if stop_event is not None:
                    stop_event.set()
                    t.join(timeout=2)
                torch.cuda.empty_cache()
                new_size = current_batch_size // 2
                if new_size < 1:
                    raise RuntimeError(
                        "CUDA out of memory even with batch_size=1. "
                        "Try a smaller model, enable quantization, or reduce max_new_tokens."
                    )
                msg = (f"OOM at batch_size={current_batch_size} — "
                       f"reducing to {new_size} for the rest of this job.")
                if status_bar is not None:
                    status_bar.set_description(msg)
                else:
                    print(msg)
                current_batch_size = new_size

    self.result = llm_results
    return llm_results
