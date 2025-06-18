# Pleonasty

Pleonasty is a Python library built on top of [vLLM](https://github.com/vllm-project/vllm) and Hugging Face Transformers that streamlines batch annotation, analysis, and interactive chat with large language models. It makes it easy to apply the same prompt or system instructions to hundreds or thousands of texts and save results in CSV form, or to start an interactive chat loop for ad-hoc experimentation.

## Key Features

* **Batch Annotation**: Annotate large text datasets (CSV or Python lists) with a custom LLM prompt.
* **CSV I/O**: Read input columns, retain metadata, and write annotated outputs as CSV.
* **Chat Mode**: Launch an interactive REPL for back‑and‑forth conversation with your model.
* **Token‑based Chunking**: Split long documents into N‑token chunks to avoid overflow.
* **Flexible Model Loading**: Support for in‑flight 4‑bit quantization via BitsAndBytes, tensor‑parallel across multiple GPUs, CPU offload, and Hugging Face Hub authentication.

## Installation

```bash
pip install pleonast
```

### GPU & vLLM Requirements

* Python 3.10+
* PyTorch with CUDA support (if using GPU inference)
* vLLM v0.9.1+

Set environment variables **before** importing `pleonasty` or `vllm` if you need custom cache locations:

```bash
# Hugging Face cache (models, tokenizers):
export HF_HOME=/data/models/hf
# vLLM internal cache (kernels, metadata):
export VLLM_CACHE_ROOT=/data/models/vllm_cache
```

## Quickstart

### 1. Initialize Pleonast

```python
from pleonasty import Pleonast

# Load a 7B instruct model in 4-bit, auto‑sharded on all GPUs
ple = Pleonast(
    model="meta-llama/Llama-3.1-7B-Instruct",
    quantize_model=True,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
    hf_token="<YOUR_HF_TOKEN>"  # for private repos
)
```

### 2. Batch Analyze Python List

```python
texts = ["Hello world!", "The capital of France is"]
sampling = {"temperature": 0.01, "max_tokens": 20, "top_p": 0.95}
ple.batch_analyze_to_csv(
    texts=texts,
    text_metadata={"id": [1, 2]},
    output_csv="out.csv",
    chunk_into_n_tokens=1024,
    **sampling
)
# writes a CSV with columns: id, text, Input_WC, LLM_Response
```

### 3. Batch Analyze a CSV File

```python
ple.batch_analyze_csv_to_csv(
    input_csv="data/input.csv",
    text_columns_to_process=["post_text"],
    metadata_columns_to_retain=["user_id", "timestamp"],
    output_csv="data/annotated.csv",
    chunk_into_n_tokens=2048,
    temperature=0.01,
    max_tokens=512,
    top_k=10
)
```

### 4. Interactive Chat Mode

```python
# Launch a REPL chat
ple.chat_mode(
    temperature=0.7,
    top_k=50,
    max_tokens=100,
    bot_name="Annotator",
    system_prompt="You are an expert psychological annotator."
)
# Type your messages, type 'quit' to exit.
```

### 5. Other Examples

I have included an example notebook is included in this repo that shows how it can be used. I have also included a "chat mode" where you can load up an LLM and have back-and-forth interactions with it — an example of this is also provided in a sample notebook.

## Advanced Usage

#### Setting Message Context Programmatically

```python
# Few-shot or system prompts for all calls
msgs = [
    {"role": "system", "content": "Annotate for sentiment."},
    {"role": "user",   "content": "Example: I love this!"},
    {"role": "assistant", "content": "POSITIVE"}
]
ple.set_message_context(msgs)
```

#### Loading Context from CSV

```python
ple.set_message_context_from_CSV("prompts/context.csv")
```

#### Token‑level Chunking

```python
chunks = ple.chunk_by_tokens(long_text, chunk_size=2000)
```

#### Converting Prompt to Template String

```python
template_str = ple.convert_prompt_to_template_str(msgs)
```

## API Reference

| Method                                   | Description                                          |
| ---------------------------------------- | ---------------------------------------------------- |
| `Pleonast(...)`                          | Initialize engine with model, quantization, etc.     |
| `ple.set_message_context(msgs)`          | Set chat context as list of role/content dicts       |
| `ple.set_message_context_from_CSV(file)` | Load context prompts from CSV (`role`,`content`)     |
| `ple.chunk_by_tokens(text, n)`           | Split text into `n`-token chunks                     |
| `ple.analyze_text(texts, **params)`      | Annotate list of texts, returns `LLM_Result` objects |
| `ple.batch_analyze_to_csv(...)`          | Annotate Python list & write CSV                     |
| `ple.batch_analyze_csv_to_csv(...)`      | Annotate CSV file & write CSV                        |
| `ple.chat_mode(...)`                     | Launch interactive chat REPL                         |
| `ple.convert_prompt_to_template_str`     | Serialize messages via HF tokenizer’s chat template  |

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open issues or pull requests at https://github.com/ryanboyd/pleonasty

## License

MIT License
