# Pleonasty

Pleonasty is a Python library that makes it easy to apply a local open-weight LLM to large text datasets for batch annotation and analysis. Point it at a Hugging Face model, write a prompt, and get structured CSV output — one annotated row per text, with automatic token-based chunking for long documents. It also includes a lightweight utility for parsing JSON fields out of LLM responses.

## Key Features

* **Batch annotation** — annotate large text datasets (CSV file or Python list) with a custom LLM prompt, results saved to CSV.
* **Token-based chunking** — long documents are automatically split into N-token chunks so they never overflow the context window.
* **JSON response parsing** — extract structured fields from LLM responses that return JSON objects, with automatic aggregation across chunks.
* **Chat mode** — interactive REPL for back-and-forth conversation with a loaded model.
* **Flexible model loading** — works with any Hugging Face causal LM; supports 4-bit quantization, multi-GPU, CPU offload, gated/private repos.
* **Cross-platform** — runs on Linux and Windows; no vLLM required.
* **CLI** — all major workflows available from the terminal after `pip install`.

## Installation

```bash
pip install pleonasty
```

To enable 4-bit quantization (recommended when you have a GPU):

```bash
pip install pleonasty[quantization]   # installs bitsandbytes
```

### Requirements

* Python 3.10+
* PyTorch 2.0+ (with CUDA for GPU inference)

Set `HF_HOME` before importing if you want models cached somewhere specific:

```bash
export HF_HOME=/data/models/hf
```

## Quickstart

### 1. Initialize Pleonast

```python
from pleonasty import Pleonast

ple = Pleonast(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantize_model=True,       # 4-bit via bitsandbytes (requires pip install pleonasty[quantization])
    # hf_token="<YOUR_HF_TOKEN>",  # for gated / private repos
)
```

All extra keyword arguments are forwarded to `AutoModelForCausalLM.from_pretrained()`, so anything that function accepts can be passed here:

```python
ple = Pleonast(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantize_model=False,
    torch_dtype="bfloat16",                  # explicit weight dtype
    device_map="cuda:0",                     # pin to a specific GPU (default: "auto")
    attn_implementation="flash_attention_2", # faster attention if flash-attn is installed
    trust_remote_code=True,                  # needed for some community models
)
```

### 2. Set a Prompt

```python
# From a CSV file with "role" and "content" columns:
ple.set_message_context_from_CSV("prompts/annotate_sentiment.csv")

# Or directly in Python (zero-shot, few-shot, system prompt — anything goes):
ple.set_message_context([
    {"role": "system",    "content": "Classify the sentiment of the text as POSITIVE, NEGATIVE, or NEUTRAL."},
    {"role": "user",      "content": "I love this product!"},   # few-shot example
    {"role": "assistant", "content": "POSITIVE"},
])
```

### 3. Annotate a CSV File

```python
ple.batch_analyze_csv_to_csv(
    input_csv="data/input.csv",
    text_columns_to_process=["post_text"],
    metadata_columns_to_retain=["user_id", "timestamp"],
    output_csv="data/annotated.csv",
    chunk_into_n_tokens=2048,
    max_new_tokens=512,
    temperature=0.01,
    top_k=10,
)
# Output columns: user_id, timestamp, text, Input_WC, LLM_Response
```

### 4. Annotate a Python List

```python
texts = ["I love this!", "The capital of France is Paris."]
ple.batch_analyze_to_csv(
    texts=texts,
    text_metadata={"id": [1, 2]},
    output_csv="out.csv",
    chunk_into_n_tokens=1024,
    max_new_tokens=256,
    temperature=0.01,
)
```

### 5. Parse JSON Responses

If your prompt asks the model to respond with a JSON object, use `parse_json_output` to extract the fields into separate columns. When a document was split into multiple chunks, rows are aggregated automatically (numerics averaged, lists merged, strings joined).

```python
from pleonasty import parse_json_output

parse_json_output(
    input_csv="data/annotated.csv",
    json_fields=["is_present", "presence_score", "evidence_spans", "justification"],
    output_csv="data/annotated_parsed.csv",
    group_by="user_id",   # collapse multiple chunks per user into one row
)
```

### 6. Interactive Chat

```python
ple.chat_mode(
    temperature=0.75,
    top_k=10,
    max_new_tokens=500,
    bot_name="Annotator",
    system_prompt="You are an expert psychological annotator.",
)
# Type messages at the prompt; type 'quit' to exit.
```

## CLI

All major workflows are available from the terminal after `pip install pleonasty`.

### Annotate a CSV

```bash
pleonasty annotate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context-csv prompts/my_prompt.csv \
  --input-csv data/texts.csv \
  --text-columns post_text \
  --metadata-columns user_id timestamp \
  --output-csv data/annotated.csv \
  --chunk-tokens 2048 \
  --max-new-tokens 512 \
  --temperature 0.01
```

### Parse JSON Responses

```bash
pleonasty parse \
  --input-csv data/annotated.csv \
  --json-fields is_present presence_score evidence_spans justification \
  --group-by user_id \
  --output-csv data/annotated_parsed.csv
```

`pleonasty parse` has no dependency on torch or transformers and works on any machine.

### Interactive Chat

```bash
pleonasty chat \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --system-prompt "You are a helpful research assistant." \
  --max-new-tokens 500
```

Run `pleonasty <subcommand> --help` to see all options.

## Generation Parameters

Generation parameters are passed as keyword arguments to `batch_analyze_csv_to_csv`, `batch_analyze_to_csv`, and `analyze_text`. They are forwarded directly to Hugging Face's `model.generate()`, so any argument that function accepts is valid.

| Parameter | Default | Notes |
|---|---|---|
| `max_new_tokens` | 512 | Max tokens the model may generate per chunk |
| `temperature` | 0.7 | Higher = more creative; lower = more deterministic |
| `top_k` | 50 | Sample from the top-k most likely next tokens |
| `top_p` | 0.9 | Nucleus sampling probability threshold |
| `repetition_penalty` | 1.0 | Values > 1 penalise repeated phrases |
| `do_sample` | auto | Automatically enabled when temperature/top_k/top_p are set |

`max_tokens` is accepted as an alias for `max_new_tokens` for backwards compatibility.

## API Reference

### `Pleonast` class

| Method | Description |
|---|---|
| `Pleonast(model, ...)` | Load a model. `quantize_model=True` enables 4-bit quantization. Extra kwargs go to `from_pretrained()`. |
| `set_message_context(msgs)` | Set the prompt as a list of `{"role": ..., "content": ...}` dicts. |
| `set_message_context_from_CSV(path)` | Load prompt from a CSV with `role` and `content` columns. |
| `chunk_by_tokens(text, chunk_size)` | Split text into chunks of at most `chunk_size` tokens. |
| `analyze_text(texts, **gen_kwargs)` | Annotate a list of texts; returns a list of `LLM_Result` objects. |
| `batch_analyze_to_csv(texts, ...)` | Annotate a Python list and write results to a CSV. |
| `batch_analyze_csv_to_csv(input_csv, ...)` | Annotate a CSV file and write results to a new CSV. |
| `chat_mode(...)` | Launch an interactive chat session. |
| `convert_prompt_to_template_str(msgs)` | Apply the model's chat template to a message list and return the string. Useful for preparing fine-tuning data. |

### `LLM_Result` object

Each call to `analyze_text` returns a list of `LLM_Result` objects with these attributes:

| Attribute | Description |
|---|---|
| `input_text` | The chunk of text that was sent to the model |
| `response_text` | The model's generated response |
| `WC` | Word count of the input chunk |
| `elapsed_time` | Seconds taken to generate this result |

### `parse_json_output` (standalone function)

```python
from pleonasty import parse_json_output

parse_json_output(
    input_csv,           # path to pleonasty output CSV
    json_fields,         # list of JSON key names to extract
    output_csv=None,     # defaults to <input>_parsed.csv
    response_column="LLM_Response",
    group_by=None,       # str or list[str] — column(s) to aggregate on
    encoding="utf-8-sig",
)
```

When `group_by` is set, rows sharing the same key are merged: numerics are averaged, lists are concatenated, strings are joined with newlines. A `num_chunks` column records how many rows were merged.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests at https://github.com/ryanboyd/pleonasty

## License

MIT License
