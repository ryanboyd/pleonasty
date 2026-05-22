# Pleonasty

Pleonasty is a Python library that makes it easy to apply a local open-weight LLM to large text datasets for batch annotation and analysis. Point it at a Hugging Face model (or an OpenAI-compatible API endpoint), write a prompt, and get structured CSV output — one annotated row per text, with automatic token-based chunking for long documents. It also includes a lightweight utility for parsing JSON fields out of LLM responses.

## Key Features

* **Batch annotation** — annotate large text datasets (CSV file or Python list) with a custom LLM prompt, results saved to CSV.
* **Token-based chunking** — long documents are automatically split into N-token chunks so they never overflow the context window.
* **JSON response parsing** — extract structured fields from LLM responses that return JSON objects, with automatic aggregation across chunks.
* **Thinking model support** — strip reasoning blocks (e.g. `<think>...</think>`) into their own column before parsing JSON.
* **Chat mode** — interactive REPL for back-and-forth conversation with a loaded model.
* **Flexible model loading** — works with any Hugging Face causal LM; supports 4-bit quantization, multi-GPU, CPU offload, gated/private repos.
* **API backend** — point pleonasty at any OpenAI-compatible endpoint (Ollama, DeepSeek API, Together, Groq, etc.) instead of loading weights locally.
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

To use the API backend (Ollama, DeepSeek API, etc.):

```bash
pip install pleonasty[api]            # installs openai
```

### Requirements

* Python 3.10+
* PyTorch 2.0+ (with CUDA for GPU inference; not required when using the API backend)

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

#### Using an API backend

Instead of loading weights locally, you can point pleonasty at any OpenAI-compatible API endpoint. This works with Ollama (local), the DeepSeek API, Together, Groq, Fireworks, LM Studio, and anything else that speaks the `/v1/chat/completions` protocol.

```python
# Ollama running locally (default base URL — no api_key needed)
ple = Pleonast(
    model="llama3.1:8b",      # exact name from `ollama list`
    backend="api",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
)

# DeepSeek cloud API
ple = Pleonast(
    model="deepseek-chat",
    backend="api",
    api_base="https://api.deepseek.com/v1",
    api_key="sk-...",
)

# Together AI
ple = Pleonast(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    backend="api",
    api_base="https://api.together.xyz/v1",
    api_key="<TOGETHER_API_KEY>",
)
```

When using the API backend, no model is loaded locally — `quantize_model`, `device_map`, `torch_dtype`, and other GPU parameters are ignored. The `openai` package must be installed (`pip install pleonasty[api]`).

#### Models without a chat template

Some models ship without a Jinja chat template (e.g. DeepSeek-V3.2). Pleonasty will warn you at load time and fall back to a simple `User: / Assistant:` format. For correct results with such models, pass the model's own encoding function as `prompt_formatter`:

```python
import sys
sys.path.insert(0, "/path/to/model/encoding")
from encoding_dsv32 import encode_messages

ple = Pleonast(
    model="deepseek-ai/DeepSeek-V3.2",
    trust_remote_code=True,
    prompt_formatter=lambda msgs: encode_messages(msgs, thinking_mode="non-thinking"),
)
```

`prompt_formatter` must be a callable that accepts a list of `{"role": ..., "content": ...}` dicts and returns a formatted string.

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

`json_fields` is optional — if omitted, field names are auto-discovered from the union of all successfully parsed rows.

#### Thinking / reasoning models

Models like DeepSeek-R1 and QwQ prefix their response with a reasoning block enclosed in tags (e.g. `<think>...</think>`). Pass `reasoning_end_tag` to strip the reasoning into its own column before parsing JSON:

```python
parse_json_output(
    input_csv="data/annotated.csv",
    reasoning_end_tag="</think>",        # split at this tag
    reasoning_column="LLM_Reasoning",    # default column name — can be omitted
)
```

The output CSV gains a `LLM_Reasoning` column (containing everything up to and including the tag) and JSON is parsed only from the text that follows. Rows where the tag is not found are parsed in full as usual, so the option is safe to use on mixed datasets.

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

Using the API backend from the CLI:

```bash
pleonasty annotate \
  --model deepseek-chat \
  --backend api \
  --api-base https://api.deepseek.com/v1 \
  --api-key sk-... \
  --context-csv prompts/my_prompt.csv \
  --input-csv data/texts.csv \
  --text-columns post_text \
  --output-csv data/annotated.csv
```

### Parse JSON Responses

```bash
pleonasty parse \
  --input-csv data/annotated.csv \
  --json-fields is_present presence_score evidence_spans justification \
  --group-by user_id \
  --output-csv data/annotated_parsed.csv
```

For thinking models:

```bash
pleonasty parse \
  --input-csv data/annotated.csv \
  --reasoning-end-tag "</think>" \
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

Generation parameters are passed as keyword arguments to `batch_analyze_csv_to_csv`, `batch_analyze_to_csv`, and `analyze_text`. For the transformers backend they are forwarded to `model.generate()`; for the API backend they are forwarded to the chat completions request.

| Parameter | Default | Notes |
|---|---|---|
| `max_new_tokens` | 512 | Max tokens the model may generate per chunk |
| `temperature` | 0.7 | Higher = more creative; lower = more deterministic |
| `top_k` | 50 | Sample from the top-k most likely next tokens (transformers backend only) |
| `top_p` | 0.9 | Nucleus sampling probability threshold |
| `repetition_penalty` | 1.0 | Values > 1 penalise repeated phrases (transformers backend only) |
| `do_sample` | auto | Automatically enabled when temperature/top_k/top_p are set (transformers backend only) |
| `batch_size` | 1 | Number of texts processed per `model.generate()` call — see below (transformers backend only) |

`max_tokens` is accepted as an alias for `max_new_tokens` for backwards compatibility.

### Batched inference (`batch_size`)

By default pleonasty processes one text at a time. Setting `batch_size > 1` sends multiple texts to the GPU in a single `model.generate()` call, which can be 2–4× faster because the GPU is doing genuine parallel work rather than sitting mostly idle between calls.

```python
ple.batch_analyze_csv_to_csv(
    input_csv="data/texts.csv",
    text_columns_to_process=["post_text"],
    output_csv="data/annotated.csv",
    max_new_tokens=512,
    batch_size=8,   # process 8 texts at once
)
```

**Cross-text independence is guaranteed.** Each sequence in a batch is protected by its own attention mask — tokens from one text are completely invisible to tokens in another. Batched results are mathematically identical to processing each text separately (with greedy decoding; stochastic sampling will produce different draws from the same distribution, not semantic bleed-through).

**VRAM and automatic backoff.** Larger batches use more VRAM, scaling roughly with `batch_size × (input_tokens + max_new_tokens)`. If a batch exceeds available VRAM, pleonasty catches the out-of-memory error, halves the batch size, prints a message, and retries — permanently using the lower size for the rest of the job. This means:

- A slightly-too-large `batch_size` self-corrects within the first one or two batches and then runs stably.
- If even `batch_size=1` causes OOM, pleonasty raises a clear error rather than looping.
- You never need to babysit the job: set a generous `batch_size` and let it find its own level.

A good starting point on a 2×A6000 (96 GB VRAM) with an 8B model is `batch_size=16`; for a 70B model quantized to 4-bit, start around `batch_size=4`. Tune upward until you see the backoff message, then drop back one step.

From the CLI:

```bash
pleonasty annotate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --batch-size 16 \
  --input-csv data/texts.csv \
  --text-columns post_text \
  --output-csv data/annotated.csv
```

## API Reference

### `Pleonast` class

| Parameter | Description |
|---|---|
| `model` | HuggingFace model ID, local path, or API model name. |
| `tokenizer` | Tokenizer ID or path (defaults to `model`). Transformers backend only. |
| `quantize_model` | Enable 4-bit bitsandbytes quantization (default: `True`). Transformers backend only. |
| `hf_token` | HuggingFace access token for gated/private repos. |
| `prompt_formatter` | Callable `fn(messages) -> str` for models that lack a Jinja chat template. |
| `backend` | `"transformers"` (default) or `"api"`. |
| `api_base` | Base URL of the OpenAI-compatible API (default: `http://localhost:11434/v1`). |
| `api_key` | API key (default: `"ollama"`). |
| `**model_kwargs` | Forwarded to `AutoModelForCausalLM.from_pretrained()`. Includes `trust_remote_code`, `device_map`, `torch_dtype`, `attn_implementation`, etc. |

| Method | Description |
|---|---|
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
    input_csv,                          # path to pleonasty output CSV
    json_fields=None,                   # list of JSON key names to extract; auto-discovered if omitted
    output_csv=None,                    # defaults to <input>_parsed.csv
    response_column="LLM_Response",     # column containing LLM responses
    group_by=None,                      # str or list[str] — column(s) to aggregate on
    encoding="utf-8-sig",
    reasoning_end_tag=None,             # e.g. "</think>" for DeepSeek-R1 / QwQ
    reasoning_column="LLM_Reasoning",   # output column for extracted reasoning
)
```

When `group_by` is set, rows sharing the same key are merged: numerics are averaged, lists are concatenated, strings are joined with newlines. A `num_chunks` column records how many rows were merged.

When `reasoning_end_tag` is set, a `reasoning_column` column is added to the output containing the reasoning text, and JSON is parsed only from what follows the tag.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests at https://github.com/ryanboyd/pleonasty

## License

MIT License
