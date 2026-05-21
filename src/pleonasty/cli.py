import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pleonasty",
        description="Batch-annotate texts with a local open-weight LLM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── shared helpers ──────────────────────────────────────────────────────────
    def _model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", required=True,
                       help="HuggingFace model ID, local path, or API model name.")
        p.add_argument("--backend", default="transformers",
                       choices=["transformers", "api"],
                       help="Inference backend (default: transformers).")
        p.add_argument("--api-base", default=None,
                       help="Base URL for OpenAI-compatible API "
                            "(default: http://localhost:11434/v1 for Ollama).")
        p.add_argument("--api-key", default=None,
                       help="API key (default: 'ollama' for local Ollama).")
        p.add_argument("--tokenizer", default=None,
                       help="Tokenizer ID or path (transformers backend only).")
        p.add_argument("--hf-token", default=None,
                       help="HuggingFace access token for gated models.")
        p.add_argument("--no-quantize", action="store_true",
                       help="Disable 4-bit bitsandbytes quantization (transformers backend only).")
        p.add_argument("--device-map", default="auto",
                       help="Device map for from_pretrained (default: auto).")
        p.add_argument("--torch-dtype", default=None,
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Model weight dtype (default: auto).")
        p.add_argument("--trust-remote-code", action="store_true",
                       help="Pass trust_remote_code=True to from_pretrained.")
        p.add_argument("--attn-implementation", default=None,
                       help="Attention implementation, e.g. flash_attention_2.")

    def _generation_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-new-tokens", type=int, default=512,
                       help="Max tokens to generate per chunk (default: 512).")
        p.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7).")
        p.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling (default: 50).")
        p.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p nucleus sampling (default: 0.9).")
        p.add_argument("--repetition-penalty", type=float, default=1.0,
                       help="Repetition penalty (default: 1.0 = no penalty).")

    # ── annotate ───────────────────────────────────────────────────────────────
    ann = sub.add_parser("annotate",
                         help="Batch-annotate a CSV file and write results to a new CSV.")
    _model_args(ann)
    _generation_args(ann)
    ann.add_argument("--context-csv", default=None,
                     help="CSV with role/content columns that define the message context (prompt).")
    ann.add_argument("--input-csv", required=True,
                     help="Path to the input CSV file.")
    ann.add_argument("--text-columns", nargs="+", required=True,
                     help="Column name(s) whose text should be annotated.")
    ann.add_argument("--metadata-columns", nargs="*", default=[],
                     help="Column name(s) to carry through to the output unchanged.")
    ann.add_argument("--output-csv", default="AnalysisResults.csv",
                     help="Output CSV path (default: AnalysisResults.csv).")
    ann.add_argument("--start-at-row", type=int, default=0,
                     help="Row index to start processing at (default: 0).")
    ann.add_argument("--append", action="store_true",
                     help="Append to an existing output CSV instead of overwriting.")
    ann.add_argument("--encoding", default="utf-8-sig",
                     help="File encoding for input and output CSVs (default: utf-8-sig).")
    ann.add_argument("--chunk-tokens", type=int, default=2000,
                     help="Max tokens per text chunk (default: 2000).")

    # ── parse ──────────────────────────────────────────────────────────────────
    prs = sub.add_parser(
        "parse",
        help="Extract JSON fields from LLM_Response column of a pleonasty output CSV.",
    )
    prs.add_argument("--input-csv", required=True,
                     help="Path to the pleonasty output CSV to parse.")
    prs.add_argument("--json-fields", nargs="+", required=True,
                     help="JSON key names to extract from each LLM response.")
    prs.add_argument("--output-csv", default=None,
                     help="Output CSV path (default: <input>_parsed.csv).")
    prs.add_argument("--response-column", default="LLM_Response",
                     help="Column containing LLM responses (default: LLM_Response).")
    prs.add_argument("--group-by", nargs="*", default=None,
                     help="Column(s) to group rows by before aggregating chunks "
                          "(e.g. --group-by TextID).  When set, all rows with the "
                          "same key are merged into one output row.")
    prs.add_argument("--encoding", default="utf-8-sig",
                     help="File encoding (default: utf-8-sig).")

    # ── chat ───────────────────────────────────────────────────────────────────
    chat = sub.add_parser("chat",
                          help="Start an interactive chat session with a local LLM.")
    _model_args(chat)
    chat.add_argument("--max-new-tokens", type=int, default=1000,
                      help="Max tokens to generate per reply (default: 1000).")
    chat.add_argument("--temperature", type=float, default=0.75,
                      help="Sampling temperature (default: 0.75).")
    chat.add_argument("--top-k", type=int, default=10,
                      help="Top-k sampling (default: 10).")
    chat.add_argument("--bot-name", default="Bot",
                      help="Label printed before each assistant reply.")
    chat.add_argument("--system-prompt",
                      default="You are a helpful AI assistant.",
                      help="System prompt for the session.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── parse: no model needed, dispatch immediately ───────────────────────────
    if args.command == "parse":
        try:
            from pleonasty import parse_json_output
        except ImportError:
            from src.pleonasty import parse_json_output

        parse_json_output(
            input_csv=args.input_csv,
            json_fields=args.json_fields,
            output_csv=args.output_csv,
            response_column=args.response_column,
            group_by=args.group_by if args.group_by else None,
            encoding=args.encoding,
        )
        return

    # ── load model ─────────────────────────────────────────────────────────────
    try:
        from pleonasty import Pleonast
    except ImportError:
        from src.pleonasty import Pleonast

    if args.backend == "api":
        pleonast = Pleonast(
            model=args.model,
            backend="api",
            api_base=args.api_base,
            api_key=args.api_key,
        )
    else:
        model_kwargs: dict = {"device_map": args.device_map}

        if args.torch_dtype:
            import torch
            dtype_map = {
                "float16":  torch.float16,
                "bfloat16": torch.bfloat16,
                "float32":  torch.float32,
                "auto":     "auto",
            }
            model_kwargs["torch_dtype"] = dtype_map[args.torch_dtype]

        if args.trust_remote_code:
            model_kwargs["trust_remote_code"] = True

        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        pleonast = Pleonast(
            model=args.model,
            tokenizer=args.tokenizer,
            quantize_model=not args.no_quantize,
            hf_token=args.hf_token,
            **model_kwargs,
        )

    # ── dispatch ───────────────────────────────────────────────────────────────
    if args.command == "annotate":
        if args.context_csv:
            pleonast.set_message_context_from_CSV(args.context_csv)
        else:
            print("No --context-csv provided — using empty message context.")
            pleonast.set_message_context([])

        generation_kwargs = {
            "max_new_tokens":     args.max_new_tokens,
            "temperature":        args.temperature,
            "top_k":              args.top_k,
            "top_p":              args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }

        pleonast.batch_analyze_csv_to_csv(
            input_csv=args.input_csv,
            text_columns_to_process=args.text_columns,
            metadata_columns_to_retain=args.metadata_columns,
            start_at_row=args.start_at_row,
            output_csv=args.output_csv,
            append_to_existing_csv=args.append,
            file_encoding=args.encoding,
            chunk_into_n_tokens=args.chunk_tokens,
            **generation_kwargs,
        )

    elif args.command == "chat":
        pleonast.set_message_context([])
        pleonast.chat_mode(
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            bot_name=args.bot_name,
            system_prompt=args.system_prompt,
        )


if __name__ == "__main__":
    main()
