def _format_conversation(self, messages: list) -> str:
    """Format a message list into a model input string.

    Priority:
      1. self._prompt_formatter callable (user-supplied)
      2. tokenizer.apply_chat_template (standard path)
      3. _simple_format fallback (plain User/Assistant text)
    """
    if self._prompt_formatter is not None:
        return self._prompt_formatter(messages)

    if self.tokenizer.chat_template is not None:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return _simple_format(messages)


def _simple_format(messages: list) -> str:
    """Generic prompt format for models that ship no Jinja chat template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
