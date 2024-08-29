def process_text(self,
                 prompt_messages: list,
                 max_seq_length: int = 4096,
                 temperature: float = 0.3,
                 top_k: int = 10,
                 ) -> list:

    sequences = self.pipeline(
        prompt_messages,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        num_return_sequences=1,
        eos_token_id=self.tokenizer.eos_token_id,
        pad_token_id=self.tokenizer.eos_token_id,
        max_length=max_seq_length,
        truncation=True
    )

    return sequences