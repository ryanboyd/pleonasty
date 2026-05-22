def chat_mode(self,
              temperature: float = 0.75,
              top_k: int = 10,
              max_new_tokens: int = 1000,
              bot_name: str = "Bot",
              system_prompt: str = "You are a helpful AI assistant."
            ):

    print("Type 'quit' to exit chat mode.")
    history = [{"role": "system", "content": system_prompt}]

    while True:
        user_input = input(">> You: ")
        if user_input.strip().lower() == "quit":
            print("Exiting chat mode...")
            break

        history.append({"role": "user", "content": user_input})

        if self._backend == "api":
            reply = self._api_generate(
                history,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            import torch
            formatted = self._format_conversation(history)
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"\n{bot_name}: {reply}\n")
        history.append({"role": "assistant", "content": reply})
