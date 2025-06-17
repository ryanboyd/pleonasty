from vllm import SamplingParams

def chat_mode(self,
              temperature: float = 0.75,
              top_k: int = 10,
              max_tokens: int = 1000,
              bot_name: str = "Bot",
              system_prompt: str = "You are a helpful AI assistant."
            ):
    
    # 1) Set up params
    params = {
    "temperature": temperature,
    "top_k":        top_k,
    "max_tokens":   max_tokens,
}
    
    # 2) Build a SamplingParams object
    sampling_params = SamplingParams(**params)

    # 3) Start chat loop
    print("Type 'quit' to exit chat mode.")
    # Seed with an optional system prompt
    history = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_input = input(">> You: ")
        if user_input.strip().lower() == "quit":
            print("Exiting chat mode...")
            break

        # 4) Append user turn
        history.append({"role": "user", "content": user_input})

        # 5) Call vLLM’s chat API
        outputs = self.llm.chat(
            messages=[history],               # one conversation in a batch
            add_generation_prompt=True,        # ensure an “assistant:” token is appended
            use_tqdm=False,
            sampling_params=sampling_params
        )

        # 6) Extract and print the assistant’s reply
        reply = outputs[0].outputs[0].text.strip()
        print(f"\n{bot_name}: {reply}\n")

        # 7) Append assistant turn so it’s included in the next round
        history.append({"role": "assistant", "content": reply})