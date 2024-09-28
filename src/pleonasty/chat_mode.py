import torch
import transformers

def chat_mode(self, max_length: int = 1000, top_k: int = 10, temperature: float = .75, bot_name: str = "Bot") -> None:

    # just disabling all warnings for now, as there appear to be many warnings that are not relevant
    transformers.logging.set_verbosity_error()

    print("Type 'quit' (without the quotes) to exit chat mode.")
    text = ""
    step = 0
    while True:
        # take user input
        text = input(">> You: ")

        if text.strip() == "quit":
            print("Exiting chat mode...")
            break

        # encode the input and add end of string token
        input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors="pt")
        # concatenate new user input with chat history (if there is)
        bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
        step += 1
        # generate a bot response
        chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        #print the output
        output = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0],
                                       skip_special_tokens=True)
        print(f"\r\n{bot_name}: {output.strip('assistant').strip()}\r\n")
    
    transformers.logging.set_verbosity_info()
    return