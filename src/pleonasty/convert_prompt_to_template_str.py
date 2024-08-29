import torch
import transformers

def convert_prompt_to_template_str(self, prompt_messages: list) -> str:
    """
            Converts a prompt set (in the form of a list of dictionaries) to a templated string. Helpful for setting
            up some data for fine-tuning. Your prompt set should be in a format like this one:

            prompt_messages = [
                                {"role": "system", "content": "Please answer the math question."},
                                {"role": "user", "content": "1+1=?"},  # example 1
                                {"role": "assistant", "content": "2"},  # example 1
                                {"role": "user", "content": "1+2=?"},  # example 2
                                {"role": "assistant", "content": "3"},  # example 2
                                {"role": "user", "content": "2+2=?"}
                            ]

            :return:
            """

    if not isinstance(prompt_messages, list):
        raise MessageContextException("It appears that your prompt set is not a list of dictionaries.")

    for item in prompt_messages:
        if not isinstance(item, dict):
            raise MessageContextException("""Your prompt set need to be contained in dictionaries.
                     Each dictionary must have both a "role" and "content" key.""")

    return self.tokenizer.apply_chat_template(prompt_messages, 
                                              tokenize=False,
                                              add_generation_prompt=True)