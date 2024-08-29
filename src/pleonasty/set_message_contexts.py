import csv

def set_message_context(self, prompt_messages: list) -> None:
    """
    Sets the "context" for all messages that will be submitted to your LLM. This is essentially just the prompt
    history that you would like to prepend to any given text — it will be used at processing time by your
    tokenizer's "apply_chat_template" function, and the text that is analyzed will be treated as a final "user"
    message that is given to the LLM to generate its response.
    :param prompt_messages: A list of dictionaries that are used to prompt your LLM. Note that your prompts can
    be zero context, few-shot, etc., following the format below:

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
        raise MessageContextException("It appears that your message context is not a list of dictionaries.")

    for item in prompt_messages:
        if not isinstance(item, dict):
            raise MessageContextException("""Your prompt messages need to be contained in dictionaries.
             Each dictionary must have both a "role" and "content" key.""")

    self.message_context = prompt_messages
    print("Context has been set.")
    return


def set_message_context_from_CSV(self, filename: str, encoding:str = "utf-8-sig") -> None:
    """
    Sets the "context" for all messages that will be submitted to your LLM by loading it from a CSV file, where each
    row of the CSV includes the "role" and the "content" of all of the background context. Your CSV should have
    at least 2 columns with the headers "role" and "content" so that this function can pull them out.

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

    with open(filename, 'r', encoding=encoding) as fin:
        csvr = csv.reader(fin)

        header = csvr.__next__()

        if "role" not in header or "content" not in header:
            raise MessageContextException("Your input CSV must have a 'role' and 'content' column.")

        prompt_messages = []

        for row in csvr:
            role = row[header.index("role")]
            content = row[header.index("content")]

            prompt_messages.append({"role": role,
                                    "content": content})

    self.set_message_context(prompt_messages)
    return