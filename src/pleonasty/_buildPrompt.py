from copy import deepcopy

def _buildPrompt(self, input_text: str) -> list:
        """
        Builds the full prompt to pass to the LLM. This essentially strings together the 'message_context' that is set
        by the user with a single, final 'input_text' that constitutes the last 'user' message given to the LLM.
        :param input_text: The text that you would like to constitute the last 'user' message for prompting generation.
        :return:
        """

        prompt_messages = list(deepcopy(self.message_context))
        prompt_messages.append({"role": "user",
                                "content": input_text})
        return prompt_messages