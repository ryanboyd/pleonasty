import time

class LLM_Result():
    """A class that holds the output and some metadata from an LLM."""
    def __init__(self,
                 input_text: str,
                 response_text: str,
                 model_output,
                 start_time: time,
                 stop_time: time) -> None:
        self.input_text = input_text
        self.response_text = response_text
        self.model_output = model_output
        self.WC = len(input_text.strip().split())
        self.start_time = start_time
        self.stop_time = stop_time
        self.elapsed_time = stop_time - start_time
        return