from .LLM_Result import LLM_Result
from time import time

def analyze_text(self,
                 input_text: str,
                 max_seq_length: int = 4096,
                 temperature: float = 0.3,
                 top_k: int = 10):

    start_time = time()

    #print(self.message_context)

    LLM_result = self.process_text(prompt_messages=self._buildPrompt(input_text),
                                   max_seq_length=max_seq_length,
                                   temperature=temperature,
                                   top_k=top_k)

    #response_text = LLM_result[0]["generated_text"].split(self.llm_format.final_delimiter)[-1].strip()
    response_text = LLM_result[0]["generated_text"].strip()

    stop_time = time()

    self.result = LLM_Result(input_text=input_text,
                             response_text=response_text,
                             model_output=LLM_result[0],
                             start_time=start_time,
                             stop_time=stop_time)

    return