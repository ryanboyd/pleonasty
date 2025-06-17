from .LLM_Result import LLM_Result
from time import time
from vllm import SamplingParams

def analyze_text(self,
                 input_texts: list,
                 **sampling_params) -> list[LLM_Result]:

    start_time = time()

    # 1) Build one chat-conversation per input
    conversations = []
    for input_text in input_texts:
        # your _buildPrompt can still be used to shape the user turn
        conversation_history = self._buildPrompt(input_text)
        conversations.append(conversation_history)

    # 2) Wrap the sampling params
    sampling = SamplingParams(**sampling_params)

    # 3) Run vLLM's chat API
    outputs = self.llm.chat(
        messages=conversations,
        add_generation_prompt=True,
        use_tqdm=False,
        sampling_params=sampling
    )
    stop_time = time()

    # 4) Collect just the assistant's reply
    llm_results = []
    for input_text, output in zip(input_texts, outputs):
        reply = output.outputs[0].text.strip()
        llm_results.append(
            LLM_Result(
                input_text=input_text,
                response_text=reply,
                model_output=output,
                start_time=start_time,
                stop_time=stop_time
            )
        )

    self.result = llm_results
    return llm_results