import csv
import os.path
import torch
import transformers
from copy import deepcopy
from tqdm import tqdm
from time import time

def make_safe_filename(s):
    def safe_char(c):
        if c.isalnum():
            return c
        else:
            return "_"

    return "".join(safe_char(c) for c in s).rstrip("_")


# These two classes might end up being vestigial given that the transformers package already has chat templates.
# They are being kept for now, but will probably be dropped later.
class InstructionFormat():
    def __init__(self, instruction_format: str, final_delimiter: str):
        self.instruction_format = instruction_format
        self.final_delimiter = final_delimiter
        return

class LLMFormats():
    orca2 = InstructionFormat(instruction_format="<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n<|im_start|>user\n{USER_MESSAGE}<|im_end|>\n<|im_start|>assistant",
                              final_delimiter="<|im_start|>assistant\n")
    llama2 = InstructionFormat(instruction_format="""<s>[INST] <<SYS>>\n{SYSTEM_MESSAGE}\n<</SYS>>\n\n{USER_MESSAGE} [/INST]""",
                               final_delimiter="[/INST]")

class MessageContextException(Exception):
    """
    Custom exception type to be thrown when user is trying to set context but something goes wrong
    """
    pass

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


class Pleonast():
    """The main class for having an LLM generate a response to individual texts that you pass to it in a systematic fashion."""

    def __init__(self,
                 on_Windows: bool,
                 model: str,
                 tokenizer: str,
                 offload_folder: str,
                 #llm_format: InstructionFormat,
                 hf_token: str = None
                 ):

        self.result = None
        self.message_context = []
        #self.llm_format = llm_format

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_device("cuda")

        if on_Windows:

            if offload_folder is None:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model,
                    device_map="auto",
                    token=hf_token)
            else:
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model,
                    device_map="auto",
                    offload_folder=offload_folder,
                    token=hf_token)
        else:
            quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True,
                                                                  llm_int8_threshold=200.0)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model,
                device_map="auto",
                quantization_config=quantization_config,
                token=hf_token
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer,
            use_fast=True,
            token=hf_token
        )

        self.pipeline = transformers.pipeline(task="text-generation",
                                              model=self.model,
                                              tokenizer=self.tokenizer,
                                              torch_dtype=torch.float16,
                                              device_map="auto",
                                              return_full_text=False
                                              )

        print("Pleonast is initialized.")
        return

    def _buildPrompt(self, input_text: str) -> list:
        """
        Builds the full prompt to pass to the LLM. This essentially strings together the 'message_context' that is set
        by the user with a single, final 'input_text' that constitutes the last 'user' message given to the LLM.
        :param input_text: The text that you would like to constitute the last 'user' message for prompting generation.
        :return:
        """
        prompt_messages = deepcopy(self.message_context)
        prompt_messages.append({"role": "user",
                                "content": input_text})
        return prompt_messages

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
            max_length=max_seq_length)

        return sequences

    def analyze_text(self,
                     input_text: str,
                     max_seq_length: int = 4096,
                     temperature: float = 0.3,
                     top_k: int = 10):

        start_time = time()

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

    def batch_analyze_to_csv(self,
                             texts: list,
                             text_metadata: dict,
                             csv_output_location: str,
                             append_to_existing_csv: bool = False,
                             output_encoding: str = "utf-8-sig",
                             max_seq_length: int = 4096,
                             temperature: float = 0.3,
                             top_k: int = 10,
                             ) -> None:

        writemode = 'w'
        if append_to_existing_csv:
            writemode = 'a'

        with open(csv_output_location, writemode, encoding=output_encoding, newline='') as fout:

            csvw = csv.writer(fout)
            meta_headers = list(text_metadata.keys())

            if append_to_existing_csv is False:
                csvw.writerow(self.generate_csv_header(
                    metadata_headers=meta_headers))

            for i in tqdm(range(len(texts))):

                self.analyze_text(input_text=texts[i],
                                  max_seq_length=max_seq_length,
                                  temperature=temperature,
                                  top_k=top_k)

                # prep the row output with metadata
                meta_output = []
                for meta_item in meta_headers:
                    meta_output.append(text_metadata[meta_item][i])

                # complete the row output by pulling the results data
                row_output = self.generate_csv_output_row(input_metadata=meta_output)

                #write the output
                csvw.writerow(row_output)

        print("Analysis complete.")

        return

    def generate_csv_header(self, metadata_headers: list):
        """
        Helper function to generate a CSV header
        :param metadata_headers: The other headers that will be prepended to your list of archetypes
        :return:
        """
        mh = metadata_headers.copy()
        mh.extend(["text", "Input_WC", "LLM_Response"])
        return mh

    def generate_csv_output_row(self, input_metadata: list) -> list:
        """
        Generates a row of output for a CSV file by concatenating metadata columns with the LLM output columns.
        :param input_metadata:
        :return:
        """
        row_results = []
        row_results.extend(input_metadata)
        row_results.extend([self.result.input_text, self.result.WC, self.result.response_text])

        return row_results

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


# some potential stuff for parsing jsons
#import ast
#import json
#from pprint import pprint

#result_str_list = ast.literal_eval(result[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip().split(',\n\n') + "]}")
#result_dict_list = [json.loads(x) for x in result_str_list]

#for item in result_dict_list:
#  pprint(item)
#  print('\n\n')