import torch
import transformers

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




class Pleonast():
    """The main class for having an LLM generate a response to individual texts that you pass to it in a systematic fashion."""

    def __init__(self,
                 quantize_model: bool,
                 model: str,
                 tokenizer: str,
                 offload_folder: str,
                 hf_token: str = None
                 ):

        self.model_name_str = model
        self.tokenizer_name_str = tokenizer
        self.result = None
        self.message_context = []

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_device("cuda")

        if quantize_model:

            try:

                quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True,
                                                                      llm_int8_threshold=200.0)
                self.model = transformers.AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model,
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=hf_token
                )
            except:
                quantize_model = False
                print("Unable to quantize model. Trying to continue with non-quantized model...")

        if not quantize_model:
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



# some potential stuff for parsing jsons
#import ast
#import json
#from pprint import pprint

#result_str_list = ast.literal_eval(result[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip().split(',\n\n') + "]}")
#result_dict_list = [json.loads(x) for x in result_str_list]

#for item in result_dict_list:
#  pprint(item)
#  print('\n\n')