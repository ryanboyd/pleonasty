from .Pleonast import Pleonast
from ._buildPrompt import _buildPrompt
from .analyze_text import analyze_text
from .batch_analyze_to_csv import batch_analyze_to_csv
from .batch_analyze_csv_to_csv import batch_analyze_csv_to_csv
from .chat_mode import chat_mode
from .convert_prompt_to_template_str import convert_prompt_to_template_str
from .generate_csv_outputs import generate_csv_header, generate_csv_output_row
from .process_text import process_text
from .set_message_contexts import set_message_context, set_message_context_from_CSV


# Attach methods to class
Pleonast._buildPrompt = _buildPrompt
Pleonast.analyze_text = analyze_text
Pleonast.batch_analyze_to_csv = batch_analyze_to_csv
Pleonast.batch_analyze_csv_to_csv = batch_analyze_csv_to_csv
Pleonast.chat_mode = chat_mode
Pleonast.set_message_context = set_message_context
Pleonast.convert_prompt_to_template_str = convert_prompt_to_template_str
Pleonast.generate_csv_header = generate_csv_header
Pleonast.generate_csv_output_row = generate_csv_output_row
Pleonast.process_text = process_text
Pleonast.set_message_context = set_message_context
Pleonast.set_message_context_from_CSV = set_message_context_from_CSV

__all__ = ['Pleonast']