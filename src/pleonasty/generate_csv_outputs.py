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