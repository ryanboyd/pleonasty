import csv
import sys
import os.path
from tqdm import tqdm

def set_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 2
            if limit < 128 * 1024 * 1024:  # don’t go below ~128MB
                csv.field_size_limit(128 * 1024 * 1024)
                break

set_csv_field_limit()

def _clean_lines(iterable, drop_nul=True):
    """Yield sanitized text lines for csv.reader."""
    for line in iterable:
        if drop_nul and '\x00' in line:
            line = line.replace('\x00', '')
        yield line

def batch_analyze_csv_to_csv(self,
                         input_csv: str,
                         text_columns_to_process: list = [],
                         metadata_columns_to_retain: list = [],
                         start_at_row: int = 0,
                         output_csv: str = "AnalysisResults.csv",
                         append_to_existing_csv: bool = False,
                         file_encoding: str = "utf-8-sig",
                         chunk_into_n_tokens: int = 2000,
                         **sampling_params) -> None:

    """
    :param csv_input_location: The path to the CSV file that we want to process
    :param columns_to_process: The header names, or indices, of the columns containing text that we want to analyze
    :param metadata_columns_to_retain: The header names, or indices, of the metadata columns that we want to retain for our output
    :param start_at_row: The row of the input CSV where we want to start processing.
    :param csv_output_location: The path to where we want our output to be stored
    :param append_to_existing_csv: Do you want to append the output to an existing CSV file?
    :param file_encodings: The file encoding to be used for both the input and output CSV files.
    :return:
    """

    # Verify that the local file does, in fact, exist
    if not os.path.isfile(os.path.abspath(input_csv)):

        print("The input file that you are trying to use does not appear to exist. Please check the file location.")
        return

    if not os.path.exists(os.path.dirname(os.path.abspath(output_csv))):
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)))

    # Check if columns_to_process are indices or names
    useFileHeader = True
    header_row = None
    index_boundaries = {"max": None, "min": None}

    if all(isinstance(col, int) for col in text_columns_to_process):
        useFileHeader = False
        index_boundaries["max"] = max(text_columns_to_process + metadata_columns_to_retain)
        index_boundaries["min"] = min(text_columns_to_process + metadata_columns_to_retain)

    # Count the number of rows in the CSV file:
    print("Checking input file integrity and counting the number of rows...")
    numRows = 0
    with open(input_csv, 'r', encoding=file_encoding) as fin:
        csvr = csv.reader(_clean_lines(fin))

        if useFileHeader:
            header_row = csvr.__next__()
            missing_headers = []
            for input_col in text_columns_to_process:
                if input_col not in header_row:
                    missing_headers.append(input_col)
            if len(missing_headers) > 0:
                print("The following columns could not be found in your dataset's header: " + " ".join(missing_headers))
                return

        # Also, double-check that the user-specified indices are found in the data
        for line in csvr:
            numRows += 1
            if numRows % 1000000 == 0:
                print(f"Counted {numRows} rows so far...", flush=True)
            if not useFileHeader:
                if index_boundaries["min"] < 0 or index_boundaries["max"] > len(line):
                    print(f"At least one of your column indices is outside of the range of columns in your dataset: Row {numRows}")
                    return


    print(f"{numRows} rows detected.")

    print("Beginning analysis...")

    with open(input_csv, 'r', encoding=file_encoding) as fin:
        csvr = csv.reader(_clean_lines(fin))

        column_indices_to_process = []
        column_indices_for_metadata = []

        if useFileHeader:
            header_row = csvr.__next__()
            for item in text_columns_to_process:
                column_indices_to_process.append(header_row.index(item))
            for item in metadata_columns_to_retain:
                column_indices_for_metadata.append(header_row.index(item))
        else:
            column_indices_to_process = text_columns_to_process
            column_indices_for_metadata = metadata_columns_to_retain


        writemode = 'w'
        if append_to_existing_csv:
            writemode = 'a'

        with open(output_csv, writemode, encoding=file_encoding, newline='') as fout:

            csvw = csv.writer(fout)

            if append_to_existing_csv is False:
                csvw.writerow(self.generate_csv_header(
                    metadata_headers=metadata_columns_to_retain))

            for rowInProgress in tqdm(range(numRows)):

                row_to_process = csvr.__next__()

                if rowInProgress < start_at_row:
                    continue

                text_to_process = " ".join([row_to_process[i] for i in column_indices_to_process])

                # do the actual generation. the result gets saves as a list of LLM_Result()
                # to self.result for the Pleonast class
                chunked_text = self.chunk_by_tokens(text=text_to_process,
                                                    chunk_size=chunk_into_n_tokens)

                results = self.analyze_text(input_texts=chunked_text,
                                            **sampling_params)

                # prep the row output with metadata
                meta_output = [row_to_process[i] for i in column_indices_for_metadata]

                # complete the row output by pulling the results data
                for result in results:
                    row_output = self.generate_csv_output_row(result=result,
                                                              input_metadata=meta_output)

                    #write the output
                    csvw.writerow(row_output)

    print("Analysis complete.")

    return