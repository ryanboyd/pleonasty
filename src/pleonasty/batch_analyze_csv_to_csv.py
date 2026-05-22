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
            if limit < 128 * 1024 * 1024:
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

    if not os.path.isfile(os.path.abspath(input_csv)):
        print("The input file that you are trying to use does not appear to exist. Please check the file location.")
        return

    if not os.path.exists(os.path.dirname(os.path.abspath(output_csv))):
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)))

    useFileHeader = True
    header_row = None
    index_boundaries = {"max": None, "min": None}

    if all(isinstance(col, int) for col in text_columns_to_process):
        useFileHeader = False
        index_boundaries["max"] = max(text_columns_to_process + metadata_columns_to_retain)
        index_boundaries["min"] = min(text_columns_to_process + metadata_columns_to_retain)

    print("Checking input file integrity and counting the number of rows...")
    numRows = 0
    with open(input_csv, 'r', encoding=file_encoding) as fin:
        csvr = csv.reader(_clean_lines(fin))

        if useFileHeader:
            header_row = csvr.__next__()
            missing_headers = [c for c in text_columns_to_process if c not in header_row]
            if missing_headers:
                print("The following columns could not be found in your dataset's header: " + " ".join(missing_headers))
                return

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

    # Number of rows to accumulate before each analyze_text call.
    # batch_size is passed through to analyze_text for the inner chunk batching;
    # we use the same value here so a batch of rows fills one generate() call.
    row_batch_size = max(1, sampling_params.get("batch_size", 1))

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

        writemode = 'a' if append_to_existing_csv else 'w'

        with open(output_csv, writemode, encoding=file_encoding, newline='') as fout:
            csvw = csv.writer(fout)

            if not append_to_existing_csv:
                csvw.writerow(self.generate_csv_header(
                    metadata_headers=metadata_columns_to_retain))

            outer_bar  = tqdm(total=numRows, position=0, desc="Rows")
            status_bar = tqdm(bar_format="  {desc}", position=1, leave=True, desc="—")
            self._status_bar = status_bar

            # Pending rows: each entry is (row_data, meta_output, 1-based row number)
            pending = []
            rows_processed = 0

            def _flush(pending):
                if not pending:
                    return

                # Collect all chunks from all pending rows
                all_chunks   = []
                chunk_counts = []
                all_metadata = []

                for row_data, meta_output, _ in pending:
                    text   = " ".join(row_data[i] for i in column_indices_to_process)
                    chunks = self.chunk_by_tokens(text=text, chunk_size=chunk_into_n_tokens)
                    all_chunks.extend(chunks)
                    chunk_counts.append(len(chunks))
                    all_metadata.append(meta_output)

                first_num = pending[0][2]
                last_num  = pending[-1][2]
                if len(pending) == 1:
                    self._batch_label = f"Row {first_num}/{numRows}"
                else:
                    self._batch_label = f"Rows {first_num}–{last_num}/{numRows}"

                results = self.analyze_text(input_texts=all_chunks, **sampling_params)

                # Distribute results back to their source rows
                result_idx = 0
                for n_chunks, meta_output in zip(chunk_counts, all_metadata):
                    for result in results[result_idx : result_idx + n_chunks]:
                        csvw.writerow(self.generate_csv_output_row(
                            result=result, input_metadata=meta_output))
                    result_idx += n_chunks

                outer_bar.update(len(pending))

            try:
                for rowInProgress in range(numRows):
                    row_data = csvr.__next__()

                    if rowInProgress < start_at_row:
                        outer_bar.update(1)
                        continue

                    meta_output = [row_data[i] for i in column_indices_for_metadata]
                    pending.append((row_data, meta_output, rowInProgress + 1))

                    if len(pending) >= row_batch_size:
                        _flush(pending)
                        pending.clear()

                # Flush any remaining rows
                _flush(pending)

            finally:
                outer_bar.close()
                status_bar.close()
                for attr in ("_status_bar", "_batch_label"):
                    if hasattr(self, attr):
                        delattr(self, attr)

    print("Analysis complete.")
    return
