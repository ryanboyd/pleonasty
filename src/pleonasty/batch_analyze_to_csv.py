import csv
import os.path
from tqdm import tqdm

def batch_analyze_to_csv(self,
                         texts: list,
                         text_metadata: dict = {},
                         output_csv: str = "AnalysisResults.csv",
                         append_to_existing_csv: bool = False,
                         output_csv_encoding: str = "utf-8-sig",
                         max_words_per_chunk: int = 2000,
                         **sampling_params
                         ) -> None:

    if not os.path.exists(os.path.dirname(os.path.abspath(output_csv))):
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)))
        

    writemode = 'w'
    if append_to_existing_csv:
        writemode = 'a'

    with open(output_csv, writemode, encoding=output_csv_encoding, newline='') as fout:

        csvw = csv.writer(fout)
        meta_headers = list(text_metadata.keys())

        if append_to_existing_csv is False:
            csvw.writerow(self.generate_csv_header(
                metadata_headers=meta_headers))

        for i in tqdm(range(len(texts))):

            results = self.analyze_text(input_texts=[texts[i]],
                              **sampling_params)

            # prep the row output with metadata
            meta_output = []
            for meta_item in meta_headers:
                meta_output.append(text_metadata[meta_item][i])

            for result in results:
                row_output = self.generate_csv_output_row(result=result,
                                                              input_metadata=meta_output)
                #write the output
                csvw.writerow(row_output)

    print("Analysis complete.")

    return