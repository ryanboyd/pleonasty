import csv
import os.path
from tqdm import tqdm

def batch_analyze_to_csv(self,
                         texts: list,
                         text_metadata: dict = {},
                         csv_output_location: str = "AnalysisResults.csv",
                         append_to_existing_csv: bool = False,
                         output_encoding: str = "utf-8-sig",
                         max_seq_length: int = 4096,
                         temperature: float = 0.3,
                         top_k: int = 10,
                         ) -> None:

    if not os.path.exists(os.path.dirname(os.path.abspath(csv_output_location))):
        os.makedirs(os.path.dirname(os.path.abspath(csv_output_location)))
        

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