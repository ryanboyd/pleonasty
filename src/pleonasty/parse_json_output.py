import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

try:
    from json_repair import repair_json as _repair
    _HAS_REPAIR = True
except ImportError:
    _repair = None
    _HAS_REPAIR = False


def _set_csv_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 2


_set_csv_limit()


def _extract_json_str(text: str) -> str:
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    return match.group(0) if match else text


def _try_parse(response_text: str):
    """Return a lowercased-key dict parsed from an LLM response, or None."""
    raw = _extract_json_str(response_text)
    if _HAS_REPAIR:
        raw = _repair(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {k.lower(): v for k, v in parsed.items()}
    except Exception:
        pass
    return None


def _aggregate(value_lists: dict, fields: list) -> dict:
    """
    Collapse multiple per-chunk values into one aggregated value per field.

    Aggregation strategy is inferred from the type of the first non-None value:
      bool / int / float  → mean (float)
      list                → concatenated list, JSON-serialised
      str                 → newline-joined
    """
    result = {}
    for f in fields:
        vals = [v for v in value_lists.get(f, []) if v is not None]
        if not vals:
            result[f] = None
            continue
        sample = vals[0]
        if isinstance(sample, bool):
            result[f] = sum(bool(v) for v in vals) / len(vals)
        elif isinstance(sample, (int, float)):
            result[f] = sum(float(v) for v in vals) / len(vals)
        elif isinstance(sample, list):
            combined = []
            for v in vals:
                if isinstance(v, list):
                    combined.extend(v)
            result[f] = json.dumps(combined)
        else:
            result[f] = '\n'.join(str(v) for v in vals if v)
    return result


def parse_json_output(
    input_csv: str,
    json_fields: list,
    output_csv: str = None,
    response_column: str = "LLM_Response",
    group_by=None,
    encoding: str = "utf-8-sig",
) -> str:
    """
    Parse JSON fields out of an LLM response column in a pleonasty output CSV
    and write a new CSV with those fields added as extra columns.

    Parameters
    ----------
    input_csv : str
        Path to a CSV file produced by batch_analyze_to_csv or
        batch_analyze_csv_to_csv.
    json_fields : list[str]
        JSON keys to extract from each LLM response (case-insensitive).
    output_csv : str, optional
        Where to write the result.  Defaults to <input_stem>_parsed.csv.
    response_column : str
        Name of the column holding the LLM response text (default: LLM_Response).
    group_by : str or list[str], optional
        Column name(s) to group rows by before aggregating.  Use this when a
        single source document was split into chunks — all chunks that share the
        same group key are collapsed into one output row.

        Aggregation rules:
          - numeric values  → mean across chunks
          - list values     → all sub-lists concatenated (JSON-serialised)
          - string values   → joined with newlines

        A ``num_chunks`` column is added to the output showing how many rows
        were merged.
    encoding : str
        File encoding for both input and output (default: utf-8-sig).

    Returns
    -------
    str
        Path of the output CSV that was written.
    """
    if not _HAS_REPAIR:
        print("Tip: install json-repair for more robust JSON parsing "
              "(pip install json-repair).")

    if output_csv is None:
        p = Path(input_csv)
        output_csv = str(p.with_stem(p.stem + "_parsed"))

    if isinstance(group_by, str):
        group_by = [group_by]

    fields = [f.lower() for f in json_fields]

    # ── read ───────────────────────────────────────────────────────────────────
    with open(input_csv, 'r', encoding=encoding) as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header row.")
        in_fields = list(reader.fieldnames)
        if response_column not in in_fields:
            raise ValueError(
                f"Column '{response_column}' not found. "
                f"Available: {in_fields}"
            )
        if group_by:
            missing = [c for c in group_by if c not in in_fields]
            if missing:
                raise ValueError(f"group_by column(s) not found: {missing}")
        rows = list(reader)

    # ── parse JSON from each row ───────────────────────────────────────────────
    parsed_dicts = []
    n_ok = n_fail = 0
    for row in rows:
        d = _try_parse(row[response_column])
        if d is not None:
            n_ok += 1
        else:
            d = {}
            n_fail += 1
        parsed_dicts.append(d)

    print(f"JSON parsing: {n_ok} succeeded, {n_fail} failed.")

    # ── assemble output ────────────────────────────────────────────────────────
    if group_by:
        groups: dict = OrderedDict()
        for row, pd_ in zip(rows, parsed_dicts):
            key = tuple(row[c] for c in group_by)
            if key not in groups:
                groups[key] = {
                    'meta': row,
                    'values': {f: [] for f in fields},
                    'count': 0,
                }
            for f in fields:
                groups[key]['values'][f].append(pd_.get(f))
            groups[key]['count'] += 1

        out_fieldnames = in_fields + fields + ['num_chunks']
        out_rows = []
        for grp in groups.values():
            agg = _aggregate(grp['values'], fields)
            out_row = dict(grp['meta'])
            for f in fields:
                out_row[f] = agg.get(f)
            out_row['num_chunks'] = grp['count']
            out_rows.append(out_row)
    else:
        out_fieldnames = in_fields + fields
        out_rows = []
        for row, pd_ in zip(rows, parsed_dicts):
            out_row = dict(row)
            for f in fields:
                out_row[f] = pd_.get(f)
            out_rows.append(out_row)

    with open(output_csv, 'w', encoding=encoding, newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Written to: {output_csv}")
    return output_csv
