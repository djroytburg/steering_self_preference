#!/usr/bin/env python3
"""
Extract top-level 'data' array from a JSON file and write each element as one JSON line (.jsonl).
Usage:
  python scripts/extract_data_jsonl.py /path/to/input.json /path/to/output.jsonl
If output path omitted, writes next to input file with suffix '_data.jsonl'.

Approach:
  1) Try json.load() — works if file is valid JSON.
  2) If ijson is available, stream items from 'data' to avoid loading whole array.
  3) Fallback: scan file for "\"data\" *: *[" and bracket-match to extract the array substring then json.loads it.

This script is defensive and prints useful diagnostics.
"""
import sys
import os
import json
from pathlib import Path


def write_jsonl_from_iter(it, out_path):
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in it:
            out.write(json.dumps(item, ensure_ascii=False))
            out.write("\n")
            written += 1
    return written


def try_json_load(in_path, out_path):
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "data" not in obj:
            print("Loaded JSON but no top-level 'data' key found.")
            return None
        data = obj["data"]
        if not isinstance(data, list):
            print("Top-level 'data' is not a list (type=%s)." % type(data).__name__)
            return None
        n = write_jsonl_from_iter(data, out_path)
        print(f"Wrote {n} items (loaded whole file into memory).")
        return n
    except Exception as e:
        print("json.load failed:", e)
        return None


def try_ijson_stream(in_path, out_path):
    try:
        import ijson
    except Exception:
        return None
    try:
        with open(in_path, "rb") as f:
            # ijson.items(f, 'data.item') yields each item under data array
            it = ijson.items(f, 'data.item')
            n = write_jsonl_from_iter(it, out_path)
            print(f"Wrote {n} items (streamed via ijson).")
            return n
    except Exception as e:
        print("ijson streaming failed:", e)
        return None


def fallback_bracket_extract(in_path, out_path):
    # Locate the first occurrence of "\"data\"" then the first '[' after it.
    # Then perform bracket counting to find the matching closing ']' and parse that substring.
    try:
        with open(in_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        print("Could not read file for fallback:", e)
        return None
    idx = text.find('"data"')
    if idx == -1:
        print("Could not find '\"data\"' in file text.")
        return None
    # find first '[' after idx
    start = text.find('[', idx)
    if start == -1:
        print("Found 'data' but no '[' after it.")
        return None
    i = start
    depth = 0
    end = None
    L = len(text)
    # bracket-match; handle string escapes lightly (not full JSON tokenizer) — good enough for many cases
    in_string = False
    esc = False
    while i < L:
        ch = text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1
    if end is None:
        print("Failed to find matching closing ']' for data array.")
        return None
    arr_text = text[start:end]
    try:
        data = json.loads(arr_text)
    except Exception as e:
        print("json.loads on extracted array failed:", e)
        return None
    if not isinstance(data, list):
        print("Extracted JSON is not a list after parsing (type=%s)." % type(data).__name__)
        return None
    n = write_jsonl_from_iter(data, out_path)
    print(f"Wrote {n} items (parsed extracted array substring).")
    return n


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_data_jsonl.py <input.json> [output.jsonl]")
        return 2
    in_path = Path(sys.argv[1])
    if not in_path.exists():
        print("Input path does not exist:", in_path)
        return 2
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.parent / (in_path.stem + "_data.jsonl")

    print("Input:", in_path)
    print("Output:", out_path)

    # Try json.load first
    result = try_json_load(str(in_path), str(out_path))
    if result is not None:
        return 0

    # Try streaming with ijson (memory friendly)
    result = try_ijson_stream(str(in_path), str(out_path))
    if result is not None:
        return 0

    # Fallback: extract bracketed array substring and parse
    result = fallback_bracket_extract(str(in_path), str(out_path))
    if result is not None:
        return 0

    print("All extraction attempts failed. You can try installing ijson (pip install ijson) or inspect the file manually.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
