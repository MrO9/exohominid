"""Print the schema and the first few rows of one Avro file.

Usage:  python h01_peek.py path/to/file.avro [number_of_rows]
Needs:  pip install fastavro
"""
import json
import sys

from fastavro import reader

path = sys.argv[1]
n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5

with open(path, "rb") as f:
    r = reader(f)
    print("=== SCHEMA (the column names and types) ===")
    print(json.dumps(r.writer_schema, indent=2))
    print()
    print(f"=== FIRST {n_rows} ROWS ===")
    for i, row in enumerate(r):
        print(row)
        if i + 1 >= n_rows:
            break
