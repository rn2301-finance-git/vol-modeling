#!/usr/bin/env python3
import csv
import sys
from itertools import zip_longest

def main():
    if len(sys.argv) != 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} input.csv\n")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
    except Exception as e:
        sys.stderr.write(f"Error reading {filename}: {e}\n")
        sys.exit(1)
    
    if not rows or len(rows) < 2:
        sys.stderr.write("CSV file must have at least 2 rows.\n")
        sys.exit(1)

    # Get header and first data row (row 2)
    header = rows[0]
    data_row = rows[1]

    # Transpose and print just these two rows
    for pair in zip_longest(header, data_row, fillvalue=''):
        print(f"{pair[0]},{pair[1]}")

if __name__ == '__main__':
    main()