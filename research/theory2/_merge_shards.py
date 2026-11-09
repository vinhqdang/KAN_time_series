"""Concatenate gzip csv shards (same header) into one gzip csv, in sorted file-name order.
Used by reproduce.sh for b1/ and b1nl/.  Usage: python3 -I _merge_shards.py "<glob>" <out.csv.gz>"""
import glob, gzip, sys

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    files = sorted(f for f in glob.glob(sys.argv[1]) if not f.endswith(".part"))
    if not files: sys.exit(f"no shards match {sys.argv[1]}")
    header, rows = None, 0
    with gzip.open(sys.argv[2], "wb") as out:
        for f in files:
            lines = gzip.open(f, "rb").read().splitlines(keepends=True)
            if header is None:
                header = lines[0]; out.write(header)
            assert lines[0] == header, f"header mismatch in {f}"
            out.writelines(lines[1:]); rows += len(lines) - 1
    print(f"{len(files)} shards, {rows} rows -> {sys.argv[2]}")
