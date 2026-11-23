"""List the contents of the proofread-neuron SWC zip without extracting it.

Put this script next to 104_proofread_neurons_swc.zip
and run it from anywhere:

    python h01_zip_names.py        (Windows)
    python h01_zip_names.py                        (Mac or Linux)

Or give the zip path yourself:  python h01_zip_names.py path\\to\\file.zip
Needs nothing beyond standard Python. Writes zip_names.txt next to the script.
"""
import os
import sys
import zipfile

ZIP_NAME = "104_proofread_neurons_swc.zip"
here = os.path.dirname(os.path.abspath(__file__))
downloads = os.path.join(os.path.expanduser("~"), "Downloads")

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    candidates = [os.path.join(here, ZIP_NAME), os.path.join(downloads, ZIP_NAME)]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print(f"Could not find {ZIP_NAME}. Looked in:")
        for p in candidates:
            print("  ", p)
        print("Put the zip next to the script, or pass its full path as an argument.")
        sys.exit(1)

print(f"Reading: {path}  ({os.path.getsize(path):,} bytes)")
with zipfile.ZipFile(path) as z:
    infos = [i for i in z.infolist() if not i.is_dir()]
    names = [i.filename for i in infos]
    print(f"Files in zip: {len(names)}")
    print("First 15 names:")
    for nm in names[:15]:
        print("  ", nm)
    out_path = os.path.join(here, "zip_names.txt")
    with open(out_path, "w") as out:
        out.write("\n".join(names) + "\n")
    print(f"All names saved to {out_path}")
    if infos:
        first = infos[0]
        head = z.read(first.filename)[:600].decode("utf-8", errors="replace")
        print(f"\nFirst 600 characters of {first.filename}:")
        print(head)
