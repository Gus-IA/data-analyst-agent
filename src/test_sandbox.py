"""Check that the sandbox blocks network access, writes outside /out, and runaway code."""

from pathlib import Path

from tools import run_python

CSV = str(Path(__file__).resolve().parent.parent / "data" / "sample.csv")

CASES = {
    "network is blocked": "import urllib.request\nurllib.request.urlopen('https://example.com', timeout=5)",
    "root filesystem is read-only": "open('/work/x.txt', 'w').write('hi')",
    "dataset is read-only": "open('/data/data.csv', 'a').write('x')",
    "memory limit (512 MB)": "x = bytearray(2 * 1024**3)",
    "infinite loop is stopped": "while True:\n    pass",
}

for name, code in CASES.items():
    print(f"\n=== {name} ===")
    print(run_python(code, CSV))
