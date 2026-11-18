"""Tools exposed to the agent."""

import pandas as pd

from sandbox import run_code
from pathlib import Path


def describe_dataset(data_file: str) -> str:
    """Summarize a CSV so the model knows what it is working with."""
    df = pd.read_csv(data_file)
    lines = [
        f"Rows: {len(df)}, columns: {len(df.columns)}",
        "Columns (dtype, missing values):",
    ]
    for col in df.columns:
        lines.append(f"- {col} ({df[col].dtype}, {int(df[col].isna().sum())} missing)")
    lines.append("\nFirst rows:\n" + df.head(5).to_string())
    lines.append("\nNumeric summary:\n" + df.describe().T.to_string())
    return "\n".join(lines)[:6000]


def format_result(result: dict) -> str:
    """Turn a sandbox result into text the model can read."""
    parts = [f"Status: {'OK' if result['ok'] else 'ERROR'}"]
    if result["stdout"]:
        parts.append("Output:\n" + result["stdout"])
    if result["stderr"]:
        parts.append("Errors:\n" + result["stderr"])
    if result["files"]:
        parts.append("Files created in /out: " + ", ".join(result["files"]))
    return "\n".join(parts)


def run_python(code: str, data_file: str) -> str:
    """Run code in the sandbox. The CSV is at /data/data.csv; save charts in /out."""
    return format_result(run_code(code, data_file))


if __name__ == "__main__":
    CSV = str(Path(__file__).resolve().parent.parent / "data" / "sample.csv")
    print(describe_dataset(CSV)[:800])
    print("\n--- sandbox test ---")
    test_code = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "df = pd.read_csv('/data/data.csv')\n"
        "print(df.shape)\n"
        "df.hist(figsize=(10, 8)); plt.savefig('/out/hist.png')\n"
    )
    print(run_python(test_code, CSV))
