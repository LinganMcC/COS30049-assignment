"""Download the official English HC3 all.jsonl file from Hugging Face."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


HC3_URL = (
    "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
    "?download=true"
)


def download_hc3(output: str | Path, overwrite: bool = False) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not overwrite:
        print(f"HC3 already exists: {output}")
        print("Use --overwrite if you intentionally want to download it again.")
        return output

    print("Downloading official HC3 all.jsonl...")
    print(f"Destination: {output}")

    request = urllib.request.Request(
        HC3_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(request) as response, output.open("wb") as file_out:
        shutil.copyfileobj(response, file_out, length=1024 * 1024)

    print(f"Finished: {output} ({output.stat().st_size / 1024 / 1024:.1f} MB)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/hc3_all.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download_hc3(args.out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
