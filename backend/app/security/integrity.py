"""Model file integrity check, run once before the model is loaded.

torch.load and pickle can execute code hidden inside a file. If best_model.pt
or tokenizer.json is swapped (a bad download, a tampered copy in the ZIP), the
SHA-256 no longer matches SHA256SUMS and the server refuses to start rather
than loading it.
"""
import hashlib
import hmac
from pathlib import Path


class ModelIntegrityError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):   # 1 MB at a time
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files(model_dir: Path) -> None:
    """Check every file listed in SHA256SUMS (format written by `shasum -a 256`)."""
    sums_file = model_dir / 'SHA256SUMS'
    if not sums_file.is_file():
        raise ModelIntegrityError(f'{sums_file.name} is missing - refusing to load the model')

    for line in sums_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = model_dir / name.strip()
        if not path.is_file():
            raise ModelIntegrityError(f'{name} is missing - refusing to load the model')
        if not hmac.compare_digest(sha256_of(path), expected.lower()):
            raise ModelIntegrityError(f'{name} does not match its SHA-256 - refusing to load it')
