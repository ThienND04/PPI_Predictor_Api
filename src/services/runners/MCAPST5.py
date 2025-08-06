import subprocess
from .base import BaseModelRunner

class MCAPST5Runner(BaseModelRunner):
    def predict(self, id1: str, seq1: str, id2: str, seq2: str) -> float:
        result = subprocess.run(
            [
                "python", "ml_models/MCAPST5/MCAPST5-X_inference.py",
                "--id1", id1,
                "--seq1", seq1,
                "--id2", id2,
                "--seq2", seq2
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return float(result.stdout.strip().splitlines()[-1])
