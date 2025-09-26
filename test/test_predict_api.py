import pytest
from src.api.schemas.PredictInput import PredictInput
from src.services.runners.MCAPST5 import MCAPST5Runner

def test_predict_input_validation():
    # Test valid input
    valid_input = {
        "seq1": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "seq2": "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLDEVSEQLVQLLKRKLEEQAQDLP"
    }
    data = PredictInput(**valid_input)
    assert data.seq1 == valid_input["seq1"]
    assert data.model == "MCAPST5"  # default value
    
    # Test invalid amino acid
    with pytest.raises(ValueError):
        PredictInput(seq1="INVALID123", seq2="VALID")

def test_mcapst5_runner():
    runner = MCAPST5Runner()
    assert runner.model_path.exists()