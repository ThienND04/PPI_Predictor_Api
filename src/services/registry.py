from .runners.MCAPST5 import MCAPST5Runner

_model_registry = {
    "MCAPST5": MCAPST5Runner()
}

def get_runner(model_name: str):
    if model_name not in _model_registry:
        raise ValueError(f"Model '{model_name}' not supported.")
    return _model_registry[model_name]
