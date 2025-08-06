from abc import ABC, abstractmethod

class BaseModelRunner(ABC):
    @abstractmethod
    def predict(self, id1: str, seq1: str, id2: str, seq2: str) -> float:
        pass
