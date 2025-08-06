from pydantic import BaseModel, Field

class Protein(BaseModel):
    id: str = Field(..., description="ID của protein (ví dụ: P12345)")
    seq: str = Field(..., description="Chuỗi amino acid của protein")


class ProteinPair(BaseModel):
    protein1: Protein
    protein2: Protein
