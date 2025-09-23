from pydantic import BaseModel, Field, field_validator
from typing import Optional
import re

class PredictInput(BaseModel):
    id1: Optional[str] = "PROTEIN1"
    seq1: str = Field(..., description="Chuỗi amino acid của protein")
    id2: Optional[str] = "PROTEIN2"
    seq2: str = Field(..., description="Chuỗi amino acid của protein")
    model: str = "MCAPST5"
    
    @field_validator('seq1', 'seq2')
    def validate_protein_sequence(cls, v):
        # Kiểm tra chỉ chứa amino acid hợp lệ
        valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(c.upper() in valid_amino_acids for c in v.replace(' ', '')):
            raise ValueError('Sequence contains invalid amino acid characters')
        return v.upper().replace(' ', '')
    
    @field_validator('id1', 'id2')
    def validate_protein_id(cls, v):
        if v and not re.match(r'^[\\.A-Za-z0-9_-]+$', v):
            raise ValueError('Protein ID contains invalid characters')
        return v