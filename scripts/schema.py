from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocType(str, Enum):
    TEN_K = "10-k"
    TEN_Q = "10-q"
    EIGHT_K = "8-k"


class FiscalQuarter(str, Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


class ChunkMetadata(BaseModel):
    company_name: Optional[str] = Field(
        default=None,
        description="Company name (lowercase, eg. 'amazon', 'apple', 'google',...)",
    )

    doc_type: Optional[DocType] = Field(
        default=None, description="Document type (10-k, 10-q, 8-k, etc.)"
    )

    fiscal_year: Optional[str] = Field(
        default=None, description="Fiscal year of the document like '2024', '2023', etc"
    )

    fiscal_quarter: Optional[FiscalQuarter] = Field(
        default=None, description="Fiscal quarter (q1-q4) if applicable"
    )

    model_config = {"use_enum_values": True}
