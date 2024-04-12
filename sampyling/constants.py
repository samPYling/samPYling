from enum import Enum


class SamplingType(str, Enum):
    """Especifica os tipos de amostragem"""

    srs: str = "srs"
