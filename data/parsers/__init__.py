"""
Dataset Parsers

Provides parsers for different anomaly detection benchmark datasets:
    - SMDParser: Server Machine Dataset (txt format)
    - MSLSMAPParser: MSL/SMAP datasets (npy format)
    - PSMParser: PSM dataset (csv format)
"""

# Lazy imports - modules will be imported when accessed
__all__ = [
    'BaseParser',
    'SMDParser',
    'MSLSMAPParser',
    'PSMParser',
]

def __getattr__(name):
    """Lazy import for parser modules."""
    if name == 'BaseParser':
        from .base import BaseParser
        return BaseParser
    elif name == 'SMDParser':
        from .smd import SMDParser
        return SMDParser
    elif name == 'MSLSMAPParser':
        from .msl_smap import MSLSMAPParser
        return MSLSMAPParser
    elif name == 'PSMParser':
        from .psm import PSMParser
        return PSMParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
