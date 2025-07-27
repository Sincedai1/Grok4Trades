"""
Data Manager Module

Handles data acquisition, validation, and normalization from various sources
for the QuantumSol backtesting system.
"""

from .data_manager import DataManager
from .sources import CryptoCompareSource, KrakenSource, FREDSource
from .utils import validate_data, normalize_data, convert_currency

__all__ = [
    'DataManager',
    'CryptoCompareSource',
    'KrakenSource',
    'FREDSource',
    'validate_data',
    'normalize_data',
    'convert_currency'
]
