"""
Data source implementations for fetching financial data from various APIs.
"""

from .base import DataSource
from .cryptocompare import CryptoCompareSource
from .kraken import KrakenSource
from .fred import FREDSource

__all__ = [
    'DataSource',
    'CryptoCompareSource',
    'KrakenSource',
    'FREDSource'
]
