# Type-only override for alpaca-py 0.43.2. Pylance reads this instead of the real
# module. Identical to alpaca/data/historical/stock.py except return types drop the
# '| RawData' (dict) branch, which only exists with raw_data=True -- dman_algo
# never sets it. Regenerate after upgrading alpaca-py. No runtime effect.
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Union
from alpaca.common.constants import DATA_V2_MAX_LIMIT
from alpaca.common.enums import BaseURL
from alpaca.common.rest import RESTClient
from alpaca.common.types import RawData
from alpaca.data import Bar, Quote, Snapshot, Trade
from alpaca.data.historical.utils import parse_obj_as_symbol_dict
from alpaca.data.models import BarSet, QuoteSet, TradeSet
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest, StockLatestQuoteRequest, StockLatestTradeRequest, StockQuotesRequest, StockSnapshotRequest, StockTradesRequest

class StockHistoricalDataClient(RESTClient):

    def __init__(self, api_key: Optional[str]=None, secret_key: Optional[str]=None, oauth_token: Optional[str]=None, use_basic_auth: bool=False, raw_data: bool=False, url_override: Optional[str]=None, sandbox: bool=False) -> None:
        ...

    def get_stock_bars(self, request_params: StockBarsRequest) -> BarSet:
        ...

    def get_stock_quotes(self, request_params: StockQuotesRequest) -> QuoteSet:
        ...

    def get_stock_trades(self, request_params: StockTradesRequest) -> TradeSet:
        ...

    def get_stock_latest_trade(self, request_params: StockLatestTradeRequest) -> Dict[str, Trade]:
        ...

    def get_stock_latest_quote(self, request_params: StockLatestQuoteRequest) -> Dict[str, Quote]:
        ...

    def get_stock_latest_bar(self, request_params: StockLatestBarRequest) -> Dict[str, Bar]:
        ...

    def get_stock_snapshot(self, request_params: StockSnapshotRequest) -> Dict[str, Snapshot]:
        ...
