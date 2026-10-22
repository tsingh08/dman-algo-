# Type-only override for alpaca-py 0.43.2. Pylance reads this instead of the real
# module. Identical to alpaca/trading/client.py except return types drop the
# '| RawData' (dict) branch, which only exists with raw_data=True -- dman_algo
# never sets it. Regenerate after upgrading alpaca-py. No runtime effect.
import json
import warnings
from typing import List, Optional, Union
from uuid import UUID
from pydantic import TypeAdapter
from alpaca.common import RawData
from alpaca.common.enums import BaseURL
from alpaca.common.rest import RESTClient
from alpaca.common.utils import validate_symbol_or_asset_id, validate_symbol_or_contract_id, validate_uuid_id_param
from alpaca.trading.models import AccountConfiguration, Asset, Calendar, Clock, ClosePositionResponse, CorporateActionAnnouncement, OptionContract, OptionContractsResponse, Order, PortfolioHistory, Position, TradeAccount, Watchlist
from alpaca.trading.requests import CancelOrderResponse, ClosePositionRequest, CreateWatchlistRequest, GetAssetsRequest, GetCalendarRequest, GetCorporateAnnouncementsRequest, GetOptionContractsRequest, GetOrderByIdRequest, GetOrdersRequest, GetPortfolioHistoryRequest, OrderRequest, ReplaceOrderRequest, UpdateWatchlistRequest

class TradingClient(RESTClient):

    def __init__(self, api_key: Optional[str]=None, secret_key: Optional[str]=None, oauth_token: Optional[str]=None, paper: bool=True, raw_data: bool=False, url_override: Optional[str]=None) -> None:
        ...

    def submit_order(self, order_data: OrderRequest) -> Order:
        ...

    def get_orders(self, filter: Optional[GetOrdersRequest]=None) -> List[Order]:
        ...

    def get_order_by_id(self, order_id: Union[UUID, str], filter: Optional[GetOrderByIdRequest]=None) -> Order:
        ...

    def get_order_by_client_id(self, client_id: str) -> Order:
        ...

    def replace_order_by_id(self, order_id: Union[UUID, str], order_data: Optional[ReplaceOrderRequest]=None) -> Order:
        ...

    def cancel_orders(self) -> List[CancelOrderResponse]:
        ...

    def cancel_order_by_id(self, order_id: Union[UUID, str]) -> None:
        ...

    def get_all_positions(self) -> List[Position]:
        ...

    def get_open_position(self, symbol_or_asset_id: Union[UUID, str]) -> Position:
        ...

    def close_all_positions(self, cancel_orders: Optional[bool]=None) -> List[ClosePositionResponse]:
        ...

    def close_position(self, symbol_or_asset_id: Union[UUID, str], close_options: Optional[ClosePositionRequest]=None) -> Order:
        ...

    def exercise_options_position(self, symbol_or_contract_id: Union[UUID, str]) -> None:
        ...

    def get_portfolio_history(self, history_filter: Optional[GetPortfolioHistoryRequest]=None) -> PortfolioHistory:
        ...

    def get_all_assets(self, filter: Optional[GetAssetsRequest]=None) -> List[Asset]:
        ...

    def get_asset(self, symbol_or_asset_id: Union[UUID, str]) -> Asset:
        ...

    def get_clock(self) -> Clock:
        ...

    def get_calendar(self, filters: Optional[GetCalendarRequest]=None) -> List[Calendar]:
        ...

    def get_account(self) -> TradeAccount:
        ...

    def get_account_configurations(self) -> AccountConfiguration:
        ...

    def set_account_configurations(self, account_configurations: AccountConfiguration) -> AccountConfiguration:
        ...

    def get_watchlists(self) -> List[Watchlist]:
        ...

    def get_watchlist_by_id(self, watchlist_id: Union[UUID, str]) -> Watchlist:
        ...

    def create_watchlist(self, watchlist_data: CreateWatchlistRequest) -> Watchlist:
        ...

    def update_watchlist_by_id(self, watchlist_id: Union[UUID, str], watchlist_data: UpdateWatchlistRequest) -> Watchlist:
        ...

    def add_asset_to_watchlist_by_id(self, watchlist_id: Union[UUID, str], symbol: str) -> Watchlist:
        ...

    def delete_watchlist_by_id(self, watchlist_id: Union[UUID, str]) -> None:
        ...

    def remove_asset_from_watchlist_by_id(self, watchlist_id: Union[UUID, str], symbol: str) -> Watchlist:
        ...

    def get_corporate_announcements(self, filter: GetCorporateAnnouncementsRequest) -> List[CorporateActionAnnouncement]:
        ...

    def get_corporate_announcement_by_id(self, corporate_announcment_id: Union[UUID, str]) -> CorporateActionAnnouncement:
        ...

    def get_option_contracts(self, request: GetOptionContractsRequest) -> OptionContractsResponse:
        ...

    def get_option_contract(self, symbol_or_id: Union[UUID, str]) -> OptionContract:
        ...
