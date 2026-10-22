# Type-only override for alpaca-py 0.43.2 alpaca/data/timeframe.py. The real
# TimeFrame.Day/Minute/... are built with an untyped `classproperty` decorator,
# so Pylance sees a descriptor where a TimeFrame is returned. No runtime effect.
from enum import Enum
from typing import ClassVar

class TimeFrameUnit(str, Enum):
    Minute = "Min"
    Hour = "Hour"
    Day = "Day"
    Week = "Week"
    Month = "Month"

class TimeFrame:
    Minute: ClassVar[TimeFrame]
    Hour: ClassVar[TimeFrame]
    Day: ClassVar[TimeFrame]
    Week: ClassVar[TimeFrame]
    Month: ClassVar[TimeFrame]
    amount_value: int
    unit_value: TimeFrameUnit
    def __init__(self, amount: int, unit: TimeFrameUnit) -> None: ...
    @property
    def amount(self) -> int: ...
    @property
    def unit(self) -> TimeFrameUnit: ...
    @property
    def value(self) -> str: ...
    @staticmethod
    def validate_timeframe(amount: int, unit: TimeFrameUnit) -> None: ...
