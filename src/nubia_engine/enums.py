"""Closed domain categories used by the NUBIA foundation."""

from enum import StrEnum


class Empire(StrEnum):
    """One of the two opposing empires."""

    A = "A"
    B = "B"


class Terrain(StrEnum):
    """A board square's terrain."""

    LAND = "LAND"
    SEA = "SEA"


class RowKind(StrEnum):
    """The role of a row within its owning empire's half."""

    PALACE = "PALACE"
    MARKET = "MARKET"
    TOWN_CENTER = "TOWN_CENTER"
    OUTSKIRTS = "OUTSKIRTS"
    BORDER = "BORDER"


class PieceType(StrEnum):
    """Gameplay-distinct piece types."""

    IMPERION = "IMPERION"
    QUEEN = "QUEEN"
    NORTH_CENTRAL_WAR_CHIEF = "NORTH_CENTRAL_WAR_CHIEF"
    EAST_AFRICAN_HIGH_CHIEF = "EAST_AFRICAN_HIGH_CHIEF"
    SOUTH_AFRICAN_ADVISOR = "SOUTH_AFRICAN_ADVISOR"
    WEST_AFRICAN_MYSTIC = "WEST_AFRICAN_MYSTIC"
    PEASANT = "PEASANT"


class BrainwashAvailability(StrEnum):
    """A Mystic's one-use Brainwash power state."""

    AVAILABLE = "AVAILABLE"
    SPENT = "SPENT"


class ActionKind(StrEnum):
    """An ordinary action supported by the Milestone 2 engine."""

    MOVE = "MOVE"
    CAPTURE = "CAPTURE"
    SWITCH = "SWITCH"
