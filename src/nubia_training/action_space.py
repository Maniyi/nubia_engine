"""Versioned, perspective-normalized fixed action indexing for NUBIA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from nubia_engine import Action, ActionKind, Empire, GameState, legal_actions
from nubia_training.errors import (
    ActionIndexCollisionError,
    IllegalOrStaleActionError,
    InvalidActionIndexError,
    UnresolvedActionIndexError,
    UnsupportedActionKindError,
)

ACTION_SPACE_VERSION = 1
BOARD_SQUARE_COUNT = 100
ACTION_KIND_BLOCK_SIZE = BOARD_SQUARE_COUNT * BOARD_SQUARE_COUNT

# This order is a compatibility contract. It deliberately does not use enum order.
ACTION_KIND_ORDER: tuple[ActionKind, ...] = (
    ActionKind.MOVE,
    ActionKind.CAPTURE,
    ActionKind.SWITCH,
    ActionKind.GBESELE,
    ActionKind.BRAINWASH,
    ActionKind.REBRAINWASH,
)
ACTION_KIND_OFFSETS: tuple[tuple[ActionKind, int], ...] = tuple(
    (kind, ordinal * ACTION_KIND_BLOCK_SIZE)
    for ordinal, kind in enumerate(ACTION_KIND_ORDER)
)
ACTION_SPACE_SIZE = len(ACTION_KIND_ORDER) * ACTION_KIND_BLOCK_SIZE


@dataclass(frozen=True, slots=True)
class ActionSpaceSpec:
    """Immutable public metadata for action-space version 1."""

    version: int
    action_kind_order: tuple[str, ...]
    action_kind_offsets: tuple[int, ...]
    size: int
    formula: str
    orientation: str
    index_dtype: str


ACTION_SPACE_SPEC = ActionSpaceSpec(
    version=ACTION_SPACE_VERSION,
    action_kind_order=tuple(kind.value for kind in ACTION_KIND_ORDER),
    action_kind_offsets=tuple(offset for _kind, offset in ACTION_KIND_OFFSETS),
    size=ACTION_SPACE_SIZE,
    formula="kind_offset + normalized_source * 100 + normalized_destination",
    orientation="Empire A canonical; Empire B rotated 180 degrees",
    index_dtype="uint16",
)

_KIND_ORDINAL = {kind: ordinal for ordinal, kind in enumerate(ACTION_KIND_ORDER)}


def _validate_action_kinds() -> None:
    if set(ACTION_KIND_ORDER) != set(ActionKind) or len(ACTION_KIND_ORDER) != len(
        ActionKind
    ):
        raise UnsupportedActionKindError(
            "ACTION_KIND_ORDER must explicitly contain every engine ActionKind"
        )


_validate_action_kinds()


def _perspective_or_active(state: GameState, perspective: Empire | None) -> Empire:
    if not isinstance(state, GameState):
        raise TypeError("state must be a GameState")
    if perspective is None:
        return state.side_to_move
    if not isinstance(perspective, Empire):
        raise TypeError("perspective must be an Empire or None")
    return perspective


def _normalized_square_index(row: int, column: int, perspective: Empire) -> int:
    canonical = row * 10 + column
    return canonical if perspective is Empire.A else 99 - canonical


def _structural_index(action: Action, perspective: Empire) -> int:
    try:
        kind_ordinal = _KIND_ORDINAL[action.kind]
    except KeyError as error:
        raise UnsupportedActionKindError(
            f"unsupported action kind: {action.kind!r}"
        ) from error
    source = _normalized_square_index(
        action.source.row, action.source.column, perspective
    )
    destination = _normalized_square_index(
        action.destination.row, action.destination.column, perspective
    )
    return kind_ordinal * ACTION_KIND_BLOCK_SIZE + source * 100 + destination


def _indexed_legal_actions(
    state: GameState, perspective: Empire
) -> tuple[tuple[int, Action], ...]:
    indexed = tuple(
        (_structural_index(action, perspective), action)
        for action in legal_actions(state)
    )
    seen: dict[int, Action] = {}
    for index, action in indexed:
        previous = seen.get(index)
        if previous is not None:
            raise ActionIndexCollisionError(
                f"legal actions {previous!r} and {action!r} collide at index {index}"
            )
        seen[index] = action
    return indexed


def action_to_index(
    state: GameState,
    action: Action,
    *,
    perspective: Empire | None = None,
) -> int:
    """Return the fixed structural index for a current legal action."""

    normalized_for = _perspective_or_active(state, perspective)
    if not isinstance(action, Action):
        raise IllegalOrStaleActionError("action must be an Action")
    actions = legal_actions(state)
    if action not in actions:
        raise IllegalOrStaleActionError(
            "action is illegal, forged, or stale for the supplied state"
        )
    index = _structural_index(action, normalized_for)
    matches = tuple(
        candidate
        for candidate in actions
        if _structural_index(candidate, normalized_for) == index
    )
    if len(matches) != 1:
        raise ActionIndexCollisionError(
            f"expected one legal action at index {index}, found {len(matches)}"
        )
    return index


def index_to_action(
    state: GameState,
    index: int,
    *,
    perspective: Empire | None = None,
) -> Action:
    """Resolve an index to its exact current engine-generated legal action."""

    normalized_for = _perspective_or_active(state, perspective)
    if isinstance(index, bool) or not isinstance(index, int):
        raise InvalidActionIndexError("action index must be an integer, not Boolean")
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise InvalidActionIndexError(
            f"action index must be between 0 and {ACTION_SPACE_SIZE - 1}"
        )
    matches = tuple(
        action
        for candidate_index, action in _indexed_legal_actions(state, normalized_for)
        if candidate_index == index
    )
    if not matches:
        raise UnresolvedActionIndexError(
            f"index {index} does not resolve to a legal action in this state"
        )
    if len(matches) > 1:  # Defensive: _indexed_legal_actions already checks this.
        raise ActionIndexCollisionError(
            f"index {index} resolves to {len(matches)} legal actions"
        )
    return matches[0]


def legal_action_indices(
    state: GameState, *, perspective: Empire | None = None
) -> tuple[int, ...]:
    """Return legal structural indices in canonical engine action order."""

    normalized_for = _perspective_or_active(state, perspective)
    return tuple(
        index for index, _action in _indexed_legal_actions(state, normalized_for)
    )


def legal_action_mask(
    state: GameState, *, perspective: Empire | None = None
) -> NDArray[np.bool_]:
    """Return an owned, read-only dense Boolean mask of current legal actions."""

    indices = legal_action_indices(state, perspective=perspective)
    writable = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
    if indices:
        writable[np.asarray(indices, dtype=np.uint16)] = True
    immutable = np.frombuffer(writable.tobytes(), dtype=np.bool_)
    return immutable
