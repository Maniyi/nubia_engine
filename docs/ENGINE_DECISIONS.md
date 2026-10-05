# NUBIA Engine Decisions

## Legal-action availability

Every valid, non-terminal NUBIA position has at least one legal action.

NUBIA has no stalemate or no-legal-move outcome. If the engine produces zero legal actions for a non-terminal state, that represents an engine defect rather than a valid game result.

## Imperion and defended pieces

The Imperion does not defend squares through ordinary capture.

For GBESELE, an opposing piece is defended only when another non-Imperion piece of the same empire could capture an enemy occupying that piece's square using its ordinary capture rules.

Brainwash does not count as defence because it is not a capture.

GBESELE remains a special Imperion action. It does not make the Imperion a defender and must not be included when calculating whether a target is defended.

## Terminal evaluation order

After a legal action is applied, evaluate game-ending conditions in this order:

1. Victory by a Peasant occupying an opposing natural-resource mine, including victory caused immediately by Brainwash.
2. Officer scoring when neither empire has any Peasants remaining.
3. Draw evaluation:
   - third occurrence of the same position;
   - 40 consecutive individual turns without a capture, Brainwash, or re-brainwashing.

If third repetition and the 40th quiet turn occur after the same action, the result is a draw. The engine may retain both draw reasons for diagnostics; neither condition takes precedence over the other for determining the outcome.

Once a terminal result is produced, no further action may be applied.

## Repetition position key

The quiet-ply counter is stored in `GameState` but excluded from the repetition position key.

The repetition key includes:

- the empire that moves next;
- every occupied square;
- each piece's type;
- each piece's current allegiance;
- each piece's original allegiance where it affects re-brainwashing;
- the remaining Brainwash availability of every surviving Mystic;
- any other state that changes the legal actions available from the position.

Move number, action history, notation, quiet-ply count, repetition counts and display-only information are excluded.

Piece IDs should be used internally to preserve Mystic identity, but IDs that have no rules-relevant effect must not cause otherwise identical positions to produce different repetition keys.