NUBIA — board, starting setup, and notation
This specification describes the confirmed board geometry, orientation, starting positions, and notation. 
Board and theme
NUBIA represents two African empires competing to claim the opposing empire's natural-resource mines. The board has 100 squares in a 10 × 10 grid. LAND and SEA squares alternate horizontally and vertically.
Each empire's half has five rows. From the middle toward that empire's back edge: border (b), outskirts (o), town center (c), market (m), palace (p). Columns are called paths, numbered 1–10 from left to right as viewed by the empire seated behind its own palace. The opposing layout is rotated 180°.
Use a half identifier as well as row and path to identify a square uniquely, e.g. A:o2 and B:o2. A/B are accepted half identifiers for this specification. A:b denotes empire A's border row; B:o denotes empire B's outskirts row. Append the path number to identify a square, e.g. A:b4.
Terrain and fixed viewing direction
Each empire's own p1 is SEA. Square types alternate horizontally and vertically across the entire board, including the boundary between the two border rows. From either owner's viewpoint:
Own row
Odd paths (1, 3, 5, 7, 9)
Even paths (2, 4, 6, 8, 10)
Palace (p)
SEA
LAND
Market (m)
LAND
SEA
Town center (c)
SEA
LAND
Outskirts (o)
LAND
SEA
Border (b)
SEA
LAND

For a single fixed image, place empire A at the bottom and empire B at the top. Top-to-bottom rows are B:p, B:m, B:c, B:o, B:b, A:b, A:o, A:c, A:m, A:p. A's paths run 1–10 left-to-right; B's paths run 10–1 left-to-right. Thus A:p1 is the bottom-left square and B:p1 is the top-right square; both are SEA. A square on B's path n appears in fixed-board column 11 − n.
Starting pieces
Piece type
Manual symbol
Starting squares on its own half
Imperion
I
p5, 
Queen
Q
p6, to the Imperion's right from its owner's viewpoint
North-Central War Chief
W
m4, m7
East African High Chief
C
m2, m9
South African Advisor
A
p2, p9
West African Mystic
M
m3, m8
Peasant
P
c1–c10








Resources
Resources occupy the outskirts row at paths 2, 4, 6, 8, 10 on each empire's own half: A:o2, A:o4, A:o6, A:o8, A:o10 and, by the 180° rotation, B:o2, B:o4, B:o6, B:o8, B:o10. In the fixed board view, B's resources therefore appear in columns 1, 3, 5, 7, 9.






Move notation
Symbol
Symbol description
Meaning
-
Dash
Movement
x
Lowercase letter x
Capture
⚡
Lightning bolt
Gbesele
🌀
Cyclone / whirlwind
Brainwash



Board sketch
Fixed view: empire B at top, empire A at bottom. Column headings use A's viewpoint; B's owner-relative paths run 10 to 1 from left to right. □ = LAND; ■ = SEA. These are schematic markers, not prescribed visual colors. R = resource; P = peasant. Pieces belong to the half in which they start.
Half:row
1
2
3
4
5
6
7
8
9
10
B:p
□
■ A
□
■
□ Q
■ I
□
■
□ A
■
B:m
■
□ C
■ M
□ W
■
□
■ W
□ M
■ C
□
B:c
□ P
■ P
□ P
■ P
□ P
■ P
□ P
■ P
□ P
■ P
B:o
■ R
□
■ R
□
■ R
□
■ R
□
■ R
□
B:b
□
■
□
■
□
■
□
■
□
■
A:b
■
□
■
□
■
□
■
□
■
□
A:o
□
■ R
□
■ R
□
■ R
□
■ R
□
■ R
A:c
■ P
□ P
■ P
□ P
■ P
□ P
■ P
□ P
■ P
□ P
A:m
□
■ C
□ M
■ W
□
■
□ W
■ M
□ C
■
A:p
■
□ A
■
□
■ I
□ Q
■
□
■ A
□






Imperion (I)
Starting position: Its throne at A:p5 or B:p5, according to its empire.
Immunity: The Imperion cannot be threatened or captured.
Movement: The Imperion may jump to any other unoccupied square within its own palace row. Intervening pieces do not block this movement.
Capture — GBESELE (⚡): An opposing piece qualifies for capture only when all three conditions hold:
It occupies a square within the Imperion’s own half of the board.
It is undefended. A piece is defended when another piece of its own empire could capture an enemy occupying its square.
It lies on the same row, column, or diagonal as the Imperion, with no intervening piece of either empire.
When GBESELE is performed, all qualifying opposing pieces are captured simultaneously. There is no maximum number of targets, and the player cannot choose a subset.
All targets are assessed using the board position before any captures occur. A piece hidden behind another target cannot be captured during the same GBESELE, even when the blocking target is removed.
The Imperion remains on its current square. Physically, the player lifts it and returns it to the same square; all captured pieces are removed.
GBESELE counts as one move.



Queen (Q)
Starting position: A:p6 or B:p6, immediately to the right of its empire’s starting Imperion, from its owner’s viewpoint.
Movement: The Queen may move any number of squares along a single row, column, or diagonal, in either direction, to an unoccupied square. It cannot jump over pieces of either empire.
Capture: The Queen captures an opposing piece by moving to its square along a single row, column, or diagonal, with no intervening pieces. The captured piece is removed, and the Queen occupies its square. It cannot capture an Imperion.
Terrain: The Queen may freely cross and occupy both LAND and SEA squares.



South African Advisor (A)
Starting positions: A:p2 and A:p9 for Empire A; B:p2 and B:p9 for Empire B.
Movement: The Advisor may move any number of squares along a single row or column, in either direction, to an unoccupied square. It cannot move diagonally or jump over pieces of either empire.
Capture: The Advisor captures an opposing piece by moving to its square along a single row or column, with no intervening pieces. The captured piece is removed, and the Advisor occupies its square. It cannot capture an Imperion.
Terrain: The Advisor may freely cross and occupy both LAND and SEA squares.

East African High Chief (C)
Starting positions: A:m2 and A:m9 for Empire A; B:m2 and B:m9 for Empire B.
Primary movement: The High Chief may move any number of squares along a single diagonal, in either direction, to an unoccupied square. It cannot jump over pieces of either empire.
Secondary movement — switching: The High Chief may move exactly one square horizontally left or right to an unoccupied square within the board. This changes its square type from LAND to SEA or from SEA to LAND. Switching cannot capture and consumes one turn; it cannot be combined with primary movement in that turn.
Capture: The High Chief captures an opposing piece by moving to its square along a single diagonal, with no intervening pieces. The captured piece is removed, and the High Chief occupies its square. It cannot capture an Imperion.
Terrain: The High Chief may occupy both LAND and SEA squares. Diagonal movement preserves its square type; switching changes it.


North-Central War Chief (W)
Starting positions: A:m4 and A:m7 for Empire A; B:m4 and B:m7 for Empire B.
Movement: The War Chief moves in an L-shape: two squares horizontally and one vertically, or two squares vertically and one horizontally, in any direction. The destination must be within the board and unoccupied.
The War Chief may jump over pieces of either empire. Intervening pieces do not block its movement.
Capture: The War Chief captures an opposing piece occupying an L-shaped destination square. The captured piece is removed, and the War Chief occupies its square. Intervening pieces do not block capture. It cannot capture an Imperion.
Terrain: The War Chief may occupy both LAND and SEA squares. Every L-shaped move changes its square type.



West African Mystic (M)
Starting positions: A:m3 and A:m8 for Empire A; B:m3 and B:m8 for Empire B.
Movement: The Mystic may move one or two squares along a single row, column, or diagonal, in either direction, to an unoccupied square. It cannot jump over pieces of either empire. For a two-square move, the intervening square must be unoccupied.
Capture: The Mystic may capture an opposing piece exactly two squares away along a single row, column, or diagonal. The intervening square must be unoccupied. The captured piece is removed, and the Mystic occupies its square.
The Mystic cannot capture an adjacent piece or an Imperion.
Terrain: The Mystic may occupy both LAND and SEA squares.
Special action — BRAINWASH (🌀)
Availability: Each Mystic begins with one Brainwash use, represented by a whirlwind marker. This power may be used only once per Mystic during the game.
Target: An opposing piece exactly two squares away along a single row, column, or diagonal, with the intervening square unoccupied. Any piece except an Imperion may be brainwashed.
Effect: The target changes allegiance to the Mystic’s current empire. It retains its piece type and remains on its square, but faces and fights for its new empire. The Mystic remains on its original square. No piece is captured or removed.
Physically, the player touches the Mystic to the target and returns the Mystic to its original square. The whirlwind marker is transferred from the Mystic to the converted piece, marking the conversion and showing that the Mystic has spent its power.
Orientation: A brainwashed piece adopts its new empire’s orientation. For a Peasant, forward now points toward the palace of the empire opposing its new allegiance, reversing its previous forward movement and diagonal capture directions.
Re-brainwashing restores the piece’s original allegiance and orientation.
Brainwashed Mystics: Conversion does not change a Mystic’s remaining power. An unused Brainwash remains available; a spent Brainwash remains spent.
Remedy — re-brainwashing
A Mystic belonging to a converted piece’s original empire may spend its own unused Brainwash power to restore that piece’s original allegiance and orientation.
The target must be exactly two squares away along a single row, column, or diagonal, with the intervening square unoccupied. Both pieces remain on their current squares.
The restored piece’s conversion marker is removed completely. The restoring Mystic also loses its whirlwind marker because its power has been spent; no conversion marker is placed on the restored piece.
Turn cost: Ordinary movement, capture, Brainwash, or re-brainwashing each counts as one move. These actions cannot be combined in the same move.


Peasant (P)
Starting positions: A:c1–A:c10 for Empire A; B:c1–B:c10 for Empire B. All Peasants follow the same rules.
Forward direction: Forward points toward the palace of the empire opposing the Peasant’s current allegiance. Brainwash changes its allegiance and reverses its forward direction. Re-brainwashing restores its original allegiance and orientation.
Movement: A Peasant may move one square forward or one square horizontally left or right to an unoccupied square. It cannot move backward.
Capture: A Peasant captures an opposing piece one square diagonally forward, to either side. The captured piece is removed, and the Peasant occupies its square.
A Peasant cannot capture horizontally, straight forward, backward, or diagonally backward. It cannot capture an Imperion.
Special movement: A Peasant may move exactly two squares horizontally left or right when it occupies the town-center or outskirts row of the empire it currently belongs to. Both the intervening square and destination must be unoccupied. This move cannot capture and may be used repeatedly while the Peasant is eligible.
Eligibility depends on its current allegiance and current position, including after Brainwash or re-brainwashing. The special move is unavailable on any other row or within the opposing empire’s half. Its ordinary one-square sideways movement remains available.
Terrain: Peasants may occupy both LAND and SEA squares.


Victory, scoring, and draws
Victory by a Peasant
An empire wins immediately when one of its Peasants occupies a natural-resource mine in the opposing empire’s half.
The game ends immediately, even if that Peasant is undefended or could otherwise be captured. The opponent receives no further turn.
Brainwash also triggers victory if the converted Peasant is already standing on a mine belonging to the empire opposing its new allegiance. The game ends immediately; there is no opportunity to re-brainwash it.
All pieces except the Imperion may occupy mine squares, but only an opposing Peasant’s occupancy triggers victory.
Threefold repetition
The game is drawn when the same position occurs for the third time. The occurrences need not be consecutive.
Positions are identical when:
The same empire is next to move.
Every square contains the same piece type with the same current allegiance.
The same Mystics have unused Brainwash power.
Any other state affecting available actions is identical.
Scoring when no Peasants remain
If neither empire has any Peasants remaining, compare the total values of their remaining officers. Each officer counts toward its current empire, including converted officers.
Officer
Value
North-Central War Chief
3
East African High Chief
5
Mystic with spent Brainwash power
3
Mystic with unused Brainwash power
7
South African Advisor
5
Queen
9

The empire with the higher total wins. Equal totals produce a draw.
Imperions are excluded from scoring because both are uncapturable and have infinite value. Peasants are priceless and are never included in scoring.
Draw without captures or Brainwash
The game is drawn after 20 turns per player—40 consecutive individual turns—without a capture, Brainwash, or re-brainwashing.
Any capture, Brainwash, or re-brainwashing resets this counter to zero.

