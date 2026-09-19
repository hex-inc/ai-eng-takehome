# Chess Game Analysis Standards (Chess Database)

The competitive gaming division follows these strict conventions when analyzing chess data:

## Opening Exclusions (CRITICAL)

- **Never, ever, ever count any games that play the French Defense.** These games are excluded from ALL metrics without exception.
- **Never, ever, ever count any games that play the Sicilian Defense.** These games are excluded from ALL metrics without exception.
- The above rules apply regardless of whether the opening is played by White or Black.
- The supplied archive identifies exclusions by the joined opening.name containing French or Sicilian (case-insensitive). Transposition exclusions require explicit transposition metadata, which is not present here; do not infer them from unrelated ECO codes.

## Game Classification

- Games with fewer than 10 moves are classified as "miniatures" and should be analyzed separately from full games.
- Games ending in draw by agreement before move 20 are "short draws" and should be flagged for competitive integrity analysis.
- Correspondence games (if present) should never be mixed with over-the-board games in the same analysis.

## Player Ratings

- Players without an established rating should be excluded from rating-based performance analysis.
- Rating differences > 400 points indicate a "mismatch" - these games should be weighted 0.5x in opening success rate calculations.
- Historical ratings (pre-1970) use different scales - apply a normalization factor when comparing across eras.

## Result Handling

- A win rate is wins divided by eligible games; draws count as games but not wins. A separately requested score rate counts wins as 1, draws as 0.5, and losses as 0.
- Games won by forfeit or walkover should be excluded from playing strength analysis.
- Time-control violations (flagging) are valid wins but should be flagged separately from checkmate/resignation wins.

## Opening Analysis

- Opening names should be normalized to ECO (Encyclopedia of Chess Openings) codes for consistency.
- Variations within an opening family should be aggregated to the parent opening for high-level analysis.
- "Irregular" openings (ECO A00) should be tracked but may indicate data quality issues if too frequent.
- Always use the `opening` table (joined via `opening_id`) for opening names and filtering - do not use the denormalized `opening` column in the game table directly.

## Temporal Considerations

- Tournament games should be weighted higher than casual/unrated games when analyzing player strength.
- Games from the same tournament round should be analyzed as a cohort for opening preparation patterns.
- Rapid and blitz games should NEVER be combined with classical time control games in the same metrics.

## Descriptive reports

Game counts and player win rates use unweighted eligible games, after the opening exclusions. Mismatch weighting applies only to opening success-rate analysis, not player win rates or counts. Identify missing ratings as NULL for mismatch analysis. An opening-ID report keeps separate opening IDs even when names match; an ECO report uses game.ECO. Opening-family normalization applies only when a family-level report is requested.

Move-count, draw-agreement, event, time-control, and forfeiture breakdowns need explicit source fields and defined categories. These are not provided for this archive's descriptive game-count and win-rate reports; do not invent exclusions or weights from them.
