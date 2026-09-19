# Baseball Sabermetrics Guidelines (lahman_2014 Database)

The baseball analytics division adheres to these measurement standards:

## Batting Metrics

- Batting average (BA) = Hits / At Bats. NEVER include walks in the denominator.
- On-base percentage (OBP) includes walks, HBP, and sacrifice flies in the calculation.
- Slugging percentage (SLG) = Total Bases / At Bats. Weight: 1B=1, 2B=2, 3B=3, HR=4.
- Seasonal batting rate statistics require at least 100 at-bats in the full season. Career statistics include all seasons and stints; apply only the career minimum specified in the question.

## Pitching Standards

- ERA (Earned Run Average) = (Earned Runs × 9) / Innings Pitched.
- WHIP = (Walks + Hits) / Innings Pitched.
- Pitchers with fewer than 50 innings in a season are classified as "relievers" for analysis purposes.
- Quality starts (QS) = 6+ innings pitched with 3 or fewer earned runs.

## Fielding Calculations

- Fielding percentage = (Putouts + Assists) / (Putouts + Assists + Errors).
- Position-specific benchmarks vary significantly - always compare within position groups.
- Utility players (multiple positions) should have fielding stats reported per position, not aggregated.

## Historical Adjustments

- Era-adjusted comparisons require an explicitly supplied adjustment model. Descriptive leaderboards for raw batting average, ERA, WHIP, games started, and career hits use recorded counts from all years without era adjustments.
- Steroid era (1994-2004) statistics are reported as-is but should be flagged in comparative analysis.
- Negro League statistics (when available) should be included in career totals for Hall of Fame analysis.

## Award and Recognition

- MVP voting should use first-place votes as the primary metric, not total points.
- All-Star appearances before 1933 (first game) cannot be compared with later years.
- Hall of Fame voting support is votes / ballots for that voting year. Voting-result reports can include unsuccessful nominations; inductee reports require inducted = 'Y'.

## Hall of Fame Table Structure

- The `halloffame` table contains ALL voting/nomination records, not just successful inductees.
- To filter for actual Hall of Fame members, you MUST use `inducted = 'Y'`.
- To filter for players specifically (excluding managers, umpires, executives), use `category = 'Player'`.
- When querying "Hall of Fame players", always apply BOTH filters: `inducted = 'Y' AND category = 'Player'`.

## Team Performance

- Pythagorean wins = Expected wins based on runs scored vs. runs allowed.
- Teams outperforming Pythagorean expectation by more than 5 wins are "lucky" - flag for regression analysis.
- Playoff performance should be weighted separately from regular season for clutch analysis.

## Seasons, stints, and pitching eligibility

Batting and pitching rows can represent separate team stints within a season. Sum counts by playerID and yearID before computing seasonal rates, applying minimums, or ranking seasons. IPouts counts outs; innings pitched = SUM(IPouts) / 3. Calculate ERA and WHIP from summed counts, not by averaging row-level rates or using the stored ERA field.

Non-relievers have at least 50 innings in the full season. Starting-pitcher rankings require both at least 50 innings and at least one game started. An unrestricted WHIP leaderboard includes relievers and requires only positive innings. Career Hall of Fame hits have no at-bat minimum. Induction year is the successful Player-category induction year; BBWAA voting support is unrelated to the first-place-vote convention for MVP awards.
