# Hockey Statistics Guidelines

The hockey analytics team adheres to the following conventions when querying the Hockey database:

## Scoring Metrics

- Plus/minus (+/-) statistics from pre-1968 seasons are unreliable and should be excluded from historical comparisons.
- Playoff statistics (Post* columns) must ALWAYS be reported separately from regular season stats - never combine them.
- Goals scored in shootouts (if any) do NOT count toward official goal totals in our reports.
- Average goals per season is calculated as total goals divided by number of distinct seasons played.

## Goalie Evaluation

- Goalies with fewer than 20 games played in a season should be classified as "backups" and excluded from starting goalie rankings.
- Save percentage calculations must exclude empty-net situations - our business defines this as "true save percentage."
- Shutouts achieved through combined goaltending (multiple goalies) should not count toward individual shutout totals.

## Team Performance

- Any team that relocated mid-season should have their stats split by location in that year's reporting.
- WHA (World Hockey Association) statistics from the `lgID = 'WHA'` records are for historical reference only - never include in NHL career totals.
- Teams with fewer than 41 home games in a season had scheduling anomalies - flag these in reports.

## Award Considerations

- When analyzing award winners, exclude any player who won due to a shortened season (fewer than 60 games league-wide).
- Hall of Fame induction year should not be used as a proxy for career quality - many deserving players are excluded or delayed.

## Career scoring totals

For NHL career points and goals, use Hockey.Scoring rows with lgID = 'NHL', sum regular-season Pts or G across all stints, and count distinct year values for seasons. Post* fields are playoff statistics and are excluded. The regular-season G field is the recorded official total; do not infer shootout adjustments from unrelated fields. Career leaderboards need no goalie, award, home-schedule, or plus/minus exclusions.
