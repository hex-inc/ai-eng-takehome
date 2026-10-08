# Formula 1 Racing Data Guidelines (ErgastF1 Database)

The motorsports analytics division follows these strict conventions:

## Points and Standings

- Pre-2010 points systems are incompatible with modern scoring. When comparing drivers across eras, ONLY use position-based rankings, not points.
- DNF (Did Not Finish) results should still count toward "races entered" but NOT toward "races completed" metrics.
- A podium finish is a position 1, 2, or 3.
- Position values of 0 or NULL are unclassified results. Exclude them from podium, points, top-10, and position-comparison calculations. Position alone does not determine whether a car started or completed the race.

## Lap Time Analysis

- Weather-controlled benchmark comparisons must separate wet and dry races. This archive has no weather flag; a descriptive fastest-recorded-lap leaderboard pools the archive and does not claim weather-controlled comparability.
- Fastest-recorded-lap leaderboards use each driver's official fastest lap in a race, with results.rank between 1 and 10 inclusive. Match lapTimes.lap to results.fastestLap for the same raceId and driverId. Exclude unknown ranks.
- Any lap time under 60 seconds is likely a data error - exclude from all calculations.

## Constructor Performance

- When measuring constructor reliability, only count races where BOTH cars started.
- Separate constructor results before 1980 when an era breakdown is requested. Descriptive lifetime race/start counts include all years; comparable points totals use 2010 onwards.
- Points scored during sprint races (introduced 2021) must be reported in a separate column, never combined with main race points.

## Driver Comparisons

- Only compare teammates who raced at least 10 races together in the same season.
- Drivers with fewer than 20 career starts should be classified as "rookies" regardless of their tenure.

## Starts and completion

Join results.statusId to status.statusId. A result represents a started entry unless status is 'Did not start', 'Did not qualify', 'Did not prequalify', 'Withdrew', or '107% Rule'. Grid 0 does not imply a non-start: pit-lane starters and retirements count. A completed result has status 'Finished' or a lapped-finisher status of the form '+N Lap' or '+N Laps', where N is a positive integer. A classified retirement is not completed. Completion rate divides completed races by starts, not all entries.

Historical races can have multiple car-result rows for the same driver. Driver-level starts, completed races, and podium finishes each count distinct raceId values. A driver starts a race if any entry started, completes it if any started entry completed, and has a podium finish if any result has position 1, 2, or 3. Do not count multiple car rows as multiple races for one driver.

Count distinct drivers for constructor race-start eligibility. For teammate comparisons, first require at least 10 shared starts for that driver pair and constructor in the requested season, then report only races where both completed and both have positive finishing positions. A rookie has at least one but fewer than 20 career starts; exclude non-start entries before counting.

Points are summed from individual race results, not cumulative standings. The archive ends in 2017 and has no sprint races. The comparable-points period begins inclusively in 2010.

Constructor-season points reports include only pairs with at least one eligible individual result (a positive finishing position). A constructorResults summary entry without any eligible individual result does not create a zero-point row in this report.
