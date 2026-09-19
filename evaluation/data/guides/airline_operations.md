# Airline Operations Metrics (Airline Database)

Aviation analytics must adhere to the following operational standards:

## Flight Status

- Completed flights have actual departure AND arrival times and Cancelled = 0.
- Cancelled flights (L_CANCELLATION codes) are excluded from on-time performance but included in total scheduled flights.
- Diverted flights count as completed for the original route, NOT the diverted destination.

## Delay Classifications

- Our business defines "on-time" as arriving within 15 minutes of scheduled arrival (industry standard).
- Arrival delays of 15 minutes or less are on-time; moderate delays are greater than 15 and at most 180 minutes. Delay-only distributions exclude on-time flights, but averages over completed flights include them.
- Delays over 3 hours are "severe" and must be reported separately with root cause analysis.
- Weather delays (carrier not at fault) should be excluded from carrier performance metrics. A NULL WeatherDelay value means no weather delay occurred (not unknown).

## Route Analysis

- In this monthly extract, a thin route has fewer than 50 completed flights in the reporting month. Origin and destination identify the route. Regional aggregation is for regional summaries; a thin-route inventory counts the qualifying pairs themselves.
- Distance groups (L_DISTANCE_GROUP_250) should be used for fair comparisons, not absolute distance.
- Hub airports (top 30 by traffic) should be analyzed separately from spoke airports.

## Carrier Metrics

- Carrier codes can change due to mergers - maintain a mapping table for continuous carrier history.
- Regional carriers operating under major carrier brands should be attributed to the major carrier for customer-facing metrics.
- New entrant carriers (operating less than 2 years) should be flagged and analyzed separately.

## Time Period Rules

- Q4 (October-December) includes holiday travel surge - weight metrics by normal seasonal patterns.
- January and September are "reset months" - exclude from trend analysis as they show artificial patterns.
- Year-over-year comparisons must account for day-of-week alignment - use ISO weeks when possible.

## Scope and available data

The operational extract covers January 2016. Its descriptive carrier, route, and day-of-week reports use the recorded carrier and original route, including completed diversions. Weather exclusions apply only to carrier performance, not route or day-of-week metrics. NULL WeatherDelay means zero.

Average arrival delay means AVG(ArrDelayMinutes), the nonnegative minutes late, across all completed flights. It is not the signed ArrDelay and does not exclude on-time flights. Counts of diversions include all recorded diversions, even those that did not complete.

Seasonal weighting and reset-month exclusions apply to trend comparisons, not a report of a single month. Merger, brand, hub, and new-entrant adjustments require a supplied mapping or metadata; none is included in this extract. Do not infer these adjustments for its descriptive reports. Root-cause, severity, and distance breakdowns apply when those breakdowns are requested.
