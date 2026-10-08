# Accident Data Analysis Standards (Accidents Database)

The safety and compliance division uses these rules for accident data:

## Incident Classification

- Only incidents with a completed investigation (all required fields populated) should be included in official statistics.
- Near-misses require an explicit near-miss classification. A recorded collision without injury is a property-damage accident, not a near-miss.
- Multi-vehicle incidents should be counted as ONE incident but with multiple vehicle records linked.

## Severity Scoring

- Fatality incidents always take priority classification regardless of other factors.
- Injury severity should be categorized: Fatal > Severe > Moderate > Minor > Property Damage Only.
- "Unknown" injury severity should be excluded from severity distribution calculations.

## Geographic Analysis

- Administrative units (upravna_enota) are the standard geographic dimension.
- Urban vs. rural classification is determined by the administrative unit, not the specific location.
- Cross-border incidents should be attributed to the unit where the primary impact occurred.

## Person Records

- Each person involved (oseba) should be categorized by role: Driver, Passenger, Pedestrian, Other.
- Driver error vs. external factors should be analyzed separately for prevention programs.
- Vulnerable road users (pedestrians, cyclists) require separate safety metric tracking.

## Temporal Patterns

- Time-of-day analysis should use 4-hour blocks: Night (0-4), Early (4-8), Morning (8-12), Afternoon (12-16), Evening (16-20), Late (20-24).
- Weekend incidents (Saturday 6PM - Monday 6AM) are analyzed separately for impaired driving studies.
- Holiday periods (7 days around major holidays) should be flagged and analyzed separately.

## Reporting Rules

- Comparative accident rates require population or vehicle-kilometer exposure. A descriptive inventory explicitly requesting counts reports raw counts.
- Year-over-year comparisons must use the same reporting methodology.
- Preliminary data (current year) should be clearly labeled as subject to revision.

## Raw incident inventory

Accidents.nesreca contains one row per recorded accident; count these rows for raw accident counts by upravna_enota. Do not join person/vehicle records to count incidents, require injury, or infer investigation completion from unrelated non-NULL fields. The snapshot does not provide a documented completion flag or exposure denominator. Certified official statistics and normalized rates require those additional inputs; a raw inventory does not impose those unavailable filters. Severity, vulnerable-user, and time-window rules apply only to analyses of those dimensions.
