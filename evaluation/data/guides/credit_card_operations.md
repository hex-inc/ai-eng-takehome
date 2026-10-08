# Credit Card Operations Guidelines (Credit Database)

The payments division operates under these strict business rules:

## Charge Classification

- Charges with charge_code 'RF' are refunds - these should be SUBTRACTED from gross charge volume, not counted separately.
- Charges under $5.00 (charge_amt < 5) are classified as "micro-transactions" and should be excluded from average transaction value calculations.
- Any charge exactly equal to $0.01 is a test transaction - ALWAYS exclude from all analytics.

## Member Segmentation

- Members with fewer than 3 charges in a 12-month period are "inactive" and should not be counted in active member metrics.
- "Premium" members have net lifetime charge volume exceeding $10,000. Segment them separately in membership-segmentation reports; do not split an individual-member leaderboard into tiers.
- Members without any charges in the current statement period should still be counted in "total members" but excluded from "transacting members."

## Provider Analysis

- Provider categories should be mapped to our internal taxonomy:
  - Categories 1-10: Essential spending
  - Categories 11-20: Discretionary spending
  - Categories 21+: Other
- Never report provider-level metrics for providers with fewer than 100 total charges - aggregate as "Long tail providers."

## Statement Reconciliation

- Computed net statement amounts equal purchases minus refunds. Stored statement_amt is an archived statement snapshot and may differ from the full recorded ledger; reconciliation reports should report that difference rather than substitute it for the computed amount.
- Statements with negative balances indicate data quality issues - flag but do not exclude from reporting.
- Month-end statements (charge_dt between 28th and 31st) may have timing differences - use statement_no for period assignment, not charge_dt.

## Fraud Rules

- Any charge over $5,000 that is followed by a refund within 24 hours should be flagged for fraud review.
- Multiple charges from the same provider_no within 60 seconds indicate potential duplicate processing.

## Aggregation definitions

Amounts are exact decimal currency. Refund amounts are stored as positive magnitudes and have charge_code = 'RF'; all other codes, including NULL, are purchases. Exclude $0.01 test transactions before calculating any measure.

"Charge count" counts purchases, not refunds. "Net charge volume", "lifetime charges", and "total charges/amount" mean purchases minus refunds. Average transaction value uses purchases of at least $5 only. In a report with both a count and an amount, the count excludes refunds while the net amount subtracts them. Provider categories come from charge.category_no; labels are exactly "Essential spending", "Discretionary spending", and "Other".

Statement reports use statement_no values present in Credit.statement, joined by that identifier. Member-statement reports additionally group by charge.member_no, the ledger's owner. A statement period is not a calendar quarter. General member, spending, and provider analytics retain charges even when their statement_no is absent from the archived statement table. Active-member thresholds apply only when active members are requested. A diagnostic inventory of long-tail providers reports individual provider IDs and their purchase counts, before aggregation into the long-tail group. Include providers with zero purchases in that inventory.
