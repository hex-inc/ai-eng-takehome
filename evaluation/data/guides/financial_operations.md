# Financial Database Business Rules

When analyzing financial data from the `financial` database, the following rules MUST be observed:

## Loan Classifications

- Loans with status 'A' are considered "performing" loans and should be included in standard metrics.
- Loans with status 'B' are "watch list" loans - exclude them from default rate calculations but include them in total portfolio size.
- Loans with status 'C' or 'D' are non-performing - these should NEVER be counted in profitability metrics.

## Transaction Handling

- All transactions with k_symbol = 'UROK' (interest) or NULL should be excluded from revenue calculations - we only count fee-based income with known categorization.
- When calculating account balances, ignore any transaction dated before 1995-01-01 as these are from a legacy system migration and are unreliable.
- Credit transactions (type = 'PRIJEM') under 1000 units should be classified as "micro-deposits" and excluded from average deposit calculations.

## District Aggregations

- For district-level summaries, first keep district ID 1 (Prague) separate, using its A2 name.
- Next combine districts 70-77 into "Eastern Region", even when an individual district has fewer than 50 accounts.
- Combine remaining districts with fewer than 50 accounts into "Other"; otherwise use A2. Determine account counts from all accounts before joining loans. A diagnostic inventory may list which individual districts fall below 50 accounts. An explicitly requested Prague-versus-other-districts summary uses those two regions.

## Metric definitions

The authoritative source for these account, transaction, loan, and district rules is the financial schema: account, trans, loan, and district. Other banking datasets, including cs, are separate sources and are not substitutes for reports using these definitions.

Loan classification labels are "Performing" (A), "Watch List" (B), and "Non-Performing" (C and D combined). Portfolio counts and amounts include every classification; they are not profitability measures. Default rate is the number of C/D loans divided by the number of A/C/D loans.

The UROK/NULL exclusion applies to fee revenue only, not general transaction volume or deposits. The 1995 cutoff applies to balances and requests explicitly excluding legacy transactions. Deposit counts and total amounts include every PRIJEM transaction, including micro-deposits, interest, unknown categories, and historical records. Only the average deposit amount excludes deposits below 1000. Apply each measure's population separately.
