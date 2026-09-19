# HR Data Policies (employee Database)

Human Resources data analysis must comply with the following internal policies:

## Tenure Calculations

- Employee tenure is calculated from hire_date to the current date (or termination date if applicable).
- Employees hired before 1990-01-01 are part of the "legacy workforce" and should be analyzed separately in retention studies.
- Never report individual employee ages - only aggregate statistics by decade (30s, 40s, 50s, etc.).

## Department Rules

- The "d009" department (Customer Service) was split in 2005 - historical headcount comparisons must account for this.
- Employees can appear in multiple departments (dept_emp) - for current headcount and salary reports, use only assignments with to_date = DATE '9999-01-01', then select the greatest from_date per employee. If dates tie, select the smallest dept_no.
- Department managers (dept_manager) should be excluded from non-management headcount metrics.

## Salary Analytics

- Salary data is point-in-time - always specify the effective date range when reporting compensation metrics.
- "Outlier" salaries (more than 3 standard deviations from the department mean) should be flagged but not excluded.
- Salary summaries require at least 5 distinct employees per reported group. Suppress smaller groups when no broader grouping is requested.

## Title Progression

- Title changes within 90 days of hire are "corrections" and should not count as promotions.
- The title "Senior Engineer" can only be compared with equivalent titles from after 1995 due to title inflation.
- Employees with the same title for more than 7 years should be flagged as "tenure risk" in retention reports.

## Gender Reporting

- Gender-based analytics require minimum cell sizes of 10 to be reported.
- For descriptive mean salaries by gender, include a "Difference" column: that gender's mean minus the mean across all current salaried employees, including suppressed groups.
- A pay-equity conclusion requires controls for department, title, and tenure. A descriptive comparison of raw mean salaries is allowed, but is not an adjusted pay-equity estimate.

## Current snapshot and report scope

Current salary records have to_date = DATE '9999-01-01'. Current employees are employees with a current department assignment; use that population for salary summaries. If multiple current salary records exist, use the one with the greatest from_date. Headcount and salary summaries include managers unless non-management is explicitly requested. The 10-person minimum applies only to gender groups; other salary groups use the 5-person minimum.

Historical department splits and title adjustments apply to historical comparisons with a supplied crosswalk, not current snapshots. Retention, promotion, and outlier flags are separate analyses and do not change descriptive headcount or average-salary populations.
