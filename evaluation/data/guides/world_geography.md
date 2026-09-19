# World Geography Data Standards (world / Countries Databases)

The international operations division uses these geographic data conventions:

## Country Identification

- Use ISO 3166-1 alpha-3 country codes as the primary identifier, not country names.
- Country names can vary (e.g., "USA" vs "United States") - always join on codes.
- Successor-state analyses require a supplied historical crosswalk. Descriptive reports of this archive retain its recorded codes and names.

## City Data

- "Major cities" are defined as those with population > 2,000,000.
- Capital status can be reported as a flag when requested; it does not override a major-city population threshold.
- City populations change frequently - always note the census/estimate year.

## Language Analysis

- Countries may have multiple official languages - don't assume one-to-one mapping.
- "Percentage of speakers" is for the listed language - percentages across languages may exceed 100% due to multilingualism.
- Indigenous languages with small speaker populations should be preserved in data but may need aggregation for analysis.

## Population Metrics

- Population density = Population / Surface Area (in persons per sq km).
- For this archive, use the population stored in world.Country or world.City. No alternative estimate dates are available.
- Population projections should be clearly labeled as estimates with confidence ranges.

## Economic Indicators

- GNP/GDP figures must specify the year and whether they're nominal or PPP-adjusted.
- Per-capita metrics require matching population figures from the same year.
- Economic data for very small countries may be unreliable - flag countries with < 100,000 population.

## Regional Groupings

- Regions (continents) are the highest aggregation level.
- Sub-regions should align with UN geographic classifications for consistency.
- "Developing" vs "Developed" classifications change over time - specify the classification year/source.

## Data Quality

- Life expectancy and infant mortality rates from conflict zones may be estimates.
- Independence dates should be verified for recently formed nations.
- Flag territories and dependencies separately from sovereign nations.

## Authoritative snapshot

Use world.Country, world.City, and world.CountryLanguage for country, city, population, economic, and language reports. Countries and Mondial are separate datasets and are not substitutes. Join country records on Code/CountryCode, while displaying recorded names as requested. Density requires positive population and positive SurfaceArea.

This is an archived snapshot; source estimate years and nominal/PPP labels were not preserved. Do not invent them, treat the figures as current estimates, or add unavailable metadata to the result. GNP is reported as stored, without per-capita or currency conversion. Official-language flags are stored as 'T' and 'F'. Return flags and metadata only when requested. Population and economic quality flags do not exclude rows unless a question specifies an exclusion.
