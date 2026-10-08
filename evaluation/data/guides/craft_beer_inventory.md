# Craft Beer Inventory Guidelines (CraftBeer Database)

The beverage division tracks craft beer with these business rules:

## Beer Classification

- Beers with ABV (alcohol by volume) at or above 9.5% are "high-gravity" and subject to different distribution regulations.
- IBU (International Bitterness Units) above 100 are "extreme" - segment separately for customer preference analysis.
- Beers without IBU data are likely lagers or wheat beers - impute a value of 20 for analysis purposes.

## Brewery Metrics

- Breweries with fewer than 3 distinct beer IDs in catalog are "microbreweries". Regional summaries aggregate them regionally; an inventory identifying microbreweries lists them individually, including zero-beer breweries.
- Breweries producing more than 20 distinct beers are "production breweries" and analyzed separately.
- Brewery location (city/state) is critical for distribution analysis - flag any brewery without valid location data.

## Style Analysis

- For aggregate reports by style, use the exact raw-style-to-category mapping in CraftBeer.style_categories (columns style and category). NULL or unmapped styles use Other. Individual beer listings retain the original style name. The normalized categories are:
  - IPA
  - Stout/Porter
  - Lager/Pilsner
  - Wheat
  - Sour
  - Other
- "Session" versions of beers (ABV < 5%) should be tracked separately from their full-strength counterparts.

## Inventory Rules

- Seasonal beers (pumpkin, winter warmers, etc.) should be flagged for inventory planning.
- Never stock more than 90 days of high-gravity beers due to shelf life concerns.
- Beers without an oz (serving size) value should default to 12 oz for calculations.

## Pricing

- Price-per-ounce is the standard metric for value comparisons, not total price.
- Beers priced more than 2x the category average should be flagged as "premium" tier.
- Calculate "ABV per dollar" as an efficiency metric for value-conscious customers.

## Measurement and report scope

ABV is an exact fractional decimal with three decimal places, representing percentage ABV to one decimal place (0.095 = 9.5%). High-gravity includes the boundary. Session beers have ABV strictly below 0.050. Unknown ABV does not qualify for either threshold.

Average IBU by style uses all beers, replacing NULL IBU with 20 before averaging. Counts of missing IBU inspect the original NULL values before imputation. Session, extreme-bitterness, seasonal, location, and price flags do not add grouping columns or exclude rows in a general catalog/style report; use them for the corresponding requested analysis. Catalog counts use distinct beer IDs.
