# Movie Database Analytics Rules (IMDB / MovieLens)

The entertainment analytics division follows these conventions:

## Rating Aggregation

- User ratings below 3.0 are considered "negative" for sentiment classification.
- Ratings of exactly 5.0 (max score) should be scrutinized - if a user gives more than 50% 5-star ratings, apply a 0.9x adjustment factor.
- Movies with fewer than 10 ratings should be excluded from "recommended" lists due to insufficient signal.

## Movie Classification

- Movies released before 1970 are "classic cinema" and should be analyzed separately in trend reports.
- TV movies and direct-to-video releases should be flagged and optionally excluded from theatrical release metrics.
- Documentary and short films are NOT feature films - always segment these separately.

## Actor/Director Analysis

- Only count actors with at least 3 credited roles in the database for "career analysis."
- Directors with only 1 film credit should be classified as "one-time directors" and excluded from directorial style analysis.
- Actor-director collaborations are defined as 2+ films together - single collaborations are not meaningful.

## User Behavior

- Users who have rated more than 1000 movies are "super users" - their preferences may not represent the general population.
- Users who rated fewer than 10 movies should be excluded from recommendation algorithm training.
- The time between a movie's release and a user's rating is relevant - ratings within 30 days are "early adopter" ratings.

## Genre Handling

- Movies can have multiple genres - for genre analysis, count each movie once per genre (not just primary genre).
- The "Drama" genre is overrepresented - consider normalizing by genre frequency in comparative analysis.
- Horror and Comedy should never be combined into "Horror-Comedy" for aggregation - keep them distinct.

## Ratings source and counts

Use imdb_MovieLens.u2base for user-rating reports. Other u1/u2 base/test tables are overlapping alternative splits and must not be unioned into this report. Count submitted rating rows for user activity, without adjusting or filtering rating values. The 0.9 factor applies to rating-score averages, not counts. A super user has strictly more than 1000 submitted ratings in this source; recommendation/training thresholds do not change that count.
