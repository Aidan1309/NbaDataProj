# NBA MVP Prediction

A data pipeline that scrapes NBA player, team, and MVP voting data from [Basketball-Reference](https://www.basketball-reference.com/), cleans and merges it, and trains a regression model to predict MVP vote share.

## How it works

The project runs in three stages, one notebook per stage:

### 1. `web_scraping.ipynb`
Scrapes Basketball-Reference for the 1995–2023 seasons:
- **MVP voting results** — pulled with `requests` + `BeautifulSoup`
- **Per-game player stats** — pulled with **Selenium**, since the stats table is loaded dynamically
- **Team standings** — pulled with `requests` + `BeautifulSoup`

Each dataset is saved to CSV (`mvps.csv`, `players.csv`, `teams.csv`).

### 2. `nbaDataClean.ipynb`
Cleans and merges the raw data into a single dataset:
- Strips formatting artifacts (e.g. `*` after player/team names)
- Consolidates players traded mid-season into a single row (using their `TOT` totals)
- Maps team abbreviations to full names (`nicknames.csv`) to join player and team stats
- Fixes inconsistent values (e.g. `—` for games-back → `0`)
- Produces the master dataset: `player_mvp_stats.csv`
- Includes some quick exploratory plots (top scorers, stat correlations with MVP vote share)

### 3. `mlNBA.ipynb`
Trains and evaluates a model to predict MVP vote share:
- **Ridge regression** on ~30 predictors (box score stats + team performance metrics like win %, SRS)
- **Backtesting**: trains on past seasons, predicts each following season, and scores predictions with a custom **mean average precision** metric (checks whether the model's predicted top-5 MVP candidates match the real top-5)
- **Feature engineering**: adds stat ratios (a player's stats relative to that season's league average) and categorical encodings for position/team
- Compares Ridge regression against a **Random Forest Regressor**

## Data

| File | Description |
|---|---|
| `mvps.csv` | Raw MVP voting results, 1995–2023 |
| `players.csv` | Raw per-game player stats, 1995–2023 |
| `teams.csv` | Raw team standings, 1995–2023 |
| `nicknames.csv` | Team abbreviation → full name mapping |
| `player_mvp_stats.csv` | Final merged dataset used for modeling |

## Requirements

- Python 3
- `pandas`, `beautifulsoup4`, `requests`, `selenium`, `scikit-learn`
- ChromeDriver (for the Selenium-based scraping step)

## Notes

- The scraping notebook expects local folders (`mvp/`, `player/`, `team/`) to store raw HTML pages before parsing — create these before re-running the scraping steps.
- Re-scraping will hit Basketball-Reference's live site; be mindful of request rate.
