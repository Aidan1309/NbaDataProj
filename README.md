# NBA MVP Prediction

A data pipeline that scrapes NBA player, team, and MVP voting data from [Basketball-Reference](https://www.basketball-reference.com/), cleans and merges it, and trains a regression model to predict MVP vote share. Data now runs through the 2025-26 season (see *Updating through 2026* below).

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
| `player_mvp_stats.csv` | Final merged dataset used for modeling (1995–2023) |
| `player_mvp_stats_2026.csv` | Same dataset extended through the 2025-26 season |
| `new_data/` | Raw 2023-24 – 2025-26 player, team and MVP-voting tables, plus `mvp_predictions.csv` |
| `update_and_predict.py` | Merges `new_data/` into the master dataset, backtests, and predicts MVP for 2024–2026 |

## Updating through 2026

`update_and_predict.py` extends the dataset to 2025-26 using the same cleaning steps as `nbaDataClean.ipynb`, then backtests Ridge and a Random Forest (train on all earlier seasons, predict each season from 2000 on).

```
pip install pandas scikit-learn
python update_and_predict.py .
```

**Notes on the new data**
- Scraped from Basketball-Reference (2024, 2025, 2026 seasons). To keep it small, player rows are limited to 14+ PPG and 30+ games played. Every MVP vote-getter in those seasons is included.
- Traded players are collapsed to one row (total stats, last team), matching the original cleaning.

**Results (model trained only on earlier seasons)**

| Season | Actual MVP | Ridge pick | Random Forest pick |
|---|---|---|---|
| 2023-24 | Nikola Jokić | Giannis Antetokounmpo | Nikola Jokić |
| 2024-25 | Shai Gilgeous-Alexander | Giannis Antetokounmpo | Shai Gilgeous-Alexander |
| 2025-26 | Shai Gilgeous-Alexander | Nikola Jokić | Shai Gilgeous-Alexander |

Over 2000–2026 the Random Forest picks the actual MVP in 14 of 27 seasons vs 9 of 27 for Ridge. On the notebook's top-5 average-precision metric the two are close over 2000–2023 (Ridge 0.74, Random Forest 0.71), so the Random Forest's edge is mostly in picking the winner and in producing realistic vote shares.

**Known issues in `mlNBA.ipynb`**
- `backtest()` calls `reg.predict` instead of `model.predict` and ignores its `years` argument, so the Ridge vs Random Forest comparison in the notebook is not valid. `update_and_predict.py` has the corrected version.
- The stat-ratio and position/team encoding columns are created but never added to `predictors`.
- The newest seasons only contain high-scoring players, so ratio-to-league-average features would need the full player pool.

## Requirements

- Python 3
- `pandas`, `numpy`, `beautifulsoup4`, `requests`, `selenium`, `scikit-learn`
- ChromeDriver (for the Selenium-based scraping step)

## Notes

- The scraping notebook expects local folders (`mvp/`, `player/`, `team/`) to store raw HTML pages before parsing — create these before re-running the scraping steps.
- Re-scraping will hit Basketball-Reference's live site; be mindful of request rate.
