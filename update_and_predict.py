"""
Merge the 2023-24 .. 2025-26 seasons into the NBA MVP dataset and re-run the
regression from mlNBA.ipynb.

Usage:  python update_and_predict.py [project_folder]

Expects in project_folder:
    player_mvp_stats.csv   (1995-2023 master dataset from nbaDataClean.ipynb)
    nicknames.csv
    new_data/players_{2024,2025,2026}.csv, teams_new.csv, mvps_new.csv
Writes:
    player_mvp_stats_2026.csv         updated master dataset
    new_data/mvp_predictions.csv      top-10 predicted vs actual for each new season
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

base = sys.argv[1] if len(sys.argv) > 1 else "."
nd = os.path.join(base, "new_data")
NEW_YEARS = [2024, 2025, 2026]

# ---------------------------------------------------------------- build data
old = pd.read_csv(os.path.join(base, "player_mvp_stats.csv"), index_col=0)

players = pd.concat([pd.read_csv(os.path.join(nd, f"players_{y}.csv")) for y in NEW_YEARS], ignore_index=True)
mvps = pd.read_csv(os.path.join(nd, "mvps_new.csv"))[["Player", "Year", "Pts Won", "Pts Max", "Share"]]

teams = pd.read_csv(os.path.join(nd, "teams_new.csv"))
teams["Team"] = teams["Team"].str.replace("*", "", regex=False)
teams["GB"] = pd.to_numeric(teams["GB"].replace("—", "0"))

nick = pd.read_csv(os.path.join(base, "nicknames.csv")).set_index("Abbreviation")["Name"].to_dict()

# every MVP vote-getter must exist in the player table
chk = mvps.merge(players[["Player", "Year"]], how="left", on=["Player", "Year"], indicator=True)
missing = chk[chk["_merge"] == "left_only"]
print("MVP vote-getters not found in player table:", len(missing))
if len(missing):
    print(missing[["Player", "Year", "Share"]].to_string(index=False))

new = players.merge(mvps, how="left", on=["Player", "Year"])
new[["Pts Won", "Pts Max", "Share"]] = new[["Pts Won", "Pts Max", "Share"]].fillna(0)
new["Team"] = new["Tm"].map(nick)
assert new["Team"].notna().all(), new[new["Team"].isna()]["Tm"].unique()
new = new.merge(teams, how="left", on=["Team", "Year"])
assert new["W"].notna().all()
new = new[old.columns]

stats = pd.concat([old, new], ignore_index=True)
# save the master file as-is (blanks stay blank, same as the original CSV) ...
stats.to_csv(os.path.join(base, "player_mvp_stats_2026.csv"))
# ... and fill blanks with 0 only for modeling, like mlNBA.ipynb does
stats = stats.fillna(0)
print(f"master dataset: {stats.shape[0]} rows, seasons {stats.Year.min()}-{stats.Year.max()}")

# ---------------------------------------------------------------- model code
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%',
              'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
              'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']


def find_ap(combo):
    """Average precision of the predicted top players vs the real top 5 (same metric as the notebook)."""
    actual = combo.sort_values("Share", ascending=False).head(5)
    predicted = combo.sort_values("predictions", ascending=False)
    ps, found, seen = [], 0, 1
    for _, row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1
    return sum(ps) / len(ps)


def add_ranks(combo):
    combo = combo.sort_values("Share", ascending=False)
    combo["Rk"] = range(1, combo.shape[0] + 1)
    combo = combo.sort_values("predictions", ascending=False)
    combo["Predicted_Rk"] = range(1, combo.shape[0] + 1)
    return combo


def backtest(stats, model, years, predictors):
    """Train on all seasons before each year, predict that year."""
    aps, all_pred = [], []
    for y in years:
        train = stats[stats["Year"] < y]
        test = stats[stats["Year"] == y]
        model.fit(train[predictors], train["Share"])
        combo = test[["Player", "Share"]].copy()
        combo["predictions"] = model.predict(test[predictors])
        combo["Year"] = y
        combo = add_ranks(combo)
        all_pred.append(combo)
        aps.append(find_ap(combo))
    return aps, pd.concat(all_pred)


# ---------------------------------------------------------------- run
OLD_EVAL = list(range(2000, 2024))   # the range your notebook evaluated (plus 2023)
ALL_EVAL = list(range(2000, 2027))

models = {
    "Ridge(alpha=0.1)  [your notebook's model]": Ridge(alpha=0.1),
    "RandomForest(50 trees)  [fixed comparison]": RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5, n_jobs=-1),
}

out_rows = []
for name, model in models.items():
    aps, preds = backtest(stats, model, ALL_EVAL, predictors)
    ap_by_year = dict(zip(ALL_EVAL, aps))
    print("\n" + "=" * 78)
    print(name)
    print(f"  mean AP 2000-2023: {np.mean([ap_by_year[y] for y in OLD_EVAL]):.3f}   "
          f"mean AP 2024-2026: {np.mean([ap_by_year[y] for y in NEW_YEARS]):.3f}")
    hits = sum(
        preds[preds["Year"] == y].sort_values("predictions", ascending=False).iloc[0]["Player"]
        == preds[preds["Year"] == y].sort_values("Share", ascending=False).iloc[0]["Player"]
        for y in ALL_EVAL
    )
    print(f"  picked the actual MVP in {hits} of {len(ALL_EVAL)} seasons (2000-2026)")
    for y in NEW_YEARS:
        p = preds[preds["Year"] == y].sort_values("predictions", ascending=False).head(5)
        a = preds[preds["Year"] == y].sort_values("Share", ascending=False).head(5)
        print(f"\n  {y} season  (AP = {ap_by_year[y]:.3f})")
        print("   model top 5 (predicted share)         |  actual top 5 (vote share)")
        for i in range(5):
            pr, ac = p.iloc[i], a.iloc[i]
            print(f"   {i+1}. {pr['Player']:<26}{pr['predictions']:6.3f}  |  {ac['Player']:<26}{ac['Share']:6.3f}")
        top10 = preds[preds["Year"] == y].sort_values("predictions", ascending=False).head(10).copy()
        top10["model"] = name.split("[")[0].strip()
        out_rows.append(top10)

pd.concat(out_rows).to_csv(os.path.join(nd, "mvp_predictions.csv"), index=False)
print("\nwrote player_mvp_stats_2026.csv and new_data/mvp_predictions.csv")
