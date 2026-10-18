"""Шаг 2 на синтетических данных: два «клана» игроков с разными вкусами."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from steamrec import model as M
from steamrec.cli import main

SHOOTERS = [10, 11, 12, 13, 14]
FARMS = [20, 21, 22, 23, 24]


def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(40):
        pool = SHOOTERS if u < 20 else FARMS
        for a in rng.choice(pool, 4, replace=False):
            rows.append((f"7656{u:04d}", int(a), int(rng.integers(0, 3000))))
        rows.append((f"7656{u:04d}", 99, 5))          # тестовый сервер есть у всех
    inter = pd.DataFrame(rows, columns=["steamid", "appid", "playtime_forever"])
    inter["playtime_2weeks"] = 0
    inter["rtime_last_played"] = 0
    games = pd.DataFrame({
        "appid": SHOOTERS + FARMS + [99],
        "name": [f"Shooter {a}" for a in SHOOTERS] + [f"Farm {a}" for a in FARMS] + ["Shooter Test Server"],
        "type": ["game"] * 11,
        "genres": [["Action"]] * 5 + [["Simulation"]] * 5 + [["Action"]],
        "categories": [["Multi-player", "Online Co-op"]] * 5 + [["Single-player"]] * 5 + [[]],
        "is_coop": [True] * 5 + [False] * 5 + [False],
        "is_multiplayer": [True] * 5 + [False] * 6,
        "is_singleplayer": [False] * 5 + [True] * 6,
        "remote_play_together": [False] * 11,
    })
    tags = pd.DataFrame(
        [(a, t, 100) for a in SHOOTERS for t in ("FPS", "Shooter")]
        + [(a, t, 100) for a in FARMS for t in ("Farming Sim", "Relaxing")]
    , columns=["appid", "tag", "votes"])
    return inter, games, tags


def test_prepare_drops_junk_and_rare():
    inter, games, _ = make_data()
    extra = pd.DataFrame([("76560000", 777, 10)], columns=["steamid", "appid", "playtime_forever"])
    df = M.prepare_interactions(pd.concat([inter, extra]), games, min_user_games=3, min_item_users=2)
    assert 99 not in set(df["appid"])        # test server
    assert 777 not in set(df["appid"])       # одна копия на всю выборку


def test_confidence_keeps_unplayed_as_weak_signal():
    c = M.confidence(np.array([0, 60, 6000]), alpha=1.0)
    assert c[0] == 1.0 and c[0] < c[1] < c[2]


def test_als_and_tags_separate_clans():
    inter, games, tags = make_data()
    m = M.train(inter, games, tags, factors=4, reg=0.1, min_user_games=3)
    als, tag = M.taste_similarity(m, inter, ["76560000", "76560001", "76560030"])
    assert als.iloc[0, 1] > als.iloc[0, 2]
    assert tag.iloc[0, 1] > tag.iloc[0, 2]


def test_recommend_stays_in_taste_and_skips_owned():
    inter, games, tags = make_data()
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    owned = {10, 11, 12}
    recs = [a for a, _ in M.recommend(m, owned, n=2)]
    assert set(recs) <= set(SHOOTERS) - owned
    near = [a for a, _ in M.similar_games(m, 20, n=3, content=True)]
    assert set(near) <= set(FARMS)


def test_holdout_keeps_test_items_known_and_small_users_untouched():
    inter, games, _ = make_data()
    df = M.prepare_interactions(inter, games, min_user_games=3)
    train, test = M.holdout_split(df, frac=0.25, min_games=4)
    assert set(test["appid"]) <= set(train["appid"])
    assert len(test) and not set(map(tuple, test.values)) & set(map(tuple, train.values))


def test_model_roundtrip(tmp_path):
    inter, games, tags = make_data()
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    m.save(tmp_path / "m.npz")
    m2 = M.TasteModel.load(tmp_path / "m.npz")
    assert np.allclose(m.ease, m2.ease) and m2.params["factors"] == 4


def test_cli_train_taste_recs(tmp_path, capsys):
    inter, games, tags = make_data()
    proc = tmp_path / "processed"
    proc.mkdir()
    users = pd.DataFrame({"steamid": sorted(inter["steamid"].unique())})
    users["personaname"] = [f"player{i}" for i in range(len(users))]
    users["is_seed"] = users.index < 3
    users.to_parquet(proc / "users.parquet")
    inter.to_parquet(proc / "interactions.parquet")
    games.to_parquet(proc / "games.parquet")
    tags.to_parquet(proc / "game_tags.parquet")
    games[["appid", "name"]].to_parquet(proc / "library_apps.parquet")
    d = ["--data", str(tmp_path)]
    # маленькая выборка: обучаем с порогом по умолчанию (5 игр) не получится, поэтому без eval
    assert main(d + ["train", "--no-eval"]) == 0
    assert (tmp_path / "model" / "taste.npz").exists()
    assert main(d + ["taste"]) == 0
    assert main(d + ["recs", "--user", "player0", "--multiplayer", "-n", "3"]) == 0
    out = capsys.readouterr().out
    assert "Shooter" in out and "Farm" not in out.split("Рекомендации")[-1]


# ---------- шаг 3: группы ----------

from steamrec import group as G


def test_aggregate_strategies():
    # игра 0 нравится всем средне, игра 1 очень нравится двоим и совсем не нравится третьему
    U = np.array([[0.3, 1.0, 0.0], [0.3, 1.0, 0.1], [0.3, 0.0, 1.0]])
    assert G.aggregate(U, "mean", 1) == [1]
    assert G.aggregate(U, "misery", 1) == [0]
    # fair: после игры 1 третий участник недоволен, поэтому вторая игра для него
    assert G.aggregate(U, "fair", 2) == [1, 2]
    assert G.aggregate(U, "mean", 3, allowed=np.array([True, False, True])) == [2, 0]


def test_group_recs_skip_games_everyone_owns_and_explain():
    inter, games, tags = make_data()
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    members = {s: inter[inter["steamid"] == s] for s in ["76560000", "76560001", "76560002"]}
    names = dict(zip(games["appid"], games["name"]))
    recs = G.recommend_group(m, games, members, names, {}, n=3)
    everyone = set.intersection(*(set(l["appid"]) for l in members.values()))
    assert recs and all(r.appid not in everyone for r in recs)
    assert all(r.appid in SHOOTERS for r in recs)          # фермы одиночные и не про этот клан
    for r in recs:
        assert set(r.owners) | set(r.buyers) == set(members)
        assert len(r.utilities) == 3 and all(0 <= u <= 1 for u in r.utilities)


def test_owned_utility_grows_with_playtime_and_is_capped():
    inter, games, tags = make_data()
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    lib = pd.DataFrame({"appid": [10, 11], "playtime_forever": [60, 600000]})
    cand = G.multiplayer_mask(m, games)
    U = G.member_utilities(m, {"x": lib}, cand)[0]
    items = list(m.items[cand])
    assert U[items.index(10)] < U[items.index(11)] == pytest.approx(G.OWNED_MAX)


def test_pve_filter_drops_moba_and_battle_royale():
    tags = pd.DataFrame([(570, "MOBA", 900), (570, "Co-op", 100),
                         (1172470, "Battle Royale", 900),
                         (548430, "PvE", 900), (548430, "PvP", 50),
                         (1, "PvP", 900), (1, "Co-op", 100)], columns=["appid", "tag", "votes"])
    assert G.pvp_core_appids(tags) == {570, 1172470, 1}


def test_release_year_parses_store_strings():
    games = pd.DataFrame({"appid": [1, 2, 3], "release_date": ["Nov 16, 2009", "To be announced", "2023"]})
    y = G.release_year(games)
    assert y[1] == 2009 and np.isnan(y[2]) and y[3] == 2023


def test_fresh_filter_nobody_owns_and_recent():
    inter, games, tags = make_data()
    games = games.assign(release_date=["Jan 1, 2015"] * 4 + ["Mar 1, 2026"] + ["Jan 1, 2015"] * 6)
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    members = {s: inter[inter["steamid"] == s] for s in ["76560000", "76560001"]}
    owned = set().union(*(set(l["appid"]) for l in members.values()))
    recs = G.recommend_group(m, games, members, {}, {}, n=5, nobody_owns=True,
                             released_after=pd.Timestamp("2025-10-18"))
    assert all(r.appid == 14 and not r.owners for r in recs)
    assert (14 in owned) == (not recs)


def test_genre_filter_uses_top_tags_and_store_genres():
    inter, games, tags = make_data()
    hit = G.genre_appids(games, tags, ["Шутер"])
    assert hit == set(SHOOTERS)
    assert G.genre_appids(games, tags, ["Симулятор"]) == set(FARMS)   # жанр магазина Simulation
    m = M.train(inter, games, tags, factors=4, min_user_games=3)
    members = {s: inter[inter["steamid"] == s] for s in ["76560000", "76560001"]}
    assert not G.recommend_group(m, games, members, {}, {}, n=5, genres=["Хоррор"])
