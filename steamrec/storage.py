"""Преобразование результатов в таблицы и запись в parquet."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .crawl import CrawlResult


def anon_id(steamid: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{steamid}".encode()).hexdigest()[:16]


def crawl_to_frames(res: CrawlResult) -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(list(res.users.values()))
    friendships = pd.DataFrame(res.friendships, columns=["src", "dst", "friend_since"])
    rows = [
        {
            "steamid": sid,
            "appid": g["appid"],
            "playtime_forever": g.get("playtime_forever", 0),  # минуты
            "playtime_2weeks": g.get("playtime_2weeks", 0),
            "rtime_last_played": g.get("rtime_last_played"),
        }
        for sid, games in res.libraries.items()
        for g in games
    ]
    interactions = pd.DataFrame(
        rows,
        columns=["steamid", "appid", "playtime_forever", "playtime_2weeks", "rtime_last_played"],
    )
    names = {
        g["appid"]: g.get("name")
        for games in res.libraries.values()
        for g in games
        if g.get("name")
    }
    library_apps = pd.DataFrame(
        {"appid": list(names), "name": list(names.values())}, columns=["appid", "name"]
    )
    return {
        "users": users,
        "friendships": friendships,
        "interactions": interactions,
        "library_apps": library_apps,
    }


def anonymize(frames: dict[str, pd.DataFrame], salt: str) -> dict[str, pd.DataFrame]:
    if not salt:
        raise ValueError("Для --anonymize задайте STEAM_ID_SALT в .env")
    out = dict(frames)
    h = lambda s: s.map(lambda x: anon_id(str(x), salt))  # noqa: E731
    users = frames["users"].copy()
    users["steamid"] = h(users["steamid"])
    out["users"] = users.drop(columns=["personaname"], errors="ignore")
    fr = frames["friendships"].copy()
    fr["src"], fr["dst"] = h(fr["src"]), h(fr["dst"])
    out["friendships"] = fr
    inter = frames["interactions"].copy()
    inter["steamid"] = h(inter["steamid"])
    out["interactions"] = inter
    return out


def write_frames(frames: dict[str, pd.DataFrame], out_dir: Path | str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in frames.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths


def select_apps(
    interactions: pd.DataFrame,
    seed_ids: set[str],
    min_owners: int = 2,
    max_apps: int | None = None,
) -> list[int]:
    """Игры для запроса в магазин: все игры компании плюс популярные в выборке.

    Store API медленный (около 200 запросов за 5 минут), поэтому редкие игры,
    которые есть только у одного человека из выборки, по умолчанию пропускаем.
    """
    owners = interactions.groupby("appid")["steamid"].nunique().sort_values(ascending=False)
    seed_apps = set(interactions.loc[interactions["steamid"].isin(seed_ids), "appid"])
    seed_part = [a for a in owners.index if a in seed_apps]
    rest = [a for a in owners.index if a not in seed_apps and owners[a] >= min_owners]
    apps = seed_part + rest
    if max_apps is not None:
        apps = apps[: max(max_apps, len(seed_part))]
    return [int(a) for a in apps]
