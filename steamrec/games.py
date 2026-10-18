"""Метаданные игр: жанры, категории магазина, признаки co-op/multiplayer, теги."""
from __future__ import annotations

import logging
from typing import Iterable

from .api import SteamSpyAPI, StoreAPI

log = logging.getLogger(__name__)

# ID категорий Steam Store (поле categories в appdetails)
CAT_SINGLE = {2}
CAT_MULTI = {1, 9, 20, 24, 27, 36, 37, 38, 39, 44, 47, 48, 49}  # любой мультиплеер
CAT_COOP = {9, 38, 39, 48}  # Co-op, Online Co-op, Shared/Split Screen Co-op, LAN Co-op
CAT_ONLINE_COOP = {38}
CAT_LOCAL_COOP = {39, 24}  # split screen co-op или split screen вообще
CAT_PVP = {36, 37, 47, 49}  # Online/Shared/LAN PvP, PvP
CAT_REMOTE_PLAY = {44}  # Remote Play Together: играть вместе, имея одну копию
CAT_CROSSPLAY = {27}


def parse_store(appid: int, d: dict) -> dict:
    cats = {c["id"] for c in d.get("categories", []) if "id" in c}
    price = d.get("price_overview") or {}
    return {
        "appid": appid,
        "name": d.get("name"),
        "type": d.get("type"),
        "is_free": bool(d.get("is_free")),
        "genres": [g["description"] for g in d.get("genres", [])],
        "categories": [c["description"] for c in d.get("categories", [])],
        "category_ids": sorted(cats),
        "is_singleplayer": bool(cats & CAT_SINGLE),
        "is_multiplayer": bool(cats & CAT_MULTI),
        "is_coop": bool(cats & CAT_COOP),
        "is_online_coop": bool(cats & CAT_ONLINE_COOP),
        "is_local_coop": bool(cats & CAT_LOCAL_COOP),
        "is_pvp": bool(cats & CAT_PVP),
        "remote_play_together": bool(cats & CAT_REMOTE_PLAY),
        "crossplay": bool(cats & CAT_CROSSPLAY),
        "price_usd": price.get("final", 0) / 100 if price else (0.0 if d.get("is_free") else None),
        "metacritic": (d.get("metacritic") or {}).get("score"),
        "release_date": (d.get("release_date") or {}).get("date"),
        "coming_soon": (d.get("release_date") or {}).get("coming_soon"),
        "platforms": [k for k, v in (d.get("platforms") or {}).items() if v],
        "developers": d.get("developers", []),
        "publishers": d.get("publishers", []),
        "short_description": d.get("short_description"),
        "recommendations": (d.get("recommendations") or {}).get("total"),
    }


def parse_steamspy(appid: int, d: dict) -> tuple[dict, list[dict]]:
    tags = d.get("tags") or {}
    if isinstance(tags, list):  # у игр без тегов SteamSpy отдаёт []
        tags = {}
    stats = {
        "appid": appid,
        "owners_range": d.get("owners"),
        "positive": d.get("positive"),
        "negative": d.get("negative"),
        "median_forever": d.get("median_forever"),
        "ccu": d.get("ccu"),
    }
    tag_rows = [{"appid": appid, "tag": t, "votes": int(v)} for t, v in tags.items()]
    return stats, tag_rows


def collect_games(
    appids: Iterable[int],
    store: StoreAPI,
    steamspy: SteamSpyAPI | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Возвращает (games, game_tags, steamspy_stats)."""
    games, tags, spy_stats, missing = [], [], [], 0
    appids = list(appids)
    for i, appid in enumerate(appids, 1):
        d = store.app_details(appid)
        if d is None:
            missing += 1  # удалённые из магазина или региональные игры
        else:
            games.append(parse_store(appid, d))
        if steamspy is not None:
            s = steamspy.app_details(appid)
            if s is not None:
                stats, rows = parse_steamspy(appid, s)
                spy_stats.append(stats)
                tags.extend(rows)
        if i % 50 == 0:
            log.info("Игры: %d/%d (нет в магазине: %d)", i, len(appids), missing)
    log.info("Игры собраны: %d, нет в магазине: %d", len(games), missing)
    return games, tags, spy_stats
