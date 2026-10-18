"""Обход графа друзей в ширину: вы и ваша компания, их друзья, друзья друзей."""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field

from .api import PUBLIC_VISIBILITY, SteamWebAPI

log = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    max_depth: int = 2  # 0 только seed, 1 плюс друзья, 2 плюс друзья друзей
    max_friends_per_user: int = 30  # для глубины >= 1; друзья seed берутся все
    max_users: int = 500  # сколько открытых профилей обойти всего
    random_seed: int = 42


@dataclass
class CrawlResult:
    seeds: list[str]
    users: dict[str, dict] = field(default_factory=dict)  # steamid -> профиль + depth
    friendships: list[dict] = field(default_factory=list)
    libraries: dict[str, list[dict]] = field(default_factory=dict)


def _is_public(summary: dict | None) -> bool:
    return bool(summary) and summary.get("communityvisibilitystate") == PUBLIC_VISIBILITY


def crawl(api: SteamWebAPI, seeds: list[str], cfg: CrawlConfig) -> CrawlResult:
    rng = random.Random(cfg.random_seed)
    res = CrawlResult(seeds=list(dict.fromkeys(seeds)))

    def register(steamids: list[str], depth: int) -> list[str]:
        """Записывает новых пользователей, возвращает открытые профили."""
        summaries = {p["steamid"]: p for p in api.player_summaries(steamids)}
        public = []
        for sid in steamids:
            s = summaries.get(sid, {})
            res.users[sid] = {
                "steamid": sid,
                "depth": depth,
                "is_seed": sid in res.seeds,
                "is_public": _is_public(s),
                "personaname": s.get("personaname"),
                "country": s.get("loccountrycode"),
                "time_created": s.get("timecreated"),
                "crawled": False,  # True, если мы запрашивали его библиотеку
            }
            if _is_public(s):
                public.append(sid)
        return public

    queue: deque[tuple[str, int]] = deque()
    public_seeds = register(res.seeds, 0)
    for sid in res.seeds:
        if sid not in public_seeds:
            log.warning("Профиль %s закрыт: библиотека и друзья будут недоступны", sid)
        queue.append((sid, 0))  # seed пробуем всегда, вдруг открыта только библиотека
    scheduled = len(queue)

    while queue:
        sid, depth = queue.popleft()

        res.users[sid]["crawled"] = True
        games = api.owned_games(sid)
        res.users[sid]["library_public"] = games is not None
        if games is not None:
            res.libraries[sid] = games

        if depth >= cfg.max_depth:
            continue
        friends = api.friend_list(sid)
        res.users[sid]["friends_public"] = friends is not None
        if not friends:
            continue
        for f in friends:
            res.friendships.append(
                {"src": sid, "dst": f["steamid"], "friend_since": f.get("friend_since")}
            )

        new = [f["steamid"] for f in friends if f["steamid"] not in res.users]
        if depth >= 1 and len(new) > cfg.max_friends_per_user:
            new = rng.sample(new, cfg.max_friends_per_user)
        budget = cfg.max_users - scheduled
        if budget <= 0 or not new:
            continue
        for fid in register(new, depth + 1)[:budget]:
            queue.append((fid, depth + 1))
            scheduled += 1

        if len(res.libraries) % 25 == 0:
            log.info("Обойдено библиотек: %d, в очереди: %d", len(res.libraries), len(queue))

    log.info(
        "Готово: пользователей %d, открытых библиотек %d, связей %d",
        len(res.users), len(res.libraries), len(res.friendships),
    )
    return res
