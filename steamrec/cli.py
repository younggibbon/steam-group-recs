"""Командная строка.

    python -m steamrec collect --seed <ваш профиль> --seed <друг> ...
    python -m steamrec games            # только догрузить метаданные игр
    python -m steamrec train            # шаг 2: профили вкуса
    python -m steamrec taste            # кто в компании на кого похож
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from .api import SteamSpyAPI, SteamWebAPI, StoreAPI
from .crawl import CrawlConfig, crawl
from .games import collect_games
from . import profiles
from .http import HttpClient, JsonCache, RateLimiter, Transport, requests_transport
from .storage import anonymize, crawl_to_frames, select_apps, write_frames

log = logging.getLogger("steamrec")

# Паузы между запросами (секунды). Web API: около 100k запросов в сутки;
# Store appdetails: около 200 запросов за 5 минут; SteamSpy: 1 запрос в секунду.
INTERVALS = {
    "api.steampowered.com": 0.25,
    "store.steampowered.com": 1.6,
    "steamspy.com": 1.1,
}


def load_key() -> str:
    try:
        from dotenv import find_dotenv, load_dotenv

        # .env ищем от текущей папки, а не от папки пакета
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    return os.environ.get("STEAM_API_KEY", "").strip()


def build_http(cache_path: Path, transport: Transport | None = None, sleep=time.sleep) -> HttpClient:
    limiters = {host: RateLimiter(iv, sleep=sleep) for host, iv in INTERVALS.items()}
    return HttpClient(
        transport or requests_transport(),
        cache=JsonCache(cache_path),
        limiters=limiters,
        sleep=sleep,
    )


def run_games(http: HttpClient, data_dir: Path, seed_ids: set[str], args) -> None:
    proc = data_dir / "processed"
    inter = pd.read_parquet(proc / "interactions.parquet")
    users = pd.read_parquet(proc / "users.parquet")
    if not seed_ids:
        seed_ids = set(users.loc[users["is_seed"], "steamid"])
    appids = select_apps(inter, seed_ids, args.min_owners, args.max_apps)
    log.info("Запрашиваю метаданные %d игр (это самый долгий этап, кеш сохраняется)", len(appids))
    spy = None if args.no_tags else SteamSpyAPI(http)
    games, tags, spy_stats = collect_games(appids, StoreAPI(http), spy)

    games_df = pd.DataFrame(games)
    lib = pd.read_parquet(proc / "library_apps.parquet")
    lost = lib[lib["appid"].isin(set(appids) - set(games_df.get("appid", [])))]
    games_df["in_store"] = True
    games_df = pd.concat([games_df, lost.assign(in_store=False)], ignore_index=True)

    write_frames(
        {
            "games": games_df,
            "game_tags": pd.DataFrame(tags, columns=["appid", "tag", "votes"]),
            "game_stats_steamspy": pd.DataFrame(spy_stats),
        },
        proc,
    )


def main(argv: list[str] | None = None, transport: Transport | None = None, sleep=time.sleep) -> int:
    p = argparse.ArgumentParser(prog="steamrec", description="Сбор данных Steam и профили вкуса")
    p.add_argument("--data", type=Path, default=Path("data"), help="папка с данными")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def game_opts(sp):
        sp.add_argument("--min-owners", type=int, default=2,
                        help="брать игры, которые есть хотя бы у N человек (игры компании берутся всегда)")
        sp.add_argument("--max-apps", type=int, default=None, help="ограничить число игр")
        sp.add_argument("--no-tags", action="store_true", help="не ходить в SteamSpy за тегами")

    c = sub.add_parser("collect", help="обойти друзей и собрать библиотеки, затем игры")
    c.add_argument("--seed", action="append", required=True,
                   help="Steam ID64, ссылка на профиль или vanity имя; можно несколько раз")
    c.add_argument("--depth", type=int, default=2)
    c.add_argument("--max-friends", type=int, default=30, help="друзей на человека начиная с глубины 1")
    c.add_argument("--max-users", type=int, default=500)
    c.add_argument("--skip-games", action="store_true")
    c.add_argument("--anonymize", action="store_true",
                   help="хешировать Steam ID и убрать ники (соль из STEAM_ID_SALT)")
    game_opts(c)

    g = sub.add_parser("games", help="собрать метаданные игр по уже собранным библиотекам")
    game_opts(g)
    profiles.add_parsers(sub)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd in profiles.COMMANDS:
        profiles.COMMANDS[args.cmd](args.data, args)
        return 0

    http = build_http(args.data / "raw" / "cache.sqlite", transport, sleep)
    proc = args.data / "processed"

    if args.cmd == "collect":
        key = load_key()
        if not key:
            print("Нет STEAM_API_KEY: скопируйте .env.example в .env и вставьте ключ", file=sys.stderr)
            return 2
        api = SteamWebAPI(http, key)
        seeds = [api.resolve(s) for s in args.seed]
        cfg = CrawlConfig(max_depth=args.depth, max_friends_per_user=args.max_friends,
                          max_users=args.max_users)
        res = crawl(api, seeds, cfg)
        frames = crawl_to_frames(res)
        write_frames(frames, proc)
        if not args.skip_games:
            run_games(http, args.data, set(seeds), args)
        if args.anonymize:
            frames = {n: pd.read_parquet(proc / f"{n}.parquet")
                      for n in ("users", "friendships", "interactions")}
            write_frames(anonymize(frames, os.environ.get("STEAM_ID_SALT", "")), proc)
    else:
        run_games(http, args.data, set(), args)

    log.info("Сетевых запросов: %d. Данные: %s", http.network_calls, proc.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
