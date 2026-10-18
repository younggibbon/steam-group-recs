"""Проверка сбора на фейковом Steam: без сети и без настоящего ключа."""
from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import pytest

from steamrec.api import SteamWebAPI
from steamrec.cli import main
from steamrec.crawl import CrawlConfig, crawl
from steamrec.games import parse_store
from steamrec.http import HttpClient, HttpError, JsonCache, TransientError

ME, FRIEND, PRIVATE, FOF, FOF2 = (f"7656119800000000{i}" for i in range(1, 6))

PUBLIC = {ME, FRIEND, FOF, FOF2}
FRIENDS = {ME: [FRIEND, PRIVATE], FRIEND: [ME, FOF, FOF2], FOF: [FRIEND], FOF2: [FRIEND]}
LIBS = {
    ME: [(570, "Dota 2", 5000), (548430, "Deep Rock Galactic", 1200)],
    FRIEND: [(548430, "Deep Rock Galactic", 300), (892970, "Valheim", 900)],
    FOF: [(892970, "Valheim", 50), (570, "Dota 2", 10)],
    FOF2: [(999, "Rare Game", 1)],
}
STORE = {
    570: {"name": "Dota 2", "type": "game", "is_free": True,
          "genres": [{"id": "1", "description": "Action"}],
          "categories": [{"id": 1, "description": "Multi-player"},
                         {"id": 36, "description": "Online PvP"}]},
    548430: {"name": "Deep Rock Galactic", "type": "game", "is_free": False,
             "price_overview": {"final": 2999},
             "genres": [{"id": "1", "description": "Action"}],
             "categories": [{"id": 1, "description": "Multi-player"},
                            {"id": 9, "description": "Co-op"},
                            {"id": 38, "description": "Online Co-op"}]},
    # 892970 нет в магазине (например, региональное ограничение)
}
SPY_TAGS = {548430: {"Co-op": 900, "FPS": 500}, 570: {"MOBA": 1000}}


class FakeSteam:
    def __init__(self, fail_first: int = 0):
        self.calls: list[tuple[str, dict]] = []
        self.fail_first = fail_first

    def __call__(self, url: str, params: dict):
        self.calls.append((url, params))
        if self.fail_first:
            self.fail_first -= 1
            return 429, None
        host, path = urlparse(url).netloc, urlparse(url).path
        if host == "api.steampowered.com":
            assert params.get("key") == "test-key"
        if path.endswith("ResolveVanityURL/v1/"):
            ok = params["vanityurl"] == "me"
            return 200, {"response": {"success": 1, "steamid": ME} if ok else {"success": 42}}
        if path.endswith("GetPlayerSummaries/v2/"):
            players = [
                {"steamid": s, "personaname": f"user{s[-1]}",
                 "communityvisibilitystate": 3 if s in PUBLIC else 1}
                for s in params["steamids"].split(",")
            ]
            return 200, {"response": {"players": players}}
        if path.endswith("GetFriendList/v1/"):
            sid = params["steamid"]
            if sid not in FRIENDS:
                return 401, None
            return 200, {"friendslist": {"friends": [
                {"steamid": f, "relationship": "friend", "friend_since": 1600000000}
                for f in FRIENDS[sid]]}}
        if path.endswith("GetOwnedGames/v1/"):
            sid = params["steamid"]
            if sid not in LIBS:
                return 200, {"response": {}}  # так Steam отвечает про скрытую библиотеку
            games = [{"appid": a, "name": n, "playtime_forever": t} for a, n, t in LIBS[sid]]
            return 200, {"response": {"game_count": len(games), "games": games}}
        if host == "store.steampowered.com":
            appid = int(params["appids"])
            if appid in STORE:
                return 200, {str(appid): {"success": True, "data": STORE[appid]}}
            return 200, {str(appid): {"success": False}}
        if host == "steamspy.com":
            appid = int(params["appid"])
            return 200, {"appid": appid, "owners": "1 .. 2", "tags": SPY_TAGS.get(appid, [])}
        raise AssertionError(f"неожиданный запрос {url}")


def make_http(tmp_path, transport, **kw):
    return HttpClient(transport, cache=JsonCache(tmp_path / "c.sqlite"), sleep=lambda s: None, **kw)


def test_crawl_depth_privacy_and_limits(tmp_path):
    api = SteamWebAPI(make_http(tmp_path, FakeSteam()), "test-key")
    res = crawl(api, [ME], CrawlConfig(max_depth=2, max_friends_per_user=1, max_users=10))

    assert res.users[ME]["depth"] == 0 and res.users[ME]["is_seed"]
    assert res.users[FRIEND]["depth"] == 1
    assert res.users[PRIVATE]["is_public"] is False
    assert res.users[PRIVATE]["crawled"] is False
    # на глубине 1 берём не больше max_friends_per_user новых друзей
    assert sum(res.users.get(s, {}).get("crawled", False) for s in (FOF, FOF2)) == 1
    assert set(res.libraries) >= {ME, FRIEND}
    assert PRIVATE not in res.libraries


def test_max_users_budget(tmp_path):
    api = SteamWebAPI(make_http(tmp_path, FakeSteam()), "test-key")
    res = crawl(api, [ME], CrawlConfig(max_depth=2, max_users=2))
    assert set(res.libraries) == {ME, FRIEND}


def test_resolve_formats(tmp_path):
    api = SteamWebAPI(make_http(tmp_path, FakeSteam()), "test-key")
    assert api.resolve(ME) == ME
    assert api.resolve(f"https://steamcommunity.com/profiles/{ME}/") == ME
    assert api.resolve("https://steamcommunity.com/id/me") == ME
    assert api.resolve("me") == ME
    with pytest.raises(ValueError):
        api.resolve("nobody")


def test_retry_on_429_and_cache_hides_key(tmp_path):
    fake = FakeSteam(fail_first=2)
    http = make_http(tmp_path, fake)
    api = SteamWebAPI(http, "test-key")
    assert api.owned_games(ME) is not None
    assert len(fake.calls) == 3  # два 429, потом успех
    api.owned_games(ME)
    assert len(fake.calls) == 3  # второй раз из кеша
    raw = (tmp_path / "c.sqlite").read_bytes()
    assert b"test-key" not in raw


def test_gives_up_after_retries(tmp_path):
    def always_down(url, params):
        raise TransientError("no network")

    http = make_http(tmp_path, always_down, max_retries=2)
    with pytest.raises(HttpError):
        http.get_json("https://api.steampowered.com/x", {})


def test_coop_flags():
    row = parse_store(548430, STORE[548430])
    assert row["is_coop"] and row["is_online_coop"] and row["is_multiplayer"]
    assert row["price_usd"] == 29.99
    dota = parse_store(570, STORE[570])
    assert dota["is_pvp"] and not dota["is_coop"] and dota["price_usd"] == 0.0


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("STEAM_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)  # чтобы не подхватить чужой .env
    fake = FakeSteam()
    rc = main(["--data", str(tmp_path / "data"), "collect", "--seed", "me", "--min-owners", "2"],
              transport=fake, sleep=lambda s: None)
    assert rc == 0
    proc = tmp_path / "data" / "processed"
    inter = pd.read_parquet(proc / "interactions.parquet")
    games = pd.read_parquet(proc / "games.parquet").set_index("appid")
    tags = pd.read_parquet(proc / "game_tags.parquet")

    assert {"steamid", "appid", "playtime_forever"} <= set(inter.columns)
    # игры компании берутся всегда, Valheim у двоих, Rare Game у одного и не берётся
    assert set(games.index) == {570, 548430, 892970}
    assert bool(games.loc[548430, "is_coop"]) is True
    assert bool(games.loc[892970, "in_store"]) is False and games.loc[892970, "name"] == "Valheim"
    assert set(tags.loc[tags.appid == 548430, "tag"]) == {"Co-op", "FPS"}

    # повторный запуск целиком из кеша
    n = len(fake.calls)
    main(["--data", str(tmp_path / "data"), "collect", "--seed", "me"],
         transport=fake, sleep=lambda s: None)
    assert len(fake.calls) == n


def test_cli_anonymize(tmp_path, monkeypatch):
    monkeypatch.setenv("STEAM_API_KEY", "test-key")
    monkeypatch.setenv("STEAM_ID_SALT", "pepper")
    monkeypatch.chdir(tmp_path)
    main(["--data", "data", "collect", "--seed", ME, "--anonymize", "--no-tags"],
         transport=FakeSteam(), sleep=lambda s: None)
    users = pd.read_parquet(tmp_path / "data/processed/users.parquet")
    assert "personaname" not in users.columns
    assert ME not in set(users.steamid)
    assert users.steamid.str.len().eq(16).all()


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("STEAM_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["--data", "data", "collect", "--seed", ME], transport=FakeSteam()) == 2
