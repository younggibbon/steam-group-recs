"""Обёртки над Steam Web API, Steam Store API и SteamSpy."""
from __future__ import annotations

import re
from typing import Any, Iterable

from .http import HttpClient

WEB_API = "https://api.steampowered.com"
STORE_API = "https://store.steampowered.com/api/appdetails"
STEAMSPY_API = "https://steamspy.com/api.php"

STEAMID64_RE = re.compile(r"^7656\d{13}$")
PROFILE_URL_RE = re.compile(r"steamcommunity\.com/(id|profiles)/([^/?#]+)")

# communityvisibilitystate: 1 закрыт, 3 открыт (2 и др. встречаются редко)
PUBLIC_VISIBILITY = 3


class SteamWebAPI:
    def __init__(self, http: HttpClient, key: str):
        if not key:
            raise ValueError("Не задан STEAM_API_KEY (см. .env.example)")
        self.http = http
        self.key = key

    def _call(self, path: str, **params) -> tuple[int, Any]:
        return self.http.get_json(f"{WEB_API}/{path}", {"key": self.key, **params})

    def resolve_vanity(self, name: str) -> str | None:
        status, data = self._call("ISteamUser/ResolveVanityURL/v1/", vanityurl=name)
        resp = (data or {}).get("response", {})
        if status == 200 and resp.get("success") == 1:
            return resp["steamid"]
        return None

    def resolve(self, ident: str) -> str:
        """Steam ID64, ссылка на профиль или vanity имя -> Steam ID64."""
        ident = ident.strip()
        m = PROFILE_URL_RE.search(ident)
        if m:
            kind, value = m.groups()
            if kind == "profiles":
                return value
            ident = value
        if STEAMID64_RE.match(ident):
            return ident
        steamid = self.resolve_vanity(ident)
        if steamid is None:
            raise ValueError(f"Не удалось найти профиль Steam: {ident!r}")
        return steamid

    def player_summaries(self, steamids: Iterable[str]) -> list[dict]:
        ids = list(dict.fromkeys(steamids))
        players: list[dict] = []
        for i in range(0, len(ids), 100):  # API принимает до 100 ID за раз
            status, data = self._call(
                "ISteamUser/GetPlayerSummaries/v2/", steamids=",".join(ids[i : i + 100])
            )
            if status == 200:
                players.extend((data or {}).get("response", {}).get("players", []))
        return players

    def friend_list(self, steamid: str) -> list[dict] | None:
        """None, если список друзей скрыт."""
        status, data = self._call(
            "ISteamUser/GetFriendList/v1/", steamid=steamid, relationship="friend"
        )
        if status != 200:
            return None
        return (data or {}).get("friendslist", {}).get("friends", [])

    def owned_games(self, steamid: str) -> list[dict] | None:
        """None, если библиотека скрыта настройками приватности."""
        status, data = self._call(
            "IPlayerService/GetOwnedGames/v1/",
            steamid=steamid,
            include_appinfo=1,
            include_played_free_games=1,
        )
        resp = (data or {}).get("response", {})
        # Скрытая библиотека приходит как пустой response без game_count
        if status != 200 or "game_count" not in resp:
            return None
        return resp.get("games", [])


class StoreAPI:
    def __init__(self, http: HttpClient, lang: str = "english", country: str = "us"):
        self.http = http
        self.lang = lang
        self.country = country

    def app_details(self, appid: int) -> dict | None:
        status, data = self.http.get_json(
            STORE_API, {"appids": appid, "l": self.lang, "cc": self.country}
        )
        entry = (data or {}).get(str(appid), {})
        if status != 200 or not entry.get("success"):
            return None
        return entry.get("data")


class SteamSpyAPI:
    """Пользовательские теги (Store API их не отдаёт) и оценки популярности."""

    def __init__(self, http: HttpClient):
        self.http = http

    def app_details(self, appid: int) -> dict | None:
        status, data = self.http.get_json(
            STEAMSPY_API, {"request": "appdetails", "appid": appid}
        )
        if status != 200 or not data or not data.get("appid"):
            return None
        return data
