"""Шаг 3: групповые рекомендации «во что поиграть компанией».

1. Для каждого участника считаем, насколько ему подходит каждая игра-кандидат
   (кооп или мультиплеер). Для игр, которых у него нет: оценка EASE по его библиотеке,
   делённая на его максимум, чтобы большие библиотеки не перевешивали. Для игр, которые
   уже есть: польза по времени игры, log-шкала, 200+ часов дают OWNED_MAX. OWNED_MAX
   меньше 1, чтобы список не забивался тем, во что компания и так играет.
2. Объединяем пользы участников одной из стратегий:
   - mean: среднее, «в среднем всем нормально»;
   - misery: минимум, «никто не страдает» (least misery);
   - fair: список собирается жадно, каждая следующая игра максимизирует сумму
     log(1 + накопленная польза) по участникам (пропорциональная справедливость):
     если первые игры больше нравятся одним, следующие подтянут остальных.
3. Для каждой игры показываем, у кого она уже есть, сколько стоит докупить
   остальным и почему она подходит (ближайшая игра из библиотеки участника).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model as M

OWNED_MAX = 0.3          # подобрано на случайных компаниях, см. evaluate_groups
OWNED_FULL_HOURS = 200
PLAYED_MINUTES = 120
STRATEGIES = ("fair", "mean", "misery")


def flag(v) -> bool:
    """True для True/np.True_, False для False, None и NaN."""
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and bool(v)


# Теги SteamSpy, при которых игра про соревнование игроков, даже если Steam
# пометил её как Co-op (Dota 2: кооп против ботов, Apex: отряд в королевской битве)
PVP_CORE_TAGS = {"MOBA", "Battle Royale", "Competitive", "e-sports", "Hero Shooter",
                 "Arena Shooter", "Fighting", "2D Fighter", "Tactical Shooter", "Asymmetric",
                 "Vehicular Combat", "Naval Combat", "Extraction Shooter"}
# Асимметричный PvP, который теги не ловят (у выживших кооп, но играют против человека)
PVP_CORE_MANUAL = {381210}   # Dead by Daylight
TOP_TAGS = 10


def pvp_core_appids(tags: pd.DataFrame | None) -> set[int]:
    """Игры, где соревнование игроков в основе: PvP-теги среди первых TOP_TAGS
    (по голосам), или тег PvP высоко без тега PvE."""
    if tags is None or tags.empty:
        return set()
    t = tags.sort_values(["appid", "votes"], ascending=[True, False])
    t = t.assign(rank=t.groupby("appid").cumcount())
    top = t[t["rank"] < TOP_TAGS]
    bad = set(top.loc[top["tag"].isin(PVP_CORE_TAGS), "appid"])
    pvp_high = set(t.loc[(t["tag"] == "PvP") & (t["rank"] < 8), "appid"])
    pve = set(t.loc[t["tag"] == "PvE", "appid"])
    return {int(a) for a in bad | (pvp_high - pve)}


# Жанры для фильтра: русское название -> теги SteamSpy и жанры магазина
GENRES = {
    "Шутер": {"Shooter", "FPS", "Third-Person Shooter", "Looter Shooter"},
    "Выживание": {"Survival", "Open World Survival Craft"},
    "Хоррор": {"Horror", "Survival Horror", "Psychological Horror"},
    "Песочница и строительство": {"Sandbox", "Building", "Base-Building", "Crafting"},
    "Открытый мир": {"Open World"},
    "RPG": {"RPG", "Action RPG", "CRPG"},
    "Стратегия": {"Strategy", "RTS", "Turn-Based Strategy"},
    "Симулятор": {"Simulation", "Automobile Sim"},
    "Приключения": {"Adventure", "Action-Adventure"},
    "Тактика": {"Tactical"},
    "Весёлые и для вечеринки": {"Funny", "Comedy", "Party Game", "Party"},
    "Казуальные": {"Casual", "Family Friendly"},
    "Ограбления и стелс": {"Heist", "Stealth"},
    "Зомби": {"Zombies"},
    "Рогалики": {"Roguelike", "Rogue-lite", "Action Roguelike"},
    "Фэнтези": {"Fantasy"},
    "Научная фантастика": {"Sci-fi", "Space"},
    "Гонки": {"Racing", "Driving"},
    "Спорт": {"Sports"},
    "Головоломки": {"Puzzle"},
    "Платформеры": {"Platformer", "2D Platformer"},
    "Слэшеры": {"Hack and Slash"},
    "Файтинги": {"Fighting", "2D Fighter"},
}


GENRE_TOP_TAGS = 7


def top_tags(tags: pd.DataFrame, n: int) -> pd.DataFrame:
    t = tags.sort_values(["appid", "votes"], ascending=[True, False])
    return t[t.groupby("appid").cumcount() < n]


def genre_appids(games: pd.DataFrame, tags: pd.DataFrame | None, genres: list[str]) -> set[int]:
    """Игры, у которых есть хотя бы один из выбранных жанров: среди первых GENRE_TOP_TAGS
    тегов по голосам (дальше идут случайные теги) или в жанрах магазина."""
    want = set().union(*(GENRES.get(g, {g}) for g in genres))
    hit = set()
    if tags is not None and not tags.empty:
        hit |= set(top_tags(tags, GENRE_TOP_TAGS).loc[lambda t: t["tag"].isin(want), "appid"])
    if "genres" in games:
        gg = games[["appid", "genres"]].explode("genres")
        hit |= set(gg.loc[gg["genres"].isin(want), "appid"])
    return {int(a) for a in hit}


def release_dates(games: pd.DataFrame) -> pd.Series:
    """Дата выхода по строке из магазина («Nov 16, 2009», «2023», «To be announced» -> NaT)."""
    d = pd.to_datetime(games["release_date"], format="mixed", errors="coerce")
    return pd.Series(d.to_numpy(), index=games["appid"].to_numpy())


def release_year(games: pd.DataFrame) -> pd.Series:
    return release_dates(games).dt.year


def multiplayer_mask(model: M.TasteModel, games: pd.DataFrame, mode: str = "all",
                     tags: pd.DataFrame | None = None) -> np.ndarray:
    """Кандидаты: mode="all" кооп или мультиплеер, "coop" любой кооп,
    "pve" кооп против игры, без игр, где главное соревнование игроков."""
    gi = games.set_index("appid")
    junk = M.junk_appids(games)
    pvp = pvp_core_appids(tags) if mode == "pve" else set()
    tagged = set(tags["appid"]) if tags is not None and not tags.empty else set()
    mmo = pve_tag = set()
    if mode == "pve" and tagged:
        mmo = set(tags.loc[tags["tag"] == "Massively Multiplayer", "appid"])
        pve_tag = set(tags.loc[tags["tag"] == "PvE", "appid"])

    def ok(a: int) -> bool:
        if a in junk or a not in gi.index:
            return False
        r = gi.loc[a]
        coop = flag(r.get("is_coop")) or flag(r.get("is_online_coop")) \
            or flag(r.get("remote_play_together"))
        if mode == "all":
            return coop or flag(r.get("is_multiplayer"))
        if mode == "pve":
            # без тегов SteamSpy остаётся только флаг PvP из магазина
            is_pvp = flag(r.get("is_pvp"))
            # онлайн-шутеры и танки: PvP в магазине + MMO-тег без PvE (War Thunder, H&G)
            mmo_pvp = is_pvp and a in mmo and a not in pve_tag
            return coop and a not in pvp and a not in PVP_CORE_MANUAL and not mmo_pvp \
                and (a in tagged or not is_pvp)
        return coop

    return np.array([ok(int(a)) for a in model.items])


def member_utilities(model: M.TasteModel, libraries: dict[str, pd.DataFrame],
                     candidates: np.ndarray) -> np.ndarray:
    """Матрица пользы участник x кандидат в [0, 1]."""
    cand_items = model.items[candidates].astype(int)
    U = np.zeros((len(libraries), candidates.sum()))
    for k, lib in enumerate(libraries.values()):
        s = M.user_scores(model, set(lib["appid"].astype(int)))[candidates]
        u = np.clip(s / (s.max() + 1e-12), 0, 1)
        hours = lib.groupby("appid")["playtime_forever"].sum().reindex(cand_items).to_numpy() / 60
        own = ~np.isnan(hours)
        played = np.minimum(1, np.log1p(np.nan_to_num(hours)) / np.log1p(OWNED_FULL_HOURS)) * OWNED_MAX
        U[k] = np.where(own, played, u)
    return U


def aggregate(U: np.ndarray, strategy: str, n: int, allowed: np.ndarray | None = None) -> list[int]:
    """Индексы n лучших кандидатов по выбранной стратегии."""
    allowed = np.ones(U.shape[1], bool) if allowed is None else allowed
    if strategy == "mean":
        s = U.mean(axis=0)
    elif strategy == "misery":
        s = U.min(axis=0) + 1e-3 * U.mean(axis=0)   # при равном минимуме решает среднее
    elif strategy == "fair":
        picked: list[int] = []
        acc = np.zeros(U.shape[0])
        free = allowed.copy()
        for _ in range(min(n, int(free.sum()))):
            gain = np.log1p(acc[:, None] + U).sum(axis=0)
            gain[~free] = -np.inf
            j = int(np.argmax(gain))
            picked.append(j)
            acc += U[:, j]
            free[j] = False
        return picked
    else:
        raise ValueError(f"Неизвестная стратегия {strategy}")
    s = np.where(allowed, s, -np.inf)
    return [int(j) for j in np.argsort(-s)[:min(n, int(allowed.sum()))]]


@dataclass
class GroupRec:
    appid: int
    utilities: list[float]
    owners: list[str]
    buyers: list[str]
    price: float | None
    is_free: bool
    because: str


def explain(model: M.TasteModel, appid: int, members: dict[str, pd.DataFrame],
            names: dict[int, str], U_col: np.ndarray, nick: dict[str, str]) -> str:
    """Чем игра связана с библиотекой того, кому она подходит больше всех из не владельцев.
    Вклад игры из библиотеки = вес EASE, делённый на корень её популярности, чтобы
    объяснение не сводилось к «потому что все играют в CS2»."""
    j = int(np.flatnonzero(model.items == appid)[0])
    if model.item_popularity.size:
        damp = 1.0 / np.sqrt(np.maximum(model.item_popularity, 1))
    else:
        damp = np.ones(len(model.items))
    best_txt = ""
    order = np.argsort(-U_col)
    for k in order:
        sid = list(members)[k]
        lib = members[sid]
        if appid in set(lib["appid"].astype(int)):
            continue
        played = lib[lib["playtime_forever"] >= PLAYED_MINUTES]   # объясняем тем, во что реально играет
        ix = pd.Index(model.items).get_indexer(played["appid"].astype(int))
        ix = ix[ix >= 0]
        if not len(ix):
            continue
        contrib = model.ease[ix, j] * damp[ix]
        top = ix[np.argmax(contrib)]
        if contrib.max() <= 0:
            continue
        best_txt = f"{nick.get(sid, sid)} играет в {names.get(int(model.items[top]), model.items[top])}"
        break
    return best_txt


def recommend_group(model: M.TasteModel, games: pd.DataFrame, members: dict[str, pd.DataFrame],
                    names: dict[int, str], nick: dict[str, str], strategy: str = "fair", n: int = 10,
                    new_only: bool = False, max_price: float | None = None,
                    mode: str = "all", tags: pd.DataFrame | None = None,
                    released_after: pd.Timestamp | None = None,
                    nobody_owns: bool = False, genres: list[str] | None = None) -> list[GroupRec]:
    """mode: "all", "coop" или "pve" (см. multiplayer_mask); released_after: вышла не раньше
    этой даты; nobody_owns: игры, которых нет ни у кого из компании; genres: ключи GENRES
    (подходит игра с любым из них)."""
    cand = multiplayer_mask(model, games, mode, tags)
    U = member_utilities(model, members, cand)
    items = model.items[cand]
    gi = games.set_index("appid")
    owned_by = {sid: set(lib["appid"].astype(int)) for sid, lib in members.items()}
    n_own = np.array([sum(int(a) in owned_by[s] for s in members) for a in items])
    # То, что есть у всех, компания и так знает: в рекомендации не берём
    allowed = n_own < len(members)
    if nobody_owns:
        allowed &= n_own == 0
    if genres:
        allowed &= np.isin(items.astype(int), list(genre_appids(games, tags, genres)))
    if released_after is not None:
        date = release_dates(games).reindex(items.astype(int))
        allowed &= (date >= pd.Timestamp(released_after)).fillna(False).to_numpy()
    for j, a in enumerate(items):
        a = int(a)
        owners = [s for s in members if a in owned_by[s]]
        free = a in gi.index and flag(gi.loc[a].get("is_free"))
        price = gi.loc[a].get("price_usd") if a in gi.index else None
        if max_price is not None and not free and owners != list(members):
            if price is None or pd.isna(price) or price > max_price:
                allowed[j] = False
    if new_only:
        # «новое» = хотя бы половине компании игру надо открыть для себя
        allowed &= n_own <= len(members) // 2
    out = []
    for j in aggregate(U, strategy, n, allowed):
        a = int(items[j])
        owners = [s for s in members if a in owned_by[s]]
        r = gi.loc[a] if a in gi.index else {}
        price = r.get("price_usd") if len(r) else None
        out.append(GroupRec(
            appid=a, utilities=[float(x) for x in U[:, j]], owners=owners,
            buyers=[s for s in members if s not in owners],
            price=None if price is None or pd.isna(price) else float(price),
            is_free=bool(len(r) and flag(r.get("is_free"))),
            because=explain(model, a, members, names, U[:, j], nick),
        ))
    return out


# ---------- проверка стратегий на случайных компаниях ----------

def evaluate_groups(interactions: pd.DataFrame, games: pd.DataFrame, friendships: pd.DataFrame,
                    size: int = 4, n_groups: int = 300, k: int = 10,
                    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Компания = человек и трое его друзей с библиотеками. У каждого прячем 20% игр и
    смотрим, попадают ли в общий топ-k мультиплеерные игры, в которые он играл 2+ часа.
    Из списка исключены игры, которые есть у всех (как и в настоящих рекомендациях).

    hit_rate: доля участников с хотя бы одним попаданием;
    all_hit: доля компаний, где попадание есть у каждого (никто не забыт);
    popularity: средняя доля выборки, у которой есть рекомендованная игра (меньше = смелее).
    Результат усреднён по нескольким случайным разбиениям: на ~300 библиотеках разброс большой.
    """
    rows = []
    p = M.DEFAULTS
    df = M.prepare_interactions(interactions, games, p["min_user_games"], p["min_item_users"])
    fr = friendships.astype({"src": str, "dst": str})
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_df, test = M.holdout_split(df, seed=seed)
        test = test[test["playtime_forever"] >= PLAYED_MINUTES]
        C, users, items = M.to_matrix(train_df, p["alpha"])
        model = M.TasteModel(users, items, np.zeros((len(users), 1)), np.zeros((len(items), 1)),
                             ease=M.fit_ease(C, p["ease_lambda"]))
        cand = multiplayer_mask(model, games)
        cand_items = items[cand].astype(int)
        test = test[test["appid"].isin(set(cand_items))]
        truth = test.groupby("steamid")["appid"].apply(set).to_dict()
        libs = {s: g for s, g in train_df.groupby("steamid")}
        adj: dict[str, set] = {}
        for a, b in zip(fr["src"], fr["dst"]):
            if a in libs and b in libs:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        centers = sorted(u for u in set(libs) & set(truth) if len(adj.get(u, ())) >= size - 1)
        pop = np.asarray((C > 0).sum(axis=0)).ravel()[cand].astype(float)
        for _ in range(n_groups):
            c = centers[rng.integers(len(centers))]
            g = [c] + list(rng.choice(sorted(adj[c]), size - 1, replace=False))
            members = {s: libs[s] for s in g}
            U = member_utilities(model, members, cand)
            allowed = np.stack([np.isin(cand_items, libs[s]["appid"]) for s in g]).sum(axis=0) < size
            picks = {st: aggregate(U, st, k, allowed) for st in STRATEGIES}
            picks["popular"] = [int(j) for j in np.argsort(-np.where(allowed, pop, -1))[:k]]
            for name, js in picks.items():
                top = set(cand_items[js])
                hits = [bool(top & truth[s]) for s in g if s in truth]
                rows.append({"strategy": name, "hit_rate": np.mean(hits), "all_hit": all(hits),
                             "popularity": pop[js].mean() / len(users)})
    res = pd.DataFrame(rows).groupby("strategy")[["hit_rate", "all_hit", "popularity"]].mean()
    return res.loc[["fair", "mean", "misery", "popular"]]
