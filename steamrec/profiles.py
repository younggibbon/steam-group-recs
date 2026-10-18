"""Команды шага 2: обучение и просмотр профилей вкуса."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import group as G
from . import model as M

log = logging.getLogger("steamrec")
MODEL_FILE = Path("model") / "taste.npz"


class Data:
    def __init__(self, data_dir: Path):
        proc = data_dir / "processed"
        self.users = pd.read_parquet(proc / "users.parquet")
        self.users["steamid"] = self.users["steamid"].astype(str)
        self.inter = pd.read_parquet(proc / "interactions.parquet")
        self.inter["steamid"] = self.inter["steamid"].astype(str)
        self.games = pd.read_parquet(proc / "games.parquet")
        tags = proc / "game_tags.parquet"
        self.tags = pd.read_parquet(tags) if tags.exists() else pd.DataFrame(columns=["appid", "tag", "votes"])
        names = pd.read_parquet(proc / "library_apps.parquet").set_index("appid")["name"].to_dict()
        names.update(self.games.dropna(subset=["name"]).set_index("appid")["name"].to_dict())
        self.names = names
        self.nick = self.users.set_index("steamid")["personaname"].fillna("?").to_dict()
        self.gi = self.games.set_index("appid")

    def find_user(self, ident: str) -> str:
        ident = str(ident).strip().rstrip("/").split("/")[-1]
        if ident in self.nick:
            return ident
        hit = self.users[self.users["personaname"].fillna("").str.lower() == ident.lower()]
        if len(hit) > 1 and hit["is_seed"].fillna(False).astype(bool).sum() == 1:
            hit = hit[hit["is_seed"].fillna(False).astype(bool)]
        if len(hit) == 1:
            return hit["steamid"].iloc[0]
        hit = self.users[self.users["personaname"].fillna("").str.lower().str.contains(ident.lower(), regex=False)]
        if len(hit) == 1:
            return hit["steamid"].iloc[0]
        raise SystemExit(f"Не нашёл однозначно игрока «{ident}» среди собранных"
                         + (f": {', '.join(hit['personaname'].head(8))}" if len(hit) else ""))

    def find_game(self, ident: str) -> int:
        if str(ident).isdigit():
            return int(ident)
        low = str(ident).lower()
        exact = [a for a, n in self.names.items() if str(n).lower() == low]
        if exact:
            return exact[0]
        part = [a for a, n in self.names.items() if low in str(n).lower()]
        if len(part) == 1:
            return part[0]
        raise SystemExit(f"Не нашёл однозначно игру «{ident}»"
                         + (f": {', '.join(str(self.names[a]) for a in part[:8])}" if part else ""))

    def owned(self, steamid: str) -> set[int]:
        return set(self.inter.loc[self.inter["steamid"] == steamid, "appid"].astype(int))

    def mode(self, appid: int) -> str:
        if appid not in self.gi.index:
            return ""
        r = self.gi.loc[appid]
        if G.flag(r.get("is_coop")) or G.flag(r.get("remote_play_together")):
            return "кооп"
        if G.flag(r.get("is_multiplayer")):
            return "мультиплеер"
        return "одиночная" if G.flag(r.get("is_singleplayer")) else ""

    def seeds(self) -> list[str]:
        return self.users.loc[self.users["is_seed"].fillna(False).astype(bool), "steamid"].tolist()


def load_model(data_dir: Path) -> M.TasteModel:
    path = data_dir / MODEL_FILE
    if not path.exists():
        raise SystemExit("Модели ещё нет: сначала python -m steamrec train")
    return M.TasteModel.load(path)


def cmd_train(data_dir: Path, args) -> None:
    d = Data(data_dir)
    if not args.no_eval:
        e = M.evaluate(d.inter, d.games)
        log.info("Проверка на отложенных сыгранных играх (%d игр у %d игроков), recall@%d / ndcg@%d:",
                 e["held_out"], e["users_tested"], e["k"], e["k"])
        for name, key in (("EASE", "ease"), ("ALS", "als"), ("популярное", "pop")):
            log.info("  %-11s %.3f / %.3f", name, e[f"{key}_recall"], e[f"{key}_ndcg"])
    m = M.train(d.inter, d.games, d.tags)
    m.save(data_dir / MODEL_FILE)
    log.info("Модель: %d игроков, %d игр, %d связей; векторы тегов для %d игр. Файл: %s",
             m.params["n_users"], m.params["n_items"], m.params["n_interactions"],
             len(m.content_items), (data_dir / MODEL_FILE).resolve())


def cmd_recs(data_dir: Path, args) -> None:
    d, m = Data(data_dir), load_model(data_dir)
    sid = d.find_user(args.user)
    owned = d.owned(sid)
    if not owned:
        raise SystemExit(f"У {d.nick.get(sid, sid)} библиотека закрыта или пустая")
    print(f"Рекомендации для {d.nick.get(sid, sid)} (игр в библиотеке: {len(owned)})")
    recs = M.recommend(m, owned, n=args.n * 3)
    shown = 0
    for appid, score in recs:
        mode = d.mode(appid)
        if args.multiplayer and mode not in ("кооп", "мультиплеер"):
            continue
        print(f"  {score:6.3f}  {d.names.get(appid, appid)}  {mode}")
        shown += 1
        if shown >= args.n:
            break


def _top_tags(d: Data, steamid: str, n: int = 6) -> list[str]:
    lib = d.inter[d.inter["steamid"] == steamid]
    w = lib.set_index("appid")["playtime_forever"].pipe(lambda s: M.confidence(s.to_numpy(), 1.0))
    weights = pd.Series(w, index=lib["appid"].to_numpy())
    t = d.tags[d.tags["appid"].isin(weights.index)].copy()
    if t.empty:
        return []
    t["w"] = t["appid"].map(weights) * np.log1p(t["votes"].fillna(0))
    generic = {"multiplayer", "singleplayer", "action", "free to play", "indie", "casual",
               "online co-op", "co-op", "pvp", "first-person", "third person", "great soundtrack",
               "atmospheric", "early access"}
    s = t.groupby("tag")["w"].sum().sort_values(ascending=False)
    return [x for x in s.index if x.lower() not in generic][:n]


def cmd_taste(data_dir: Path, args) -> None:
    d, m = Data(data_dir), load_model(data_dir)
    sids = [d.find_user(u) for u in args.user] if args.user else d.seeds()
    sids = [s for s in sids if d.owned(s)]
    if len(sids) < 2:
        raise SystemExit("Нужно хотя бы двое с открытыми библиотеками")
    names = [d.nick.get(s, s)[:14] for s in sids]
    als, tag = M.taste_similarity(m, d.inter, sids)
    for title, mat in (("Сходство по играм (ALS): играют в одно и то же", als),
                       ("Сходство по тегам: чем вкус отличается от типичного (от -1 до 1)", tag)):
        print(f"\n{title}")
        df = mat.copy()
        df.index, df.columns = names, names
        print(df.round(2).to_string())
    pairs = [(tag.iloc[a, b], names[a], names[b]) for a in range(len(sids)) for b in range(a + 1, len(sids))
             if not np.isnan(tag.iloc[a, b])]
    pairs.sort(reverse=True)
    if pairs:
        print(f"\nСамые похожие по вкусу: {pairs[0][1]} и {pairs[0][2]} ({pairs[0][0]:.2f}); "
              f"самые разные: {pairs[-1][1]} и {pairs[-1][2]} ({pairs[-1][0]:.2f})")
    print("\nЛюбимые теги (с учётом времени игры):")
    for s, n in zip(sids, names):
        print(f"  {n:14} {', '.join(_top_tags(d, s))}")


def cmd_similar(data_dir: Path, args) -> None:
    d, m = Data(data_dir), load_model(data_dir)
    appid = d.find_game(args.game)
    print(f"Похожие на {d.names.get(appid, appid)}")
    for title, content in (("по игрокам (ALS)", False), ("по тегам и жанрам", True)):
        res = M.similar_games(m, appid, n=args.n, content=content)
        print(f"\n  {title}:" + ("" if res else " нет данных"))
        for a, s in res:
            print(f"    {s:5.2f}  {d.names.get(a, a)}  {d.mode(a)}")


def _price(r: G.GroupRec) -> str:
    if r.is_free:
        return "бесплатно"
    return f"${r.price:.2f}" if r.price is not None else "цена неизвестна"


def cmd_group(data_dir: Path, args) -> None:
    d, m = Data(data_dir), load_model(data_dir)
    if args.eval:
        f = pd.read_parquet(data_dir / "processed" / "friendships.parquet")
        res = G.evaluate_groups(d.inter, d.games, f)
        print("Случайные компании из 4 друзей, топ-10, 5 разбиений:")
        print(res.round(3).to_string())
        return
    sids = [d.find_user(u) for u in args.user] if args.user else d.seeds()
    closed = [s for s in sids if not d.owned(s)]
    sids = [s for s in sids if d.owned(s)]
    if len(sids) < 2:
        raise SystemExit("Нужно хотя бы двое с открытыми библиотеками")
    members = {s: d.inter[d.inter["steamid"] == s] for s in sids}
    short = {s: d.nick.get(s, s) for s in sids}
    print("Компания: " + ", ".join(short.values()))
    if closed:
        print("Без библиотеки (закрыт профиль), не учитываются: "
              + ", ".join(d.nick.get(s, s) for s in closed))
    strategies = G.STRATEGIES if args.strategy == "all" else (args.strategy,)
    titles = {"fair": "справедливо: каждый получит своё", "mean": "в среднем лучше всего",
              "misery": "никто не страдает"}
    for st in strategies:
        recs = G.recommend_group(m, d.games, members, d.names, d.nick, strategy=st, n=args.n,
                                 new_only=args.new_only, max_price=args.max_price,
                                 mode="pve" if args.pve else ("coop" if args.coop else "all"),
                                 tags=d.tags, nobody_owns=args.fresh, genres=args.genre,
                                 released_after=pd.Timestamp.today().normalize() - pd.DateOffset(years=1)
                                 if args.fresh else
                                 (pd.Timestamp(year=args.since, month=1, day=1) if args.since else None))
        print(f"\n[{st}] {titles[st]}")
        for i, r in enumerate(recs, 1):
            fit = " ".join(f"{u:.2f}" for u in r.utilities)
            have = "есть у " + ", ".join(short[s] for s in r.owners) if r.owners else "ни у кого нет"
            cost = "" if r.is_free or not r.buyers else \
                (f", докупить {len(r.buyers)} чел. ~${r.price * len(r.buyers):.0f}" if r.price else "")
            print(f"{i:2}. {d.names.get(r.appid, r.appid)}  [{d.mode(r.appid)}, {_price(r)}]")
            print(f"    {have}{cost}")
            print(f"    польза по людям: {fit}" + (f"; потому что {r.because}" if r.because else ""))


def add_parsers(sub) -> None:
    q = sub.add_parser("group", help="шаг 3: во что поиграть компанией")
    q.add_argument("--user", action="append", help="по умолчанию все seed из сбора")
    q.add_argument("-n", type=int, default=10)
    q.add_argument("--strategy", choices=[*G.STRATEGIES, "all"], default="fair")
    q.add_argument("--new-only", action="store_true",
                   help="незнакомое компании: нет хотя бы у половины")
    q.add_argument("--since", type=int, help="вышли не раньше этого года")
    q.add_argument("--genre", action="append", choices=list(G.GENRES), metavar="ЖАНР",
                   help="жанр, можно несколько: " + ", ".join(G.GENRES))
    q.add_argument("--fresh", action="store_true",
                   help="только новые: ни у кого из компании нет и вышли за последний год")
    q.add_argument("--max-price", type=float, help="не дороже, $ (бесплатные проходят всегда)")
    q.add_argument("--coop", action="store_true", help="любой кооп (по флагам магазина)")
    q.add_argument("--pve", action="store_true",
                   help="кооп против игры: без MOBA, королевских битв и соревновательных")
    q.add_argument("--eval", action="store_true", help="сравнить стратегии на случайных компаниях")
    t = sub.add_parser("train", help="обучить профили вкуса (ALS, EASE, векторы тегов)")
    t.add_argument("--no-eval", action="store_true", help="не считать метрики перед обучением")
    r = sub.add_parser("recs", help="рекомендации одному игроку")
    r.add_argument("--user", required=True, help="ник, Steam ID64 или ссылка")
    r.add_argument("-n", type=int, default=15)
    r.add_argument("--multiplayer", action="store_true", help="только кооп и мультиплеер")
    s = sub.add_parser("taste", help="карта вкусов компании: кто на кого похож")
    s.add_argument("--user", action="append", help="по умолчанию все seed из сбора")
    g = sub.add_parser("similar", help="похожие игры")
    g.add_argument("--game", required=True, help="название или appid")
    g.add_argument("-n", type=int, default=10)


COMMANDS = {"group": cmd_group, "train": cmd_train, "recs": cmd_recs, "taste": cmd_taste, "similar": cmd_similar}
