"""Шаг 4: веб страница «во что поиграть компанией».

    streamlit run app.py

Нужны собранные данные (python -m steamrec collect) и модель (python -m steamrec train).
Профили, которых нет в выборке, подгружаются через Steam Web API по ключу из .env.
"""
from __future__ import annotations

import os
from pathlib import Path

import altair as alt
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import streamlit as st

from steamrec import group as G
from steamrec import model as M
from steamrec.api import SteamWebAPI
from steamrec.cli import build_http, load_key
from steamrec.profiles import MODEL_FILE, Data

ROOT = Path(__file__).parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")   # ключ ищем рядом с app.py, откуда бы ни запускали
STRATEGY_LABELS = {
    "fair": "Справедливо: каждый получит своё",
    "mean": "В среднем лучше всего",
    "misery": "Никто не страдает",
}

st.set_page_config(page_title="Во что поиграть компанией", page_icon="🎮", layout="wide")


@st.cache_resource
def load():
    if not (DATA / MODEL_FILE).exists():
        st.error("Модели нет. Сначала: `python -m steamrec train`")
        st.stop()
    return Data(DATA), M.TasteModel.load(DATA / MODEL_FILE)


@st.cache_resource
def steam_api() -> SteamWebAPI | None:
    key = load_key()
    return SteamWebAPI(build_http(DATA / "raw" / "cache.sqlite"), key) if key else None


@st.cache_data(show_spinner=False)
def fetch_profile(ident: str) -> tuple[str, str, pd.DataFrame | None]:
    """steamid, ник и библиотека профиля не из выборки (None, если закрыта)."""
    api = steam_api()
    if api is None:
        raise RuntimeError("Нет STEAM_API_KEY в .env")
    sid = api.resolve(ident)
    summ = api.player_summaries([sid])
    nick = summ[0].get("personaname", sid) if summ else sid
    games = api.owned_games(sid)
    if games is None:
        return sid, nick, None
    lib = pd.DataFrame([{"appid": g["appid"], "playtime_forever": g.get("playtime_forever", 0)}
                        for g in games], columns=["appid", "playtime_forever"])
    return sid, nick, lib


def label(d: Data, sid: str) -> str:
    n = len(d.owned(sid))
    return f"{d.nick.get(sid, sid)} ({n} игр)" if n else f"{d.nick.get(sid, sid)} (закрыт)"


d, m = load()
if os.environ.get("STEAMREC_ANON"):
    # Для скриншотов и демо: ники заменяются на «Игрок N»
    d.nick = {s: f"Игрок {i}" for i, s in enumerate(
        sorted(d.nick, key=lambda s: (s not in d.seeds(), s)), 1)}

st.title("🎮 Во что поиграть компанией")
st.caption("Рекомендации по библиотекам Steam: EASE по ~300 игрокам из графа друзей, "
           "три способа объединить вкусы компании.")

with st.sidebar:
    st.header("Компания")
    public = d.users[d.users["steamid"].isin(set(d.inter["steamid"]))]
    seeds = [s for s in d.seeds() if d.owned(s)]
    options = sorted(public["steamid"], key=lambda s: (s not in seeds, d.nick.get(s, "").lower()))
    chosen = st.multiselect("Из собранных игроков", options, default=seeds,
                            format_func=lambda s: label(d, s))
    extra = st.text_area("Или добавь профили (ссылка, Steam ID64 или ник, по одному в строке)",
                         height=90, placeholder="https://steamcommunity.com/id/...")
    st.header("Настройки")
    strategy = st.radio("Как объединять вкусы", list(STRATEGY_LABELS),
                        format_func=STRATEGY_LABELS.get)
    n = st.slider("Сколько игр показать", 5, 25, 10)
    mode = st.radio("Режим", ["all", "coop", "pve"], format_func={
        "all": "Любой мультиплеер", "coop": "Кооператив",
        "pve": "Кооп против игры (PvE), без MOBA и королевских битв"}.get)
    genres = st.multiselect("Жанры (подойдёт любой из выбранных)", list(G.GENRES))
    fresh = st.checkbox("Только новые: ни у кого нет и вышли за последний год")
    new_only = st.checkbox("Незнакомое компании: нет хотя бы у половины", disabled=fresh)
    by_year = st.checkbox("Вышли не раньше года", disabled=fresh)
    year = st.slider("Год", 2010, 2026, 2023) if by_year and not fresh else None
    if fresh:
        released_after = pd.Timestamp.today().normalize() - pd.DateOffset(years=1)
    elif year:
        released_after = pd.Timestamp(year=year, month=1, day=1)
    else:
        released_after = None
    limit = st.checkbox("Ограничить цену")
    max_price = st.slider("Не дороже, $", 0, 60, 15, disabled=not limit) if limit else None

members: dict[str, pd.DataFrame] = {s: d.inter[d.inter["steamid"] == s] for s in chosen}
nick = dict(d.nick)
problems = []
for line in [x.strip() for x in extra.splitlines() if x.strip()]:
    try:
        with st.spinner(f"Загружаю {line} из Steam"):
            sid, name, lib = fetch_profile(line)
    except Exception as e:  # noqa: BLE001 - показываем пользователю как есть
        problems.append(f"{line}: {e}")
        continue
    nick[sid] = name
    if lib is None or lib.empty:
        problems.append(f"«{name}»: библиотека закрыта, попроси открыть «Игровую информацию»")
    else:
        members[sid] = lib
for p in problems:
    st.warning(p)

if len(members) < 2:
    st.info("Выбери хотя бы двоих с открытыми библиотеками.")
    st.stop()

names = [nick.get(s, s) for s in members]
tab_recs, tab_taste, tab_about = st.tabs(["Во что поиграть", "Карта вкусов", "Как это работает"])

with tab_recs:
    recs = G.recommend_group(m, d.games, members, d.names, nick, strategy=strategy, n=n,
                             new_only=new_only and not fresh, max_price=max_price, mode=mode,
                             tags=d.tags, released_after=released_after, nobody_owns=fresh,
                             genres=genres)
    if not recs:
        st.warning("Под такие фильтры ничего не нашлось.")
    else:
        tag_sets = G.top_tags(d.tags, G.GENRE_TOP_TAGS).groupby("appid")["tag"].apply(set).to_dict()
        game_genres = {a: [name for name, ts in G.GENRES.items() if ts & tg]
                       for a, tg in tag_sets.items()}
        dates = G.release_dates(d.games).dropna()
        years = {a: (dt.strftime("%m.%Y") if fresh else str(dt.year)) for a, dt in dates.items()}
        rows = []
        for r in recs:
            price = "бесплатно" if r.is_free else (f"${r.price:.2f}" if r.price is not None else "?")
            cost = 0.0 if r.is_free or r.price is None else r.price * len(r.buyers)
            row = {
                "Игра": d.names.get(r.appid, str(r.appid)),
                "Режим": d.mode(r.appid),
                "Цена": price,
                "Вышла": years.get(r.appid),
                "Жанры": ", ".join(game_genres.get(r.appid, [])[:3]),
                "Есть у": ", ".join(nick.get(s, s) for s in r.owners) or "ни у кого",
                "Докупить": f"{len(r.buyers)} чел." + (f", ~${cost:.0f}" if cost else ""),
                "Почему": r.because,
                "Магазин": f"https://store.steampowered.com/app/{r.appid}",
            }
            for name, u in zip(names, r.utilities):
                row[name] = u
            rows.append(row)
        df = pd.DataFrame(rows)
        lead = ["Игра", "Режим", "Жанры", "Цена", "Вышла"]
        df = df[lead + names + [c for c in df.columns if c not in lead and c not in names]]
        df.index = np.arange(1, len(df) + 1)
        cfg = {name: st.column_config.ProgressColumn(name, min_value=0, max_value=1, format="%.2f")
               for name in names}
        cfg["Магазин"] = st.column_config.LinkColumn("Магазин", display_text="открыть")
        st.dataframe(df, column_config=cfg, width="stretch")
        st.caption("Полоски: насколько игра подходит каждому (0-1). У тех, у кого игра уже есть, "
                   "это время игры (до 0.3 за 200+ часов), у остальных прогноз модели.")
        top = recs[0]
        st.success(f"Мой выбор: **{d.names.get(top.appid, top.appid)}**"
                   + (f", потому что {top.because}" if top.because else ""))

with tab_taste:
    known = [s for s in members if s in set(m.users)]
    if len(known) < 2:
        st.info("Карта вкусов строится для игроков из обучающей выборки.")
    else:
        als, tag = M.taste_similarity(m, d.inter, known)
        kind = st.radio("Сходство", ["по тегам: что нравится", "по играм: во что играют"],
                        horizontal=True)
        mat = tag if kind.startswith("по тегам") else als
        short = [nick.get(s, s)[:16] for s in known]
        long = mat.set_axis(short, axis=0).set_axis(short, axis=1).reset_index(names="a") \
            .melt(id_vars="a", var_name="b", value_name="sim")
        chart = alt.Chart(long).mark_rect().encode(
            x=alt.X("b:N", title=None, sort=short), y=alt.Y("a:N", title=None, sort=short),
            color=alt.Color("sim:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                            title="сходство"),
            tooltip=["a", "b", alt.Tooltip("sim:Q", format=".2f")],
        ) + alt.Chart(long).mark_text(fontSize=12).encode(
            x=alt.X("b:N", sort=short), y=alt.Y("a:N", sort=short),
            text=alt.Text("sim:Q", format=".2f"))
        st.altair_chart(chart.properties(height=60 * len(short) + 40), width="stretch")
        st.caption("По тегам: насколько вкусы отличаются от типичного одинаково (от -1 до 1). "
                   "По играм: косинус векторов ALS.")

with tab_about:
    st.markdown("""
**Данные.** Граф друзей Steam до друзей друзей: библиотеки, время игры, карточки
магазина и теги SteamSpy. Видны только открытые профили.

**Модель.** EASE (Steck, 2019) по факту владения играми: на ~300 библиотеках точнее ALS.
На отложенных играх находит в топ-20 34% сыгранных игр против 27% у топа популярного.

**Группа.** Для каждого участника считается польза каждой кооп или мультиплеерной игры,
затем объединение:
- *в среднем*: среднее по участникам;
- *никто не страдает*: минимум;
- *справедливо*: каждая следующая игра максимизирует сумму log(1 + накопленная польза),
  поэтому список подтягивает тех, кому первые игры понравились меньше.

Игры, которые уже есть у всех, не предлагаются. Подробности и метрики в README.
""")
