"""Шаг 2: профили вкусов игроков и игр.

Коллаборативная часть: ALS для неявного фидбека (Hu, Koren, Volinsky, 2008).
Владение игрой это предпочтение p=1, время игры задаёт уверенность
c = 1 + alpha * log1p(часы): купленная и не запущенная игра остаётся слабым сигналом.

EASE (Steck, 2019): линейная модель «игра из игр» в закрытой форме. На наших данных
(около 300 библиотек) она точнее ALS, поэтому оценки для рекомендаций берём из неё,
а векторы ALS используем как профили вкуса и для карты сходства игроков.

Контентная часть: TF-IDF по тегам SteamSpy (с весом голосов), жанрам и категориям
магазина, сжатый SVD. Нужна для игр с одним владельцем, которых ALS не выучит,
и для игроков со скрытым временем игры.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Не игры: тестовые серверы, демо, моды, DLC и т.п.
GAME_TYPES = {"game"}
JUNK_NAME_WORDS = ("test server", "public test", "dedicated server", "playtest", " sdk")
# Категории магазина, которые не говорят о вкусе (доступность, звук, трейдинг)
CATEGORY_KEEP = {
    "Single-player", "Multi-player", "Co-op", "Online Co-op", "Shared/Split Screen Co-op",
    "LAN Co-op", "PvP", "Online PvP", "Shared/Split Screen PvP", "LAN PvP", "MMO",
    "Remote Play Together", "Cross-Platform Multiplayer", "VR Supported", "VR Only",
    "Steam Workshop",
}


@dataclass
class TasteModel:
    users: np.ndarray            # steamid по строкам user_factors
    items: np.ndarray            # appid по строкам item_factors
    user_factors: np.ndarray
    item_factors: np.ndarray
    content_items: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    content: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    params: dict = field(default_factory=dict)
    ease: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    item_popularity: np.ndarray = field(default_factory=lambda: np.zeros(0))  # владельцев в обучении

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, users=self.users.astype(str), items=self.items,
            user_factors=self.user_factors, item_factors=self.item_factors,
            content_items=self.content_items, content=self.content,
            params=np.array([repr(self.params)]), ease=self.ease,
            item_popularity=self.item_popularity,
        )

    @classmethod
    def load(cls, path: Path) -> "TasteModel":
        z = np.load(path, allow_pickle=False)
        import ast
        return cls(z["users"], z["items"], z["user_factors"], z["item_factors"],
                   z["content_items"], z["content"], ast.literal_eval(str(z["params"][0])),
                   z["ease"], z["item_popularity"] if "item_popularity" in z else np.zeros(0))


# ---------- подготовка данных ----------

def junk_appids(games: pd.DataFrame) -> set[int]:
    """Не игры: по типу из магазина и по названию."""
    bad = set(games.loc[games["type"].notna() & ~games["type"].isin(GAME_TYPES), "appid"])
    names = games["name"].fillna("").str.lower()
    bad |= set(games.loc[names.apply(lambda n: any(w in n for w in JUNK_NAME_WORDS)), "appid"])
    return {int(a) for a in bad}


def prepare_interactions(
    interactions: pd.DataFrame,
    games: pd.DataFrame | None = None,
    min_user_games: int = 5,
    min_item_users: int = 2,
) -> pd.DataFrame:
    """Убирает не игры, редкие игры и почти пустые библиотеки (итеративно, до устойчивости)."""
    df = interactions[["steamid", "appid", "playtime_forever"]].copy()
    df["steamid"] = df["steamid"].astype(str)
    if games is not None:
        df = df[~df["appid"].isin(junk_appids(games))]
    while True:
        n = len(df)
        owners = df.groupby("appid")["steamid"].transform("size")
        df = df[owners >= min_item_users]
        sizes = df.groupby("steamid")["appid"].transform("size")
        df = df[sizes >= min_user_games]
        if len(df) == n:
            return df.reset_index(drop=True)


def confidence(playtime_minutes: np.ndarray, alpha: float) -> np.ndarray:
    return 1.0 + alpha * np.log1p(np.asarray(playtime_minutes, dtype=float) / 60.0)


def to_matrix(df: pd.DataFrame, alpha: float, users=None, items=None):
    """Матрица уверенностей пользователь x игра (CSR) и индексы."""
    users = np.array(sorted(df["steamid"].unique())) if users is None else users
    items = np.array(sorted(df["appid"].unique())) if items is None else items
    u_idx = pd.Index(users).get_indexer(df["steamid"])
    i_idx = pd.Index(items).get_indexer(df["appid"])
    keep = (u_idx >= 0) & (i_idx >= 0)
    c = confidence(df["playtime_forever"].to_numpy()[keep], alpha)
    m = sp.csr_matrix((c, (u_idx[keep], i_idx[keep])), shape=(len(users), len(items)))
    m.sum_duplicates()
    return m, users, items


# ---------- ALS ----------

def _als_step(C: sp.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
    """Решает X при фиксированном Y: (YtY + Yu^T (Cu - I) Yu + reg I) x = Yu^T Cu 1."""
    k = Y.shape[1]
    YtY = Y.T @ Y
    X = np.zeros((C.shape[0], k))
    eye = reg * np.eye(k)
    for u in range(C.shape[0]):
        lo, hi = C.indptr[u], C.indptr[u + 1]
        if lo == hi:
            continue
        idx, cu = C.indices[lo:hi], C.data[lo:hi]
        Yu = Y[idx]
        A = YtY + (Yu.T * (cu - 1.0)) @ Yu + eye
        b = Yu.T @ cu
        X[u] = np.linalg.solve(A, b)
    return X


def fit_als(C: sp.csr_matrix, factors: int = 32, reg: float = 1.0, iterations: int = 15,
            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (C.shape[0], factors))
    Y = rng.normal(0, 0.01, (C.shape[1], factors))
    Ct = C.T.tocsr()
    for _ in range(iterations):
        X = _als_step(C, Y, reg)
        Y = _als_step(Ct, X, reg)
    return X, Y


def fit_ease(C: sp.csr_matrix, lam: float = 200.0) -> np.ndarray:
    """B = argmin ||X - XB||² + lam||B||², diag(B)=0, по бинарной матрице владения."""
    X = (C > 0).astype(np.float64)
    G = (X.T @ X).toarray()
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)


# ---------- контентные векторы игр ----------

def content_vectors(games: pd.DataFrame, tags: pd.DataFrame, dim: int = 64,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Вектор игры из тегов (вес log1p голосов), жанров и значимых категорий."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize

    rows = []
    if len(tags):
        t = tags.dropna(subset=["tag"])
        rows.append(pd.DataFrame({"appid": t["appid"], "feat": "tag:" + t["tag"].str.lower(),
                                  "w": np.log1p(t["votes"].fillna(0).clip(lower=0)) + 1.0}))
    for col, prefix, keep in (("genres", "genre:", None), ("categories", "cat:", CATEGORY_KEEP)):
        if col not in games:
            continue
        g = games[["appid", col]].explode(col).dropna()
        if keep is not None:
            g = g[g[col].isin(keep)]
        rows.append(pd.DataFrame({"appid": g["appid"], "feat": prefix + g[col].str.lower(),
                                  "w": 3.0}))
    if not rows:
        return np.array([], dtype=np.int64), np.zeros((0, 0))
    f = pd.concat(rows).groupby(["appid", "feat"], as_index=False)["w"].max()
    apps = np.array(sorted(f["appid"].unique()))
    feats = np.array(sorted(f["feat"].unique()))
    m = sp.csr_matrix((f["w"], (pd.Index(apps).get_indexer(f["appid"]),
                                pd.Index(feats).get_indexer(f["feat"]))),
                      shape=(len(apps), len(feats)))
    m = TfidfTransformer().fit_transform(m)
    dim = min(dim, min(m.shape) - 1)
    if dim >= 2:
        m = TruncatedSVD(dim, random_state=seed).fit_transform(m)
    else:
        m = m.toarray()
    return apps, normalize(m)


# ---------- обучение, рекомендации, оценка ----------

# Подобрано по сетке на отложенных сыгранных играх (см. README)
DEFAULTS = dict(factors=64, reg=10.0, alpha=1.0, iterations=15, ease_lambda=200.0,
                min_user_games=5, min_item_users=2)


def train(interactions, games, tags, **params) -> TasteModel:
    p = {**DEFAULTS, **params}
    df = prepare_interactions(interactions, games, p["min_user_games"], p["min_item_users"])
    C, users, items = to_matrix(df, p["alpha"])
    X, Y = fit_als(C, p["factors"], p["reg"], p["iterations"])
    B = fit_ease(C, p["ease_lambda"])
    capps, cvec = content_vectors(games, tags) if games is not None else (np.array([]), np.zeros((0, 0)))
    p.update(n_users=len(users), n_items=len(items), n_interactions=int(C.nnz))
    pop = np.asarray((C > 0).sum(axis=0)).ravel().astype(float)
    return TasteModel(users, items, X, Y, capps, cvec, p, B, pop)


def holdout_split(df: pd.DataFrame, frac: float = 0.2, min_games: int = 10, seed: int = 42):
    """Прячет frac игр у каждого, у кого их не меньше min_games. Игры в тесте остаются в train."""
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("steamid").indices.items():
        if len(idx) < min_games:
            continue
        test_mask[rng.choice(idx, int(round(len(idx) * frac)), replace=False)] = True
    test = df[test_mask]
    train_df = df[~test_mask]
    test = test[test["appid"].isin(set(train_df["appid"]))]
    return train_df, test


def _recall_at_k(scores: np.ndarray, C_train: sp.csr_matrix, test_items: dict[int, set], k: int):
    recalls, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u, truth in test_items.items():
        s = scores[u].copy()
        s[C_train.indices[C_train.indptr[u]:C_train.indptr[u + 1]]] = -np.inf
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        hits = np.isin(top, list(truth))
        recalls.append(hits.sum() / min(len(truth), k))
        ndcgs.append((hits * discounts).sum() / discounts[:min(len(truth), k)].sum())
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def evaluate(interactions, games, k: int = 20, seed: int = 42, min_test_minutes: int = 120,
             **params) -> dict:
    """Recall@k и NDCG@k на отложенных играх, в которые человек реально играл
    (не меньше min_test_minutes): ALS и EASE против «советовать популярное»."""
    p = {**DEFAULTS, **params}
    df = prepare_interactions(interactions, games, p["min_user_games"], p["min_item_users"])
    train_df, test = holdout_split(df, seed=seed)
    test = test[test["playtime_forever"] >= min_test_minutes]
    C, users, items = to_matrix(train_df, p["alpha"])
    u_ix, i_ix = pd.Index(users), pd.Index(items)
    truth: dict[int, set] = {}
    for sid, app in zip(test["steamid"], test["appid"]):
        truth.setdefault(u_ix.get_loc(sid), set()).add(i_ix.get_loc(app))
    X, Y = fit_als(C, p["factors"], p["reg"], p["iterations"], seed=seed)
    als = _recall_at_k(X @ Y.T, C, truth, k)
    ease = _recall_at_k((C > 0).astype(np.float64) @ fit_ease(C, p["ease_lambda"]), C, truth, k)
    pop = np.asarray((C > 0).sum(axis=0)).ravel().astype(float)
    popular = _recall_at_k(np.tile(pop, (C.shape[0], 1)), C, truth, k)
    return {"k": k, "users_tested": len(truth), "held_out": int(len(test)),
            "ease_recall": ease[0], "ease_ndcg": ease[1],
            "als_recall": als[0], "als_ndcg": als[1],
            "pop_recall": popular[0], "pop_ndcg": popular[1]}


def user_scores(model: TasteModel, owned: set[int]) -> np.ndarray:
    """Оценки EASE по библиотеке. Работает и для тех, кого не было в обучении."""
    x = np.isin(model.items, list(owned)).astype(np.float32)
    return x @ model.ease


def recommend(model: TasteModel, owned: set[int], n: int = 10) -> list[tuple[int, float]]:
    s = user_scores(model, owned)
    order = [i for i in np.argsort(-s) if int(model.items[i]) not in owned]
    return [(int(model.items[i]), float(s[i])) for i in order[:n]]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def similar_games(model: TasteModel, appid: int, n: int = 10, content: bool = False):
    items, vecs = (model.content_items, model.content) if content else (model.items, model.item_factors)
    ix = np.flatnonzero(items == appid)
    if not len(ix):
        return []
    s = _cos(vecs[ix[0]][None], vecs)[0]
    order = [i for i in np.argsort(-s) if i != ix[0]][:n]
    return [(int(items[i]), float(s[i])) for i in order]


def content_profile(model: TasteModel, interactions: pd.DataFrame, steamid: str) -> np.ndarray | None:
    """Вкус по тегам: среднее векторов игр игрока с весом уверенности. Работает и при скрытых часах."""
    lib = interactions[interactions["steamid"].astype(str) == str(steamid)]
    ix = pd.Index(model.content_items).get_indexer(lib["appid"])
    keep = ix >= 0
    if not keep.any():
        return None
    w = confidence(lib["playtime_forever"].to_numpy()[keep], model.params.get("alpha", 2.0))
    v = (model.content[ix[keep]] * w[:, None]).sum(axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


def taste_similarity(model: TasteModel, interactions: pd.DataFrame, steamids: list[str]):
    """Матрицы сходства игроков: по ALS (кто играет в одно и то же) и по тегам (что нравится)."""
    sids = [str(s) for s in steamids]
    u_ix = pd.Index(model.users).get_indexer(sids)
    als = np.full((len(sids), len(sids)), np.nan)
    have = u_ix >= 0
    if have.any():
        F = model.user_factors[u_ix[have]]
        als[np.ix_(have, have)] = _cos(F, F)
    # Почти все любят шутеры, поэтому вычитаем средний профиль по выборке:
    # сходство считается по тому, чем вкус отличается от типичного.
    everyone = [content_profile(model, interactions, s) for s in model.users]
    everyone = [p for p in everyone if p is not None]
    mean = np.mean(everyone, axis=0) if everyone else 0.0
    prof = [content_profile(model, interactions, s) for s in sids]
    tag = np.full((len(sids), len(sids)), np.nan)
    ok = np.array([p is not None for p in prof])
    if ok.any():
        P = np.stack([p for p in prof if p is not None]) - mean
        tag[np.ix_(ok, ok)] = _cos(P, P)
    return (pd.DataFrame(als, index=sids, columns=sids),
            pd.DataFrame(tag, index=sids, columns=sids))
