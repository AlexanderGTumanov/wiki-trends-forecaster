from __future__ import annotations
import matplotlib.pyplot as plt
from urllib.parse import quote
from datetime import datetime
from typing import List, Dict
import pandas as pd
import requests
import duckdb
import time
import os

def slug(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def norm(xs: list[str] | None) -> list[str] | None:
    if xs is None:
        return None
    return [slug(x).lower() for x in xs]

def list_category_pages(
    category: str,
    project: str = "en.wikipedia.org",
    subcat_depth: int = 0,
    user_agent: str = "WikiBarometer-Min/0.1 (contact@example.com)",
    sleep_s: float = 0.1) -> pd.DataFrame:

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    lang = project.split(".")[0]
    base = f"https://{lang}.wikipedia.org/w/api.php"

    def _get(params: Dict) -> Dict:
        r = session.get(base, params = params, timeout = 30)
        r.raise_for_status()
        time.sleep(sleep_s)
        return r.json()

    def _list_pages_for(cat: str) -> List[Dict]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmtype": "page",
            "cmlimit": "500",
            "format": "json",
        }
        out = []
        while True:
            data = _get(params)
            out.extend(data.get("query", {}).get("categorymembers", []))
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)
        return [{"page_id": p["pageid"], "title": p["title"]} for p in out]

    def _list_subcats_for(cat: str) -> List[str]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmtype": "subcat",
            "cmlimit": "500",
            "format": "json",
        }
        subs = []
        while True:
            data = _get(params)
            for c in data.get("query", {}).get("categorymembers", []):
                t = c["title"]
                if t.startswith("Category:"):
                    subs.append(t.split("Category:", 1)[1])
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)
        return subs

    seen = {}
    frontier = [category]
    for depth in range(subcat_depth + 1):
        next_frontier = []
        for cat in frontier:
            for p in _list_pages_for(cat):
                seen[p["page_id"]] = p["title"]
            if depth < subcat_depth:
                next_frontier.extend(_list_subcats_for(cat))
        frontier = next_frontier

    df = pd.DataFrame([{"page_id": pid, "title": title} for pid, title in seen.items()])
    if df.empty:
        print(f"Warning: no pages found for category {category}")
        return pd.DataFrame(columns = ["page_id", "title"])
    return df.sort_values("page_id").reset_index(drop = True)

def fetch_daily_pageviews(
    categories: List[str],
    start: str,
    end: str,
    project: str = "en.wikipedia.org",
    user_agent: str = "WikiBarometer-Min/0.1 (contact@example.com)",
    sleep_s: float = 0.1,
    out_type: str = "df",
    out_dir: str | None = None,
    category_name: str | None = None):

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    s = start.replace("-", "")
    e = end.replace("-", "")

    def _one(title: str):
        article = quote(title.replace(" ", "_"), safe = "_()'")
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{project}/all-access/all-agents/{article}/daily/{s}/{e}"
        )
        r = session.get(url, timeout=30)
        r.raise_for_status()
        time.sleep(sleep_s)
        return r.json().get("items", [])
    
    records = []
    if out_type == "parquet":
        if out_dir is None:
            raise ValueError("out_dir must be given when out_type = 'parquet'")
        cat_slug = slug(category_name) if category_name else "uncategorized"
        base_dir = os.path.join(out_dir, f"category={cat_slug}")
    else:
        base_dir = None
    for t in categories:
        try:
            items = _one(t)
            if not items:
                print(f"Warning: no pageviews found for '{t}'")
                continue
            recs = []
            for it in items:
                d = datetime.strptime(it["timestamp"][:8], "%Y%m%d")
                recs.append({"title": t, "date": d, "views": int(it.get("views", 0)), "year": d.year})
            if out_type == "df":
                records.extend(recs)
            elif out_type == "parquet":
                df = pd.DataFrame.from_records(recs)
                for y, sub in df.groupby("year"):
                    ydir = os.path.join(base_dir, f"year={int(y)}")
                    os.makedirs(ydir, exist_ok=True)
                    fname = slug(t)
                    sub.drop(columns = ["year"]).to_parquet(os.path.join(ydir, f"title={fname}.parquet"), index = False)
        except requests.HTTPError:
            continue
    if out_type == "df":
        if not records:
            return pd.DataFrame(columns = ["title", "date", "views"])
        return pd.DataFrame.from_records(records).sort_values(["title", "date"]).reset_index(drop = True)
    elif out_type == "parquet":
        return None
    else:
        raise ValueError("out_type must be 'df' or 'parquet'")

def to_weekly_df(data, categories = None, start = None, end = None):
    df = data.copy()
    if "category" not in df.columns:
        df["category"] = "uncategorized"
    df["date"] = pd.to_datetime(df["date"])
    if start is not None or end is not None:
        s = pd.to_datetime(start) if start is not None else df["date"].min()
        e = pd.to_datetime(end) if end is not None else df["date"].max()
        df = df[(df["date"] >= s) & (df["date"] <= e)]
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time
    if categories is not None:
        df["cat_norm"] = df["category"].str.lower()
        target = norm(categories)
        if target is not None:
            df = df[df["cat_norm"].isin(target)]
    out = (df.groupby(["category", "week_start"], as_index = False)["views"].sum().rename(columns = {"views": "weekly_views"}).sort_values(["category", "week_start"]).reset_index(drop = True))
    min_d = df["date"].min()
    max_d = df["date"].max()
    first_monday = (min_d + pd.to_timedelta((7 - min_d.weekday()) % 7, "D")).normalize()
    last_full_base = max_d - pd.Timedelta(days=6)
    last_monday = (last_full_base - pd.to_timedelta(last_full_base.weekday(), "D")).normalize()
    out = out[(out["week_start"] >= first_monday) & (out["week_start"] <= last_monday)]
    return out[["category", "week_start", "weekly_views"]]

def to_weekly_parquet(data, categories = None, start = None, end = None):
    conds = []
    target = norm(categories)
    if target:
        quoted = ", ".join("'" + c.replace("'", "''") + "'" for c in target)
        conds.append(f"lower(category) IN ({quoted})")
    if start is not None:
        conds.append(f"date >= TIMESTAMP '{start}'")
    if end is not None:
        conds.append(f"date <= TIMESTAMP '{end}'")
    where_clause = ("WHERE " + " AND ".join(conds)) if conds else ""
    q = f"""
    WITH daily AS (
        SELECT title, CAST(date AS TIMESTAMP) AS date, views, category, year
        FROM read_parquet('{data}', hive_partitioning=1)
    ),
    filtered AS (
        SELECT * FROM daily {where_clause}
    ),
    bounds AS (
        SELECT min(date) AS min_d, max(date) AS max_d FROM filtered
    ),
    limits AS (
        SELECT
            CASE
              WHEN date_trunc('week', min_d) = min_d::DATE
                THEN date_trunc('week', min_d)
              ELSE date_trunc('week', min_d) + INTERVAL 7 DAY
            END AS first_full_monday,
            CASE
              WHEN max_d >= date_trunc('week', max_d) + INTERVAL 6 DAY
                THEN date_trunc('week', max_d)
              ELSE date_trunc('week', max_d) - INTERVAL 7 DAY
            END AS last_full_monday
        FROM bounds
    ),
    weekly AS (
        SELECT category, date_trunc('week', date) AS week_start, SUM(views)::BIGINT AS weekly_views
        FROM filtered
        GROUP BY category, week_start
    )
    SELECT w.category, w.week_start, w.weekly_views
    FROM weekly w, limits lim
    WHERE w.week_start >= lim.first_full_monday
      AND w.week_start <= lim.last_full_monday
    ORDER BY w.category, w.week_start;
    """
    return duckdb.connect().execute(q).fetchdf()

def plot(data, forecast = None, categories = None, train_end: str | None = None):
    df = data.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["cat_norm"] = df["category"].str.lower()
    target = norm(categories)
    if target is not None:
        df = df[df["cat_norm"].isin(target)]
    f = None
    if forecast is not None:
        f = forecast.copy()
        f["ds"] = pd.to_datetime(f["ds"])
        f["cat_norm"] = f["category"].str.lower()
        if target is not None:
            f = f[f["cat_norm"].isin(target)]
    plt.figure(figsize = (10, 5))
    for cat, sub in df.groupby("category"):
        sub = sub.sort_values("week_start")
        label = cat.replace("_", " ")
        plt.plot(sub["week_start"], sub["weekly_views"], label = f"{label} — actual")
    if f is not None:
        for cat, sub in f.groupby("category"):
            sub = sub.sort_values("ds")
            plt.plot(sub["ds"], sub["yhat"], linestyle = "--", label = f"{cat.replace('_',' ')} — forecast")
    if train_end is not None:
        train_end = pd.to_datetime(train_end)
        plt.axvline(train_end - pd.Timedelta(weeks = 1), color = "gray", linestyle = ":", label = "Training end")
    plt.xlabel("Week")
    plt.ylabel("Weekly views")
    plt.title("Weekly views by category")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.show()
