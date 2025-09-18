from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

import logging
logging.getLogger("cmdstanpy").disabled = True

def slug(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

def norm(xs: list[str] | None) -> list[str] | None:
    if xs is None:
        return None
    return [slug(x).lower() for x in xs]

def prep_category(data: pd.DataFrame, category: str) -> pd.DataFrame:
    df = data.copy()
    df["cat_norm"] = df["category"].str.lower()
    target = norm([category])
    df = df[df["cat_norm"].isin(target)]
    df = df.rename(columns = {"week_start": "ds", "weekly_views": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop = True)

def forecast_category(df: pd.DataFrame, category: str, horizon: int = 12, kwargs: dict | None = None) -> pd.DataFrame:
    mk = dict(weekly_seasonality = False, yearly_seasonality = False)
    if kwargs:
        mk.update(kwargs)
    train = prep_category(df, category)
    if train.empty:
        return pd.DataFrame(columns = ["category", "ds", "yhat", "yhat_lower", "yhat_upper"])
    m = Prophet(**mk)
    m.fit(train)
    future = m.make_future_dataframe(periods = horizon, freq = "W")
    forecast = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast.insert(0, "category", category.replace(" ", "_"))
    return forecast

def forecast(data: pd.DataFrame, horizon: int = 12, categories: list[str] | None = None, kwargs: dict | None = None) -> pd.DataFrame:
    df = data.copy()
    df["cat_norm"] = df["category"].str.lower()
    target = norm(categories)
    if target is not None:
        df = df[df["cat_norm"].isin(target)]
    cats_in_df = df["category"].dropna().astype(str).unique().tolist()
    out = []
    for cat in cats_in_df:
        out.append(forecast_category(df, cat, horizon, kwargs))
    return (pd.concat(out, ignore_index = True) if out else pd.DataFrame(columns = ["category", "ds", "yhat", "yhat_lower", "yhat_upper"]))