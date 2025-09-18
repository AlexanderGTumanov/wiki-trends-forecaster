# Wikipedia Trend Forecaster

This project explores forecasting trends in Wikipedia pageviews. The data is collected directly from the Wikipedia API, saved in parquet format, and aggregated into weekly totals with SQL queries. Two forecasting methods are compared: Prophet, which provides quick and interpretable baseline forecasts, and a simple neural network, which captures finer variations and secondary spikes that Prophet often misses.

The goal is to show how raw web data can be turned into structured time series, and how different forecasting methods perform when applied to real-world patterns of online attention.

The project is organized into two main directories. The `/notebooks` folder contains the `wiki_trends_forecaster.ipynb` notebook. The `/src` folder holds the source code files: `utils.py` for scraping and aggregation, `prop.py` for Prophet analysis, and `nn.py` for neural network functions. When data extraction is run, a third folder `/data` is created to store the extracted datasets in parquet format.

---

## What It Does

- Scrapes Wikipedia for daily pageview data across a chosen set of categories and time ranges.
- Aggregates the raw data into weekly totals using SQL queries, stored as pandas DataFrames.
- Uses Prophet to generate large-scale forecasts of long-term trends.
- Trains a neural network to produce more accurate short-term forecasts.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone <https://github.com/AlexanderGTumanov/wiki-trends-forecaster>
   cd <wiki-trends-forecaster>

---
