# Wikipedia Trend Forecaster

This project explores forecasting trends in Wikipedia pageviews. The data is collected directly from the Wikipedia API, saved in parquet format, and aggregated into weekly totals with SQL queries. Two forecasting methods are compared: Prophet, which provides quick and interpretable baseline forecasts, and a simple neural network, which captures finer variations and secondary spikes that Prophet often misses.

The goal is to show how raw web data can be turned into structured time series, and how different forecasting methods perform when applied to real-world patterns of online attention.
