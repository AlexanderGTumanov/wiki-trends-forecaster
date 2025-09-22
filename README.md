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

## Contents of the Notebook

Notebook `/notebooks/wiki_trends_forecaster.ipynb` is divided into three main parts. First, daily pageview data is scraped for a selected set of categories, saved in parquet format, and aggregated into weekly DataFrames. The second section applies Prophet for trend analysis and highlights its shortcomings. Finally, a more robust neural network model is trained and evaluated.

---

## Contents of the `/src` folder

The **utils.py** file provides tools for scraping Wikipedia pageviews, aggregating the data, and plotting results.

- **list_category_pages(
    category: str,
    project: str = "en.wikipedia.org",
    subcat_depth: int = 0,
    user_agent: str = "WikiBarometer-Min/0.1 (contact@example.com)",
    sleep_s: float = 0.1) -> pd.DataFrame:**  
   &nbsp;&nbsp;&nbsp;Returns a dataframe of Wikipedia pages belonging to the specified **category**. The **project** parameter specifies which Wikipedia language version to query. Setting **subcat_depth > 0** allows the function to also include pages from subcategories up to the given depth. The **user_agent** and **sleep_s** arguments control API usage etiquette.

- **fetch_daily_pageviews(
    titles: List[str],
    start: str,
    end: str,
    project: str = "en.wikipedia.org",
    out_type: str = "df",
    out_dir: str | None = None,
    category_name: str | None = None
    user_agent: str = "WikiBarometer-Min/0.1 (contact@example.com)",
    sleep_s: float = 0.1):**  
  &nbsp;&nbsp;&nbsp;This function fetches daily Wikipedia pageview data for all article **titles** between **start** and **end** dates. The **project** parameter specifies the Wikipedia language version. The output can be returned as a pandas DataFrame (**out_type = "df"**), which is useful for small requests, or written to parquet files (**out_type = "parquet"**), which is optimal for large-scale requests. When writing parquet files, **category_name** provides a label to group the articles under a common folder (e.g., `category=Artificial_intelligence`). If not given, articles are saved under `"uncategorized"`. All parquet data is saved in the `data` folder created in the project’s root. The **user_agent** and **sleep_s** arguments control API usage etiquette.

- **to_weekly_df(data, categories = None, start = None, end = None)**:  
  &nbsp;&nbsp;&nbsp;Performs weekly aggregation on **data** provided in the DataFrame format. Optional parameters **categories**, **start**, and **end** allow limiting the data to a select subset of categories between given dates.

- **to_weekly_parquet(path, categories = None, start = None, end = None)**:  
  &nbsp;&nbsp;&nbsp;Performs weekly aggregation on data previously saved in parquet format at the location specified by **path**. Optional parameters **categories**, **start**, and **end** allow limiting the data to a select subset of categories between given dates.

- **plot(data, forecast = None, categories = None, train_end = None)**:  
   &nbsp;&nbsp;&nbsp;Plots the time evolution of pageviews for each category in **data**. If a **forecast** is provided, it is plotted alongside the actual values. The **categories** argument can be used to restrict the plot to a subset of categories. The optional **train_end** parameter adds a vertical line marking the end of the model’s training period.

