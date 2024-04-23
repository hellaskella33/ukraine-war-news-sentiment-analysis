# Sentiment Analysis of News Articles on the Russia-Ukraine War

## Project Overview

This project aims to analyze the sentiment of news articles related to the Russia-Ukraine war from different countries and media outlets, including the United States, Ukraine, the United Kingdom, Poland, and Germany. The primary goals are:

1. Identify potential biases, propaganda, or contrasting narratives in the portrayal of the war across these different sources.
2. Compare and contrast the sentiment scores and overall portrayal of the war events, actions, and key figures across the selected countries and media outlets.
3. Investigate how the sentiment and framing of the war narrative may have evolved over time, from the start of the war on February 24, 2022, until the present day.
4. Provide insights into the potential influence of media narratives on public opinion and perceptions of the Russia-Ukraine war in different regions.

## Data Collection

The project utilizes the [NewsCatcher API](https://newscatcherapi.com/) to collect news articles from reputable sources in the target countries. The data collection process involves:

1. **Country and Media Outlet Selection**:
   - We selected the following countries for analysis: United States, Ukraine, United Kingdom, Poland, and Germany.
   - For each country, we identified the most influential and widely-read media outlets based on their circulation, online presence, and reputation.

2. **API Query Parameters**:
   - We constructed API queries using the NewsCatcher API to retrieve articles from the selected media outlets in each country.
   - The queries included the following parameters:
     - `country`: The country code for each selected country (e.g., US, UA, GB, PL, DE).
     - `media`: The media outlet names or domains for the selected outlets in each country.
     - `topic`: Keywords related to the Russia-Ukraine war, such as "Russia," "Ukraine," "war," "conflict," etc.
     - `from_date` and `to_date`: The date range for article retrieval, spanning from February 24, 2022, to the present day.
     - `language`: The primary language of the articles (e.g., English, Ukrainian, Polish, German).

3. **Data Retrieval and Storage**:
   - We executed the API queries and retrieved the news articles from the selected media outlets.
   - The retrieved data included article titles, URLs, publication dates, content, and other relevant metadata.
   - We stored the collected data in a structured format (e.g., JSON or CSV) for further processing and analysis.

4. **Data Cleaning and Preprocessing**:
   - We performed necessary data cleaning and preprocessing steps on the collected articles.
   - This included removing duplicates, handling missing data, and formatting the text for subsequent analysis.

The data collection process resulted in a diverse and representative sample of news articles related to the Russia-Ukraine war from influential media outlets in the selected countries. This dataset forms the foundation for our sentiment analysis and investigation of media biases and narratives.

## Sentiment Analysis Approach

To capture the nuances and biases in the portrayal of the Russia-Ukraine war, the project employs advanced sentiment analysis techniques, including:

1. **Multi-Class Sentiment Classification**: Defining sentiment classes specific to the domain, such as "pro-Russia," "pro-Ukraine," "anti-war," "pro-war," and "neutral."
2. **Continuous Sentiment Score**: Assigning a continuous sentiment score or rating (e.g., -5 to +5) to each news article, capturing the intensity and direction of sentiment more precisely.

The project involves manually annotating a representative sample of the collected news articles to create a labeled dataset for training and evaluating sentiment analysis models.

## Analysis and Visualization

The sentiment analysis results will be analyzed and visualized to identify patterns, trends, and contrasting narratives across different countries and media outlets. The analysis will include:

1. Comparing sentiment scores or labels across sources and countries for the same events or topics related to the war.
2. Identifying sources or countries that consistently exhibit extreme positive or negative sentiment, indicating potential biases or propaganda.
3. Analyzing the language and framing used in biased or propagandistic articles to understand the underlying narratives and rhetorical strategies.
4. Visualizing the evolution of sentiment over time and correlating sentiment changes with significant events or developments in the war timeline.

The findings will be presented through interactive dashboards, visualizations, and written reports, highlighting the key insights, biases, and contrasting narratives identified across different countries and media outlets.

## Acknowledgments

This project is made possible thanks to the [NewsCatcher API](https://newscatcherapi.com/), which provides access to news articles from various sources worldwide. We would like to express our gratitude to NewsCatcher for their support and for granting us free access to their API for the duration of this project.
