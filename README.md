# Sentiment Analysis of News Articles on the Russia-Ukraine War

## Project Overview

This project aims to analyze the sentiment of news articles related to the Russia-Ukraine war from different countries and media outlets, including the United States, Ukraine, the United Kingdom, Poland, and Germany. The primary goals are:

1. Identify potential biases, propaganda, or contrasting narratives in the portrayal of the war across these different sources.
2. Compare and contrast the sentiment scores and overall portrayal of the war events, actions, and key figures across the selected countries and media outlets.
3. Investigate how the sentiment and framing of the war narrative may have evolved over time, from the start of the war on February 24, 2022, until the present day.
4. Provide insights into the potential influence of media narratives on public opinion and perceptions of the Russia-Ukraine war in different regions.

## Data Collection

The project utilizes the [NewsCatcher API](https://newscatcherapi.com/) to collect news articles from reputable sources in the target countries. The data collection process involves:

1. Identifying relevant news sources for each country, covering a diverse range of perspectives and ideologies.
2. Retrieving articles from these sources using the NewsCatcher API, covering the date range from February 24, 2022, to the present day.
3. Storing and organizing the collected data by source, country, and date for further analysis.

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
