# Enhancing Customer Support Efficiency  
### AI-Based Query Categorization & Resolution Time Prediction  
🚀 *CSCE 5300: Big Data and Data Science | Group 12 Project*

---

## 🧠 Project Overview

This project addresses the challenge of efficiently managing large volumes of multilingual customer support queries. Traditional manual approaches result in delayed responses and poor customer satisfaction. Our solution leverages **Big Data** and **Machine Learning** techniques to:

- Automatically categorize incoming support queries (e.g., Technical, Sales, etc.)
- Predict the expected resolution time based on query metadata
- Improve resource allocation and customer experience

---

## 🧰 Tech Stack

| Component           | Technology Used                         |
|---------------------|------------------------------------------|
| Data Processing     | Hive, Python (Pandas), Spark             |
| Data Storage        | Apache Cassandra                        |
| ML Model Training   | Apache Spark MLlib                      |
| Visualization       | AWS SageMaker                           |
| NLP Techniques      | TF-IDF, Word2Vec, VADER, TextBlob        |
| Dataset Source      | [Kaggle - Multilingual Customer Support](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets/data)


##Methodology
##Phase 1: Data Processing

Load and preprocess customer queries (handle missing values, tokenize, clean)

Extract features: urgency score, sentiment, keyword embeddings (TF-IDF / Word2Vec)

Store structured output in Cassandra


###Phase 2: Machine Learning

Train classifier to categorize queries using Spark MLlib

Train regression model to predict resolution time

Evaluate model performance using accuracy, RMSE, and F1 Score


###Visualization & Insights

Use AWS SageMaker to create dashboards showing:

Predicted vs. actual resolution times

Category-wise query distribution

Urgency heatmaps


##Key Features

NLP-powered sentiment and urgency analysis

Multilingual query handling

Scalable architecture for future expansion

Integration with Hive, Cassandra, and SageMaker



##Acknowledgments

Kaggle Dataset: Multilingual Customer Support Tickets

AWS Documentation (S3, SageMaker, EMR)

Apache Hive, Spark, Cassandra Docs

VADER Sentiment, TextBlob, and Gensim Word2Vec


---

## 📁 Project Structure

```bash
enhancing-customer-support-ai/
│
├── data/                         # Raw & processed datasets
│   ├── raw/
│   └── processed/
│
├── hive/                         # Hive SQL scripts
│   ├── create_tables.sql
│   ├── merge_datasets.sql
│   └── sample_queries.sql
│
├── cassandra/                    # Cassandra DB schema & scripts
│   ├── schema.cql
│   └── insert_data.cql
│
├── notebooks/                    # Jupyter & SageMaker notebooks
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
│
├── spark_ml/                     # Spark ML scripts
│   ├── query_classifier.py
│   └── resolution_predictor.py
│
├── nlp_utils/                    # NLP helper functions
│   ├── sentiment_analysis.py
│   └── tfidf_extractor.py
│
├── aws/                          # AWS EMR + SageMaker setup guides
│   └── emr_setup.md
│
├── presentation/                 # Slides & presentations
│   └── Final_Presentation.pdf
│
├── reports/                      # Proposal and documentation
│   ├── Project_Proposal.pdf
│   └── Final_Report.md
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview (this file)
└── LICENSE                       # License (MIT)

--- 

