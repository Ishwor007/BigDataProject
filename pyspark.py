#!/usr/bin/env python
# coding: utf-8


from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
column_names = ['tweetID', 'entity', 'sentiment', 'tweet_content']
training_data = spark.read.csv('hdfs://10.0.0.45:9000/data/twitter_training.csv', header=False, inferSchema=True).toDF(*column_names)
validation_data = spark.read.csv('hdfs://10.0.0.45:9000/data/twitter_validation.csv', header=False, inferSchema=True).toDF(*column_names)

training_data.show(5)
validation_data.show(5)

training_data = training_data.fillna({"tweet_content": ""})
validation_data = validation_data.fillna({"tweet_content": ""})

training_data.show(5)
validation_data.show(5)


training_data = training_data.fillna({"tweet_content": ""})
validation_data = validation_data.fillna({"tweet_content": ""})



from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer
from pyspark.sql.functions import col, regexp_replace
tokenizer = Tokenizer(inputCol="tweet_content", outputCol="words")
try:
    training_data = tokenizer.transform(training_data)
    training_data.show(5)
except Exception as e:
    print(f"Error during tokenization: {e}")



training_data = training_data.drop("words", "filtered_words", "features")
validation_data = validation_data.drop("words", "filtered_words", "features")

training_data = training_data.fillna({"tweet_content": ""})
validation_data = validation_data.fillna({"tweet_content": ""})

training_data = training_data.withColumn("tweet_content", regexp_replace(col("tweet_content"), "[^\\x00-\\x7F]", ""))
validation_data = validation_data.withColumn("tweet_content", regexp_replace(col("tweet_content"), "[^\\x00-\\x7F]", ""))

tokenizer = Tokenizer(inputCol="tweet_content", outputCol="words")
training_data = tokenizer.transform(training_data)
validation_data = tokenizer.transform(validation_data)

training_data.select("words").show(5)
validation_data.select("words").show(5)


remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
training_data = remover.transform(training_data)
validation_data = remover.transform(validation_data)

cv = CountVectorizer(inputCol="filtered_words", outputCol="features")
cv_model = cv.fit(training_data)
training_data = cv_model.transform(training_data)
validation_data = cv_model.transform(validation_data)



from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="sentiment", outputCol="label")
indexer_model = indexer.fit(training_data)
training_data = indexer_model.transform(training_data)
validation_data = indexer_model.transform(validation_data)

lr = LogisticRegression(featuresCol="features", labelCol="label")

lr_model = lr.fit(training_data)



training_data = training_data.fillna({"sentiment": "UNKNOWN"})
validation_data = validation_data.fillna({"sentiment": "UNKNOWN"})


training_data.select("sentiment").printSchema()
validation_data.select("sentiment").printSchema()


training_data.filter(col("sentiment").isNull()).show(5)
validation_data.filter(col("sentiment").isNull()).show(5)


training_data.select("sentiment").distinct().show()


from pyspark.sql.functions import when, col

training_data = training_data.withColumn(
    "sentiment",
    when(col("sentiment").isin(["Positive", "Negative", "Neutral"]), col("sentiment"))
    .otherwise("UNKNOWN")
)
validation_data = validation_data.withColumn(
    "sentiment",
    when(col("sentiment").isin(["Positive", "Negative", "Neutral"]), col("sentiment"))
    .otherwise("UNKNOWN")
)


from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="sentiment", outputCol="sentiment_index")
indexer_model = indexer.fit(training_data)

training_data = indexer_model.transform(training_data)
validation_data = indexer_model.transform(validation_data)



training_data.select("sentiment").printSchema()
validation_data.select("sentiment").printSchema()

training_data = training_data.fillna({"sentiment": "UNKNOWN"})
validation_data = validation_data.fillna({"sentiment": "UNKNOWN"})

training_data.select("sentiment").distinct().show()
validation_data.select("sentiment").distinct().show()




training_data.filter(col("sentiment").isNull()).count()
validation_data.filter(col("sentiment").isNull()).count()




training_data.select("sentiment").filter(~col("sentiment").rlike("^[A-Za-z ]+$")).show()
training_data = training_data.withColumn("sentiment", col("sentiment").cast("string"))
training_data.filter(col("sentiment") == "UNKNOWN").show(5)

training_data = training_data.withColumn(
    "sentiment",
    when(col("sentiment").isin(["Positive", "Negative", "Neutral"]), col("sentiment"))
    .otherwise("UNKNOWN")
)

training_data = training_data.filter(col("sentiment") != "UNKNOWN")
validation_data = training_data.filter(col("sentiment") != "UNKNOWN")


training_data.filter(col("sentiment") == "UNKNOWN").show(5)
validation_data.filter(col("sentiment") == "UNKNOWN").show(5)



from pyspark.ml.feature import StringIndexer

if 'sentiment_index' in training_data.columns:
    training_data = training_data.drop('sentiment_index')

if 'sentiment_index' in validation_data.columns:
    validation_data = validation_data.drop('sentiment_index')

indexer = StringIndexer(inputCol="sentiment", outputCol="sentiment_index")

indexer_model = indexer.fit(training_data)

training_data = indexer_model.transform(training_data)
validation_data = indexer_model.transform(validation_data)

training_data.show(5)
validation_data.show(5)


predictions = lr_model.transform(validation_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
try:
    accuracy = evaluator.evaluate(predictions)
    print(f"Validation Accuracy: {accuracy:.2f}")
except Exception as e:
    print(f"An error occurred: {e}")


from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

validation_data_with_predictions = lr_model.transform(validation_data)

validation_data_with_predictions.printSchema()

validation_data_with_predictions.select("sentiment_index", "prediction").show(5)

evaluator = MulticlassClassificationEvaluator(labelCol="sentiment_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(validation_data_with_predictions)
print(f"Validation Accuracy: {accuracy:.2f}")

validation_results = validation_data_with_predictions.select("sentiment_index", "prediction").toPandas()
label_map = {0.0: "Negative", 1.0: "Neutral", 2.0: "Positive"}
validation_results["prediction"] = validation_results["prediction"].map(label_map)
validation_results["sentiment_index"] = validation_results["sentiment_index"].map(label_map)

plt.figure(figsize=(8, 6))
sns.countplot(data=validation_results, x="prediction", palette="viridis")
plt.title("Prediction Distribution")
plt.xlabel("Predicted Sentiments")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 8))
validation_results["prediction"].value_counts().plot.pie(autopct='%1.1f%%', cmap="viridis", legend=True)
plt.title("Sentiment Prediction Proportions")
plt.ylabel("")
plt.show()

confusion_matrix = pd.crosstab(validation_results["sentiment_index"], validation_results["prediction"], rownames=["True Label"], colnames=["Predicted Label"])
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("True Sentiment")
plt.show()
