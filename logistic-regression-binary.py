# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("LogisticRegressionBollywood").getOrCreate()

# Load the dataset
data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Define the target column and feature columns
target_col = "target"
feature_cols = [col for col in data.columns if col != target_col]

# Create a vector assembler to combine feature columns into a single vector column
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Transform the data using the vector assembler
data = vector_assembler.transform(data)

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=123)

# Create and train a Logistic Regression classifier
lr = LogisticRegression(labelCol=target_col, featuresCol="features")
model = lr.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model using a MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

# Print the accuracy of the model
print(f"Accuracy: {accuracy}")

# Stop the Spark session
spark.stop()
