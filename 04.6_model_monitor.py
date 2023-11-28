# Databricks notebook source
import mlflow
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from databricks import feature_store
from datetime import datetime, timedelta
from pyspark.sql import Window

# COMMAND ----------

dbutils.widgets.text('model_name', 'hls_ml_demo')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('source_dbName', 'dbdemos.hls_ml_source')
source_dbName = dbutils.widgets.get('source_dbName')

dbutils.widgets.text('target_dbName', 'dbdemos.hls_ml_readmissions')
target_dbName = dbutils.widgets.get('target_dbName')

dbutils.widgets.text('retrain_threshold', '.59')
retrain_threshold = float(dbutils.widgets.get('retrain_threshold'))

dbutils.widgets.text('external_location', 's3://one-env-uc-external-location/kp_ml_demo_dev/target/')
external_location = dbutils.widgets.get('external_location')

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look Up Production Model

# COMMAND ----------

predictions = spark.table(f"{target_dbName}.readmission_predictions")

# COMMAND ----------

encounters = spark.table('dbdemos.hls_ml_source.encounters')

max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]

windowSpec = Window.partitionBy("PATIENT").orderBy("START")

outcomes = (
  encounters
  # We can't definitively say if anyone from the last 30 days has readmitted in 30 days
  .filter(col('START') < f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=30)))
  # Calculate the target variable
  .withColumn('last_discharge', f.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
  # Calculate if their most recent discharge was within 30 days
  .withColumn('30_DAY_READMISSION', f.when(col('START').cast('timestamp').cast('long') - col('last_discharge').cast('timestamp').cast('long') < 60*60*24*30, 1).otherwise(0))
  .select('Id', 'PATIENT','STOP', 'START', '30_DAY_READMISSION')
  .orderBy(['START'], desc=True)
)


# COMMAND ----------

compare = (
  predictions
  .join(outcomes, 'Id', 'inner')
  .withColumn('correct', (col('prediction') == col('30_DAY_READMISSION')).cast('int'))
  .select('Id', 'START', 'STOP', 'correct','model_version')
  .write
  .mode('overwrite')
  .option("mergeSchema", "true")
  .saveAsTable(f"{target_dbName}.readmission_prediction_outcomes")
)


# COMMAND ----------

spark.table(f"{target_dbName}.readmission_prediction_outcomes").display()

# COMMAND ----------

(
  spark.table(f"{target_dbName}.readmission_prediction_outcomes")
  .groupBy(f.date_trunc('dd','START').alias('start_date'), col('model_version'))
  .agg(f.mean('correct'),f.count('correct'))
  .withColumn('theshold', f.lit(.55))
).display()

# COMMAND ----------

prod_model_details = client.get_latest_versions(model_name, ['Production'])[0]

#assumes that the max version is the latest production version
max_model_version = spark.table(f"{target_dbName}.readmission_prediction_outcomes").select(f.max('model_version')).collect()[0][0]

max_model_is_current = max_model_version == prod_model_details.version
max_model_is_current

# COMMAND ----------

below_threshold = (
  spark.table(f"{target_dbName}.readmission_prediction_outcomes")
  .groupBy(f.date_trunc('dd','START').alias('start_date'))
  .agg(f.mean('correct').alias('daily_accuracy'))
  .select(f.min('daily_accuracy') < retrain_threshold )
).collect()[0][0]
below_threshold

# COMMAND ----------

dbutils.jobs.taskValues.set('retrain_model', below_threshold and max_model_is_current)

below_threshold and max_model_is_current

# COMMAND ----------

# MAGIC %md
# MAGIC Another metric that you could measure is SLA - ie. what time were the predictions made available? This would be difficult to measure in the demo, but straight forward in the real world

# COMMAND ----------


