# Databricks notebook source
import mlflow
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from datetime import datetime, timedelta
from pyspark.sql import Window

# COMMAND ----------

dbutils.widgets.text('model_name', 'kp_catalog.hls_ml.readmissions_risk')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('source_schema', 'hls_ingest.clarity')
source_schema = dbutils.widgets.get('source_schema')

dbutils.widgets.text('target_schema', 'kp_catalog.hls_ml')
target_schema = dbutils.widgets.get('target_schema')

dbutils.widgets.text('retrain_threshold', '.59')
retrain_threshold = float(dbutils.widgets.get('retrain_threshold'))

dbutils.widgets.text('external_location', 's3://one-env-uc-external-location/kp_ml_demo_dev/target/')
external_location = dbutils.widgets.get('external_location')

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

model_details = client.get_model_version_by_alias(model_name, "production")
model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look Up Production Model

# COMMAND ----------

catalog, schema, model = model_name.split('.')

# COMMAND ----------

predictions = spark.table(f"{target_schema}.{model}_predictions")

# COMMAND ----------

encounters = spark.table(f'{source_schema}.encounters')

max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]

windowSpec = Window.partitionBy("PATIENT").orderBy("START")

outcomes = (
  encounters
  # We can't definitively say if anyone from the last 30 days has readmitted in 30 days
  .filter(col('STOP') < f.lit(max_enc_date - timedelta(days=30)))
  # Calculate the target variable
  .withColumn('last_discharge', f.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
  # Calculate if their most recent discharge was within 30 days
  .withColumn('30_DAY_READMISSION', f.when(col('START').cast('timestamp').cast('long') - col('last_discharge').cast('timestamp').cast('long') < 60*60*24*30, 1).otherwise(0))
  .select('Id', 'PATIENT','STOP', 'START', '30_DAY_READMISSION')
  .orderBy(['STOP'], desc=True)
)


# COMMAND ----------

compare = (
  predictions
  .join(outcomes.drop('PATIENT','START','STOP'), 'Id', 'inner')
  .withColumn('correct', (col('prediction') == col('30_DAY_READMISSION')).cast('int'))
  .select('Id', 'START', 'STOP', '30_DAY_READMISSION', 'model_version')
  .join(spark.table(f'{target_schema}.{model}_predictions').select('Id', 'prediction'), 'Id', 'inner')
  .withColumn('prediction', col('prediction').cast(IntegerType()))
  # .withColumn('START', col('START') + f.expr(f'INTERVAL {727+84} DAYS'))
  # .withColumn('STOP', col('STOP') + f.expr(f'INTERVAL {727+84} DAYS'))
  .write
  .mode('overwrite')
  .option("mergeSchema", "true")
  .saveAsTable(f"{target_schema}.{model}_outcomes")
)


# COMMAND ----------



# COMMAND ----------

df = spark.table(f"{target_schema}.{model}_outcomes")
df.display()

# COMMAND ----------

(
  spark.table(f"{target_schema}.{model}_outcomes")
  # .join(spark.table(f'{target_schema}.{model}_predictions').select('Id', 'prediction'), 'Id', 'inner')
  .withColumn('correct', (f.col('prediction') == f.col('30_DAY_READMISSION')).cast('int'))
  .groupBy(f.date_trunc('dd','STOP').alias('stop_date'))
  .agg(f.mean('correct'),f.count('correct'))
  .withColumn('theshold', f.lit(.55))
).display()

# COMMAND ----------

retrain_model = (
  spark.table(f"{target_schema}.{model}_outcomes")
  # .join(spark.table(f'{target_schema}.{model}_predictions').select('Id', 'prediction'), 'Id', 'inner')
  .withColumn('correct', (f.col('prediction') == f.col('30_DAY_READMISSION')).cast('int'))
  .groupBy(f.date_trunc('dd','STOP').alias('stop_date'))
  .agg(f.mean('correct').alias('daily_accuracy'))
  .select(f.min('daily_accuracy') < retrain_threshold)
).collect()[0][0]

if retrain_model is None:
  retrain_model = False


dbutils.jobs.taskValues.set('retrain_model', retrain_model)

retrain_model

# COMMAND ----------

# MAGIC %md
# MAGIC Another metric that you could measure is SLA - ie. what time were the predictions made available? This would be difficult to measure in the demo, but straight forward in the real world

# COMMAND ----------

retrain_model is None

# COMMAND ----------


