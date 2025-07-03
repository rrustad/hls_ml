# Databricks notebook source
import mlflow
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from datetime import datetime, timedelta

# COMMAND ----------

dbutils.widgets.text('model_name', 'kp_catalog.hls_ml.readmissions_risk')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('source_schema', 'hls_ingest.clarity')
source_schema = dbutils.widgets.get('source_schema')

dbutils.widgets.text('target_schema', 'kp_catalog.hls_ml')
target_schema = dbutils.widgets.get('target_schema')

dbutils.widgets.text('external_location', 'abfss://kp-external-location@oneenvadls.dfs.core.windows.net/hls_ml')
external_location = dbutils.widgets.get('external_location')

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look Up Production Model

# COMMAND ----------

model_details = client.get_model_version_by_alias(model_name, "production")
model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pull Inference Data

# COMMAND ----------

#TODO edit data so that it has null discharge dates if they haven't been discharged yet, and use that to make predictions
#TODO add explainability to model so they know WHY a patient is likely to readmit

# COMMAND ----------

spark.sql(f"create schema if not exists {target_schema}")

# COMMAND ----------

catalog, schema, model = model_name.split('.')

# COMMAND ----------

encounters = spark.table(f'{source_schema}.encounters')

max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]

# COMMAND ----------


df = (
  encounters
  .filter(f.date_trunc('dd', col('STOP')) == f.date_trunc('dd',f.lit(max_enc_date)))
  .filter(col('ENCOUNTERCLASS').isin(['emergency','inpatient','urgentcare']))
)

# COMMAND ----------

preds = (
  fe.score_batch(model_uri=f"models:/{model_name}@production", df=df)
  # .select('Id', 'prediction')
  .withColumn('prediction_date', f.date_trunc('dd',f.lit(max_enc_date)))
  .withColumn('model_version', f.lit(model_details.version))
  .write
  .mode('append')
  #TODO: table name comes from the tag "inference_table" so there's no mismatch
  .saveAsTable(f'{target_schema}.{model}_predictions')
  # .writeStream
  # .format("delta")
  # .outputMode("append")
  # .option("checkpointLocation", f'{external_location}/checkpoints/readmissions_predictions')
  # .trigger(once=True)
  # .toTable(f"{target_schema}.readmission_predictions", path=f'{external_location}/readmissions_predictions')
  # .awaitTermination()
)
