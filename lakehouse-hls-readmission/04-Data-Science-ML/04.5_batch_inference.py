# Databricks notebook source
import mlflow
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from databricks import feature_store
from datetime import datetime, timedelta

# COMMAND ----------

dbutils.widgets.text('model_name', 'hls_ml_demo')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('source_dbName', 'dbdemos.hls_ml_source')
source_dbName = dbutils.widgets.get('source_dbName')

dbutils.widgets.text('target_dbName', 'dbdemos.hls_ml_readmissions')
target_dbName = dbutils.widgets.get('target_dbName')

dbutils.widgets.text('external_location', 's3://one-env-uc-external-location/kp_ml_demo_dev/target/')
external_location = dbutils.widgets.get('external_location')

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look Up Production Model

# COMMAND ----------

model_details = client.get_latest_versions(model_name, ['Production'])[0]
model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pull Inference Data

# COMMAND ----------

df = spark.readStream.format('delta').table(f'{source_dbName}.encounters').filter(
      col('ENCOUNTERCLASS').isin([
        'emergency','inpatient','urgentcare'
      ]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Feature Store To Score Batch
# MAGIC You can stream in new inference data, score it, and write it out to another delta table in a single command.

# COMMAND ----------

#TODO edit data so that it has null discharge dates if they haven't been discharged yet, and use that to make predictions
#TODO add explainability to model so they know WHY a patient is likely to readmit

# COMMAND ----------

spark.sql(f"create schema if not exists {target_dbName}")

# COMMAND ----------

#TODO parameterize
tables = (
  spark.table('dbdemos.information_schema.tables')
  .filter(col('table_schema') == 'hls_ml_readmissions')
  .select('table_name')
).collect()
tables = [table['table_name'] for table in tables]
tables

# COMMAND ----------

# dbutils.fs.rm(f'{external_location}/checkpoints/readmissions_predictions',True)
# dbutils.fs.rm(f'{external_location}/readmissions_predictions', True)
# spark.sql(f'drop table {target_dbName}.readmission_predictions')

# COMMAND ----------

# from databricks.feature_store import FeatureLookup
# from databricks import feature_store
# import mlflow
# import pyspark.sql.functions as F
# import pyspark.sql.functions as f
# from pyspark.sql.functions import col
# from pyspark.sql import Window
# from mlflow.tracking import MlflowClient
# from sklearn.ensemble import RandomForestClassifier
# from datetime import datetime, timedelta

# df = spark.read.format('delta').table(f'{source_dbName}.encounters')

# encounters = spark.table(f'{source_dbName}.encounters')

# max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]

# df = (df.filter(col('START') > f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=30))).filter(
#       col('ENCOUNTERCLASS').isin([
#         'emergency','inpatient','urgentcare'
#       ])).select(
#   'Id', 'PATIENT'
# ))

# patient_features_table = f'dbdemos.hls_ml_features.patients_features'
# encounter_features_table = f'dbdemos.hls_ml_features.encounters_features'
# age_at_enc_features_table = f'dbdemos.hls_ml_features.age_at_encounter'
 
# patient_feature_lookups = [
#    FeatureLookup( 
#      table_name = patient_features_table,
#      feature_names = [
#       'HEALTHCARE_COVERAGE',
#       'INCOME',
#       'MARITAL_D',
#       'MARITAL_M',
#       'MARITAL_S',
#       'MARITAL_W',
#       'RACE_asian',
#       'RACE_black',
#       'RACE_hawaiian',
#       'RACE_native',
#       'RACE_other',
#       'RACE_white',
#       'ETHNICITY_hispanic',
#       'ETHNICITY_nonhispanic',
#       'GENDER_F',
#       'GENDER_M'],
#      lookup_key = ["PATIENT"],
#    ),
# ]
 
# encounter_feature_lookups = [
#    FeatureLookup( 
#      table_name = encounter_features_table,
#      feature_names = [
#         'TOTAL_CLAIM_COST',
#         'new_patient',
#         '30_DAY_READMISSION_6_months',
#         '30_DAY_READMISSION_12_months',
#         'prev_admissions_6_months',
#         'prev_admissions_12_months',
#         'ENCOUNTERCLASS_emergency',
#         'ENCOUNTERCLASS_inpatient',
#         'ENCOUNTERCLASS_urgentcare'
#      ],
#      lookup_key = ["Id"],
#    ),
# ]

# age_at_enc_feature_lookups = [
#    FeatureLookup( 
#      table_name = age_at_enc_features_table,
#      feature_names = ['age_at_encounter'],
#      lookup_key = ["Id"],
#    ),
# ]

# training_set = fs.create_training_set(
#   df,
#   feature_lookups = patient_feature_lookups + encounter_feature_lookups + age_at_enc_feature_lookups,
#   label = "Id",
#   # exclude_columns = ["Id", "PATIENT", 'STOP', 'START']
# )

# training_set.load_df().display()

# COMMAND ----------

# TODO add shap values, and prediction variable importances

# COMMAND ----------

if 'readmissions_predictions' not in tables: 
  # Filter down to just the last 30 days of admissions
  # We don't want predict on our entire history if we don't have to
  # We also don't want to make predictions on data we used for training, validation, or testing

  encounters = spark.table(f'{source_dbName}.encounters')

  max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]

  df = df.filter(col('START') > f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=30)))

  preds = (
    fs.score_batch(f"models:/{model_name}/Production", df)
    .select('Id', 'prediction')
    .withColumn('prediction_date', f.date_trunc('dd',f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ'))))
    .withColumn('model_version', f.lit(model_details.version))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f'{external_location}/checkpoints/readmissions_predictions')
    .trigger(once=True)
    .toTable(f"{target_dbName}.readmission_predictions", path=f'{external_location}/readmissions_predictions')
    .awaitTermination()
  )
else: 
  preds = (
    fs.score_batch(f"models:/{model_name}/Production", df)
    .select('Id', 'prediction')
    .withColumn('prediction_date', f.date_trunc('dd',f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ'))))
    .withColumn('model_version', f.lit(model_details.version))
    .writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f'{external_location}/checkpoints/readmissions_predictions')
    .trigger(once=True)
    .toTable(f"{target_dbName}.readmission_predictions", path=f'{external_location}/readmissions_predictions')
    .awaitTermination()
  )

# COMMAND ----------

display(spark.table(f"{target_dbName}.readmission_predictions"))

# COMMAND ----------

# MAGIC %sql
# MAGIC describe history dbdemos.hls_ml_readmissions.readmission_predictions

# COMMAND ----------

# spark.sql(f'drop table {dbName}.readmission_predictions')
# dbutils.fs.rm(f'{output_path}/readmissions_predictions', True)
# dbutils.fs.rm(f'{output_path}/checkpoint', True)

# COMMAND ----------


