# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

dbutils.widgets.text('catalog', 'dbdemos', 'Catalog')
catalog=dbutils.widgets.get('catalog')

dbutils.widgets.text('db', 'hls_ml_source', 'Database')
db=dbutils.widgets.get('db')

dbutils.widgets.text('folder', "s3://one-env-uc-external-location/kp_ml_demo_dev", 'data folder')
folder=dbutils.widgets.get('folder')

ingest_path = f"{folder}/ingest/"
target_path = f"{folder}/target/"

# COMMAND ----------

dbutils.fs.ls(target_path+'checkpoints')

# COMMAND ----------

if reset_all_data:
  dbutils.fs.rm(target_path)
  spark.sql('drop schema dbdemos.hls_ml_source cascade')

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists dbdemos.hls_ml_source

# COMMAND ----------

tables = [src_path[1].replace('/','') for src_path in dbutils.fs.ls(ingest_path)]

# COMMAND ----------

for table in tables:
  print(table)
  (spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "parquet")
    # The schema location directory keeps track of your data schema over time
    .option("cloudFiles.schemaLocation", f"{target_path}/schemas/{table}")
    .load(f"{ingest_path}/{table}")
    .writeStream
    .option("checkpointLocation", f"{target_path}/checkpoints/{table}")
    .trigger(availableNow=True)
    .toTable(f"{catalog}.{db}.{table}")
    .awaitTermination()
  )

# COMMAND ----------

# MAGIC %sql
# MAGIC describe history dbdemos.hls_ml_source.patients

# COMMAND ----------


