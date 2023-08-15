# Databricks notebook source
# MAGIC %md 
# MAGIC # KP Walkthrough of Healthcare ML Demo
# MAGIC I have written this up as a point in time recommendation of how I think KP should build ML Projects given their current environment and features available? I will do my best to call out when I'm making KP specific decisions, and my recommendations and perhaps example code of what I would do differently if I did have those features.
# MAGIC
# MAGIC The HLS Lakehouse demo is sugar coated - It makes assumptions. This demo does not - it's exactly how I would build it in KP's env today

# COMMAND ----------

dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text('catalog', 'dbdemos', 'Catalog')
dbutils.widgets.text('db', 'hls_ml', 'Database')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Setup Notebook
# MAGIC imports the dataset into the local environment - in this case dbfs
# MAGIC KP Notes:
# MAGIC 1. This will likely only work in poc env at kp - this has a function that imports datasets from git repos, and dev/uat/psup/prod don't have internet access to pull them from
# MAGIC 2.

# COMMAND ----------

# MAGIC %run ./_resources/00-setup $reset_all_data=$reset_all_data $catalog=$catalog $db=$db

# COMMAND ----------

path = 'dbfs:/dbdemos/hls_ml/synthea/landing_zone'

# COMMAND ----------

dbutils.fs.ls(path)

# COMMAND ----------

df = spark.read.parquet('dbfs:/dbdemos/hls_ml/synthea/landing_zone/encounters')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select date_trunc('dd', START), count(1)
# MAGIC FROM test
# MAGIC group by 1

# COMMAND ----------

df.toPandas['START'].hist()

# COMMAND ----------


