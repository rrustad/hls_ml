# Databricks notebook source
# MAGIC %md 
# MAGIC # KP Walkthrough of Healthcare ML Demo
# MAGIC I have written this up as a point in time recommendation of how I think KP should build ML Projects given their current environment and features available? I will do my best to call out when I'm making KP specific decisions, and my recommendations and perhaps example code of what I would do differently if I did have those features.
# MAGIC
# MAGIC The HLS Lakehouse demo is sugar coated - It makes assumptions. This demo does not - it's exactly how I would build it in KP's env today. Ex. Other demos only load the data once, and don't fully show the incremental capabilities of Databricks. This demo creates incememental data, so you can see what it would be like to build your tables efficiently with incememntal ETL

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

# MAGIC %md
# MAGIC ### Demo Data
# MAGIC The dataset that we're working with is simulated EHR data that theoretically starts in June 2013 with the full patient history of patients who existed at that time. We are choosing to start "Today" 1 year prior to the end of the dataset which is ~ July 2023, so we're assuming today is July 2022.

# COMMAND ----------

# MAGIC %sql 
# MAGIC select date_trunc('dd', START), count(1)
# MAGIC FROM parquet.`dbfs:/dbdemos/hls_ml/synthea/landing_zone/encounters`
# MAGIC group by 1

# COMMAND ----------

df.toPandas['START'].hist()

# COMMAND ----------


