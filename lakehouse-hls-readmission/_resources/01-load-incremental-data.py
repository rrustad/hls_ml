# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
dbutils.widgets.text('catalog', 'dbdemos', 'Catalog')
dbutils.widgets.text('db', 'hls_ml', 'Database')
dbutils.widgets.text('folder', "/dbdemos/hls_ml/synthea", 'data folder')
folder=dbutils.widgets.get('folder')
dbutils.widgets.dropdown('mode','update',['update','init'])
mode=dbutils.widgets.get('mode')

landed_path = 

# COMMAND ----------



# COMMAND ----------


