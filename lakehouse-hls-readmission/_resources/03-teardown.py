# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"
dbutils.widgets.text('catalog', 'dbdemos', 'Catalog')
dbutils.widgets.text('db', 'hls_ml', 'Database')
dbutils.widgets.text('folder', "/dbdemos/hls_ml/synthea", 'data folder')
folder=dbutils.widgets.get('folder')

landed_path = f"{folder}/landing_zone/"
vocab_path = f"{folder}/landing_vocab/"
source_path = f"{folder}/source/"
ingest_path = f"{folder}/ingest/"
