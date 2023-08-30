# Databricks notebook source
from pyspark.sql.functions import col

# COMMAND ----------

def table_exists(table_name):
  catalog, schema, table = table_name.split('.')
  tables = spark.sql(f'''select table_name from {catalog}.information_schema.tables where table_schema = '{schema}' ''').collect()
  return table.lower() in [x['table_name'].lower() for x in tables]

# COMMAND ----------

# MAGIC %sql
# MAGIC drop schema dbdemos.hls_ml_hsp_admits cascade

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists dbdemos.hls_ml_hsp_admits;

# COMMAND ----------

(
  spark.read.format("delta").table('dbdemos.hls_ml_omop.visit_occurrence')
  # Filter down to emergency/urgentcare, and inpatient visits - excludes ambulatory, wellness, outpatient, and other
  .filter(col('VISIT_CONCEPT_ID').isin([
    # emergency and urgentcare
    9201,
    # inpatient
    9203]))
  ).createOrReplaceTempView('visit_updates')

# COMMAND ----------

spark.table('visit_updates').display()

# COMMAND ----------

if not table_exists('dbdemos.hls_ml_hsp_admits.visit_occurrence'):
  spark.table('visit_updates').write.saveAsTable('dbdemos.hls_ml_hsp_admits.visit_occurrence')
else:
  spark.sql("""
  MERGE INTO dbdemos.hls_ml_hsp_admits.visit_occurrence vo
  USING visit_updates u
  ON u.visit_occurrence_id = vo.visit_occurrence_id
  WHEN NOT MATCHED THEN INSERT *
            """)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC TODO: Consider other filters:
# MAGIC * Patients who do not live in the area of the hospital, or have missing residence info. Patients outside of hospital geography won't readmit to the same hospital
# MAGIC * Patients who died - dead patients can't readmit
# MAGIC * Patients who planned a visit in advance
# MAGIC * Patients who stay in hospital for more than 30 days
# MAGIC * Narrow down to just conditions that count against CMS Hospital Readmissions Reduction Program Readmission metrics
# MAGIC   * Acute Myocardial Infarction (AMI)
# MAGIC   * Chronic Obstructive Pulmonary Disease (COPD)
# MAGIC   * Heart Failure (HF)
# MAGIC   * Pneumonia
# MAGIC   * Coronary Artery Bypass Graft (CABG) Surgery
# MAGIC   * Elective Primary Total Hip Arthroplasty and/or Total Knee Arthroplasty (THA/TKA)
# MAGIC
