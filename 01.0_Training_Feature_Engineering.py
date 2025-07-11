# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql.window import Window
import pyspark.sql.functions as f
import pyspark.pandas as ps
from pyspark.sql.functions import pandas_udf
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Encounter Features
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
# MAGIC TODO: Add more features
# MAGIC * admit time
# MAGIC * admit day of week
# MAGIC * admit week of year

# COMMAND ----------

# parameterize your source and target data because you pull from different named resources in different environments
dbutils.widgets.text('source_schema', 'hls_ingest.clarity')
source_schema = dbutils.widgets.get('source_schema')

dbutils.widgets.text('target_schema', 'kp_catalog.hls_ml')
target_schema = dbutils.widgets.get('target_schema')

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")

# COMMAND ----------

admissions = spark.table(f'{source_schema}.admissions')

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import Column
w = Window.partitionBy("subject_id").orderBy("admittime")
df = (
    admissions
    # Find out when the patients last hospital discharge was
    .withColumn('last_discharge', f.lag(f.col('dischtime')).over(w))
    # If they don't have a recent discharge, then they are a new patient
    .withColumn('new_patient', f.when(f.col('last_discharge').isNull(), 1).otherwise(0))
    # Calculate if their most recent discharge was within 30 days
    .withColumn('IS_A_READMISSION', f.when(
        f.col('last_discharge') > f.date_trunc('dd', f.col('admittime')) - f.expr('INTERVAL 30 DAYS'), 1
    ).otherwise(0))
    
    .withColumn('30_DAY_READMISSION', f.lead('IS_A_READMISSION').over(w))
    .select(
      'subject_id',
      'admittime',
      'dischtime',
      'last_discharge',
      'new_patient',
      'IS_A_READMISSION',
      '30_DAY_READMISSION'
    )
    .orderBy(['subject_id','admittime'])
    )
df.display()

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import Column

w = Window.partitionBy("subject_id").orderBy("admittime")

enc_features = (
    admissions
    # Find out when the patients last hospital discharge was
    .withColumn('last_discharge', f.lag(f.col('dischtime')).over(w))
    # If they don't have a recent discharge, then they are a new patient
    .withColumn('new_patient', f.when(f.col('last_discharge').isNull(), 1).otherwise(0))
    # Calculate if their most recent discharge was within 30 days
    .withColumn('IS_A_READMISSION', f.when(
        f.col('last_discharge') > f.date_trunc('dd', f.col('admittime')) - f.expr('INTERVAL 30 DAYS'), 1
    ).otherwise(0))
    # Our target variable is predicting that the NEXT admission will be a readmission
    .withColumn('30_DAY_READMISSION', f.coalesce(f.lead('IS_A_READMISSION').over(w),f.lit(0)))
    # How many readmissions have they had in the last 6 months?
    .withColumn('30_DAY_READMISSION_6_months', f.sum(col('IS_A_READMISSION')).over( 
                                                          Window.partitionBy("subject_id").orderBy(col("admittime").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many readmissions have they had in the last 12 months?
    .withColumn('30_DAY_READMISSION_12_months', f.sum(col('IS_A_READMISSION')).over( 
                                                          Window.partitionBy("subject_id").orderBy(col("admittime").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    # How many total admissions have they had in the last 6 months?
    .withColumn('prev_admissions_6_months', f.count(col('admittime')).over( 
                                                          Window.partitionBy("subject_id").orderBy(col("admittime").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many total admissions have they had in the last 12 months?
    .withColumn('prev_admissions_12_months', f.count(col('admittime')).over( 
                                                          Window.partitionBy("subject_id").orderBy(col("admittime").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    
    .drop('last_discharge','IS_A_READMISSION'])

)

# COMMAND ----------

# If we were using spark ML, we'd use the built in one-hot encoder, but we're setting up data for an sklearn model. SKlearn onehot encoder only learns the distributed parition of data that it can see and therefore misses some classes sometimes. 
cat_cols = ['admission_type','admission_location','discharge_location','insurance','language','marital_status','race']
for cat_col in cat_cols:
    categories = [ (row[cat_col] if row[cat_col] is not None else "None").lower().replace(' ', '_').replace('.', '') for row in enc_features.select(cat_col).distinct().collect()]

    # Create one-hot encoded columns for each category
    for category in categories:
        enc_features = enc_features.withColumn("cat_col"+"_"+category, f.when(col(cat_col) == category, 1).otherwise(0))


# COMMAND ----------

df.display()

# COMMAND ----------

enc_features = calc_encounters_features(encounters)
enc_features.display()

# COMMAND ----------

enc_features_table = fe.create_table(
  name=f'{target_schema}.enc_features',
  primary_keys='Id',
  schema=enc_features.schema,
  description=f'Features derived from {source_schema}.encounters'
)

fe.write_table(
  name=f'{target_schema}.enc_features',
  df = enc_features,
  mode = 'merge'
)

# COMMAND ----------

# # You wouldn't need to do this - for you you might use the f.current_date() function in spark instead
# # see commented out code
# current_date = encounters.select(f.max('START')).collect()[0][0]

# filtered_enc_features = (
#   enc_features
#   # Taking away the last 30 days because we don't know if those patients readmitted or not
#   .filter(col('START') > current_date - f.expr(f'INTERVAL {int(training_months_history)*30 + 30} days'))
#   .filter(col('START') < current_date - f.expr(f'INTERVAL 30 days'))
#   )

# COMMAND ----------

# Define Patient Features logic
def calc_pat_features(data):
  data = ps.get_dummies(data.pandas_api(), columns=['MARITAL', 'RACE', 'ETHNICITY', 'GENDER'],dtype = 'int64').to_spark()
  return data.select(
    'Id',
    'HEALTHCARE_COVERAGE',
    'INCOME',
    'MARITAL_D',
    'MARITAL_M',
    'MARITAL_S',
    'MARITAL_W',
    'RACE_asian',
    'RACE_black',
    'RACE_hawaiian',
    'RACE_native',
    'RACE_other',
    'RACE_white',
    'ETHNICITY_hispanic',
    'ETHNICITY_nonhispanic',
    'GENDER_F',
    'GENDER_M'
  )
patients = spark.table(f'{source_schema}.patients')
patients_features = calc_pat_features(patients)

# COMMAND ----------

patients

# COMMAND ----------

enc_features_table = fe.create_table(
  name=f'{target_schema}.pat_features',
  primary_keys='Id',
  schema=patients_features.schema,
  description=f'Features derived from {source_schema}.patients'
)

fe.write_table(
  name=f'{target_schema}.pat_features',
  df = patients_features,
  mode = 'merge'
)

# COMMAND ----------

def calc_age_at_encounter(encounters, patients):
  return (
    encounters
    .join(patients, patients.Id == encounters.PATIENT)
    .withColumn("age_at_encounter", ((f.datediff(col('START'), col('BIRTHDATE'))) / 365.25))
    .select(encounters.Id, 'age_at_encounter')
    )

age_at_encounter = calc_age_at_encounter(spark.table(f'{source_schema}.encounters'), spark.table(f'{source_schema}.patients'))

customer_feature_table = fe.create_table(
  name=f'{target_schema}.age_at_encounter',
  primary_keys='Id',
  schema=age_at_encounter.schema,
  description='What age was the patient when they were admitted. Id in this table coresponds to encounters.Id'
)

fe.write_table(
  name=f'{target_schema}.age_at_encounter',
  df = age_at_encounter,
  mode = 'merge'
)

# COMMAND ----------


