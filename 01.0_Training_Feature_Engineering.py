# Databricks notebook source
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient
import pyspark.sql.functions as f
import pyspark.pandas as ps

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

fe = FeatureEngineeringClient()

# COMMAND ----------

encounters = spark.table(f'{source_schema}.encounters')

# COMMAND ----------

# No need to figure out what all these things do - mostly we're just creating the Target Variable 30_DAY_READMISSION
# As well as creating some encounter based features
def calc_encounters_features(data):
  df = (
    data
    # Filter down to just hospitalizations
    # Consider other filters in the future, see markdown above
    .filter(
      col('ENCOUNTERCLASS').isin([
        'emergency','inpatient','urgentcare'
      ])
    )
    # Find out when the patients last hospital discharge was
    .withColumn('last_discharge', f.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
    # If they don't have a recent discharge, then they are a new patient
    .withColumn('new_patient', f.when(col('last_discharge').isNull(), 1).otherwise(0))
    # Calculate if their most recent discharge was within 30 days
    .withColumn('30_DAY_READMISSION', f.when(col('START').cast('long') - col('last_discharge').cast('long') < 60*60*24*30, 1).otherwise(0))
    # How many readmissions have they had in the last 6 months?
    .withColumn('30_DAY_READMISSION_6_months', f.sum(col('30_DAY_READMISSION')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many readmissions have they had in the last 12 months?
    .withColumn('30_DAY_READMISSION_12_months', f.sum(col('30_DAY_READMISSION')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    # How many total admissions have they had in the last 6 months?
    .withColumn('prev_admissions_6_months', f.count(col('START')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many total admissions have they had in the last 12 months?
    .withColumn('prev_admissions_12_months', f.count(col('START')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    # 

    # .select('PATIENT', 'START', 'STOP', 'last_discharge','30_DAY_READMISSION', 'prev_admissions_6_months', 'prev_admissions_12_months')
  ).pandas_api()

  pd_df = ps.get_dummies(df, columns=['ENCOUNTERCLASS'],dtype = 'int64').to_spark()

  final_data = pd_df.select('Id','TOTAL_CLAIM_COST','new_patient','30_DAY_READMISSION_6_months','30_DAY_READMISSION_12_months','prev_admissions_6_months', 'prev_admissions_12_months', 'ENCOUNTERCLASS_emergency', 'ENCOUNTERCLASS_inpatient', 'ENCOUNTERCLASS_urgentcare')
  
  return final_data

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


