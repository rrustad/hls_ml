# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-lakehouse-hls-readmission-loca` from the dropdown menu ([open cluster configuration](https://e2-demo-field-eng.cloud.databricks.com/#setting/clusters/0815-003942-xcdwu5nj/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('lakehouse-hls-readmission')` or re-install the demo: `dbdemos.install('lakehouse-hls-readmission')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # ML: Predict and reduce 30 Day Readmissions
# MAGIC
# MAGIC <img src="https://raw.githubusercontent.com/databricks-demos/dbdemos-resources/main/images/hls/patient-readmission/hls-patient-readmision-flow-4.png" style="float: right; margin-left: 30px; margin-top:10px" width="650px" />
# MAGIC
# MAGIC We now have our data cleaned and secured. We saw how to create and analyze our first patient cohorts.
# MAGIC
# MAGIC Let's now take it to the next level and start building a Machine Learning model to predict wich patients are at risk. 
# MAGIC
# MAGIC We'll then be able to explain the model at a statistical level, understanding which features increase the readmission risk. This information will be critical in how we can specialize care for a specific patient population.

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $reset_all_data=false $catalog=dbdemos $db=hls_patient_readmission

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Building our Features for Patient readmission analysis
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/hls/patient-readmission/patient-risk-ds-flow-1.png?raw=true" width="700px" style="float: right; margin-left: 10px;" />
# MAGIC Our first step is now to merge our different tables adding extra features to be able to train our model.
# MAGIC
# MAGIC We will use the `encounters_readmissions` table as the label we want to predict:
# MAGIC
# MAGIC is there a readmission within 30 days.

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema if not exists dbdemos.hls_ml_features

# COMMAND ----------

from pyspark.sql import Window
import pyspark.pandas as ps

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

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
    .withColumn('last_discharge', F.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
    # If they don't have a recent discharge, then they are a new patient
    .withColumn('new_patient', F.when(col('last_discharge').isNull(), 1).otherwise(0))
    # Calculate if their most recent discharge was within 30 days
    .withColumn('30_DAY_READMISSION', F.when(col('START').cast('long') - col('last_discharge').cast('long') < 60*60*24*30, 1).otherwise(0))
    # How many readmissions have they had in the last 6 months?
    .withColumn('30_DAY_READMISSION_6_months', F.sum(col('30_DAY_READMISSION')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many readmissions have they had in the last 12 months?
    .withColumn('30_DAY_READMISSION_12_months', F.sum(col('30_DAY_READMISSION')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    # How many total admissions have they had in the last 6 months?
    .withColumn('prev_admissions_6_months', F.count(col('START')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*180, 0)
                                                          ))
    # How many total admissions have they had in the last 12 months?
    .withColumn('prev_admissions_12_months', F.count(col('START')).over( 
                                                          Window.partitionBy("PATIENT").orderBy(col("START").cast("long")).rangeBetween(-60*60*24*365, 0)
                                                          ))
    # 

    # .select('PATIENT', 'START', 'STOP', 'last_discharge','30_DAY_READMISSION', 'prev_admissions_6_months', 'prev_admissions_12_months')
  ).pandas_api()

  pd_df = ps.get_dummies(df, columns=['ENCOUNTERCLASS'],dtype = 'int64').to_spark()

  final_data = pd_df.select('Id','TOTAL_CLAIM_COST','new_patient','30_DAY_READMISSION_6_months','30_DAY_READMISSION_12_months','prev_admissions_6_months', 'prev_admissions_12_months', 'ENCOUNTERCLASS_emergency', 'ENCOUNTERCLASS_inpatient', 'ENCOUNTERCLASS_urgentcare')
  
  return final_data

encounters_features = calc_encounters_features(spark.table('dbdemos.hls_ml_source.encounters'))

# Create feature table with `customer_id` as the primary key.
# Take schema from DataFrame output by compute_customer_features
customer_feature_table = fs.create_table(
  name='dbdemos.hls_ml_features.encounters_features',
  primary_keys='Id',
  schema=encounters_features.schema,
  description='Encounter features'
)

fs.write_table(
  name='dbdemos.hls_ml_features.encounters_features',
  df = encounters_features,
  mode = 'merge'
)

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

patients_features = calc_pat_features(spark.table('dbdemos.hls_ml_source.patients'))

# Create feature table with `customer_id` as the primary key.
# Take schema from DataFrame output by compute_customer_features
customer_feature_table = fs.create_table(
  name='dbdemos.hls_ml_features.patients_features',
  primary_keys='Id',
  schema=patients_features.schema,
  description='Patient features'
)

fs.write_table(
  name='dbdemos.hls_ml_features.patients_features',
  df = patients_features,
  mode = 'merge'
)

# COMMAND ----------

def calc_age_at_encounter(encounters, patients):
  return (
    encounters
    .join(patients, patients.Id == encounters.PATIENT)
    .withColumn("age_at_encounter", ((F.datediff(col('START'), col('BIRTHDATE'))) / 365.25))
    .select(encounters.Id, 'age_at_encounter')
    )

age_at_encounter = calc_age_at_encounter(spark.table('dbdemos.hls_ml_source.encounters'), spark.table('dbdemos.hls_ml_source.patients'))

customer_feature_table = fs.create_table(
  name='dbdemos.hls_ml_features.age_at_encounter',
  primary_keys='Id',
  schema=age_at_encounter.schema,
  description='What age was the patient when they were admitted. Id in this table coresponds to encounters.Id'
)

fs.write_table(
  name='dbdemos.hls_ml_features.age_at_encounter',
  df = age_at_encounter,
  mode = 'merge'
)
