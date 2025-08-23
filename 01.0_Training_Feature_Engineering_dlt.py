# Databricks notebook source
import dlt
# from pyspark.sql.functions import col
# from pyspark.sql.window import Window
# import pyspark.sql.functions as f
# import pyspark.pandas as ps
# from pyspark.sql.functions import pandas_udf
# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd

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

source_schema = dlt.pipeline_config.get("source_schema", "kp_catalog.mimic_incr")
target_schema = dlt.pipeline_config.get("target_schema", "kp_catalog.hls_ml")

# COMMAND ----------

# spark.sql(f'drop schema kp_catalog.hls_ml cascade')

# COMMAND ----------

admissions = (spark.readStream.option("readChangeFeed", "true").table(f'{source_schema}.admissions')
  .filter(col('_change_type').isin('insert', 'update_postimage'))
  .withColumnRenamed('_commit_timestamp', 'commit_timestamp')
  .drop('_change_type')
  .drop('_commit_version'))

tables = spark.table(f'{target_schema.split(".")[0]}.information_schema.tables').select('table_name').collect()

if '_admissions' in [x[0] for x in tables]:
  spark.sql(f'truncate table {target_schema}._admissions')


# COMMAND ----------

catalog, schema = target_schema.split('.')

(admissions.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"/Volumes/{catalog}/{schema}/streaming_checkpoints/_admissions")
    .trigger(availableNow=True)
    .toTable(f"{target_schema}._admissions")
    .awaitTermination())

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import Column

_adm = (
    spark.table(f'{target_schema}._admissions')
    .withColumn('rnk', f.rank().over(Window.partitionBy("hadm_id").orderBy(col("commit_timestamp"))))
    .filter(col('rnk') == '1')
    .select('hadm_id', col('commit_timestamp'))
)

# COMMAND ----------

w = Window.partitionBy("subject_id").orderBy("admittime")

adm_features = (
    spark.table(f'{source_schema}.admissions')
    .join(_adm, 'hadm_id')
    # Find out when the patients last hospital discharge was
    .withColumn('last_discharge', f.lag(f.col('dischtime')).over(w))
    # If they don't have a recent discharge, then they are a new patient
    .withColumn('new_patient', f.when(f.col('last_discharge').isNull(), 1).otherwise(0))
    # Calculate if their most recent discharge was within 30 days
    .withColumn('IS_A_READMISSION', f.when(
        f.col('last_discharge') > f.date_trunc('dd', f.col('admittime')) - f.expr('INTERVAL 30 DAYS'), 1
    ).otherwise(0))
    # Our target variable is predicting that the NEXT admission will be a readmission
    .withColumn('30_DAY_READMISSION', f.coalesce(f.lead('IS_A_READMISSION').over(w),f.lit(0)).cast('double'))
    # How many 30 day readmissions have they had in the last 6 months, including current admission?
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
    
    .drop('last_discharge')
)

# COMMAND ----------

# If we were using spark ML, we'd use the built in one-hot encoder, but we're setting up data for an sklearn model. SKlearn onehot encoder only learns the distributed parition of data that it can see and therefore misses some classes sometimes. 
cat_cols = ['admission_type','admission_location','discharge_location','insurance','language','marital_status','race']
for cat_col in cat_cols:
    # pulling categories from non-incremental table + regex cleanup to make sql compatible
    categories = [ re.sub(r'[\s/()]', '_', ((row[cat_col] if row[cat_col] is not None else "None").lower())).replace('.', '') for row in adm_features.select(cat_col).distinct().collect()]

    # Create one-hot encoded columns for each category
    for category in categories:
        adm_features = adm_features.withColumn("cat_col"+"_"+category, f.when(col(cat_col) == category, 1).otherwise(0))

# COMMAND ----------

adm_table_name = 'admission_features'
tables = spark.table(f'{target_schema.split(".")[0]}.information_schema.tables').select('table_name').collect()
[x[0] for x in tables]
if adm_table_name not in tables:
  
  # Extract the schema from the adm_features dataframe
  schema = adm_features.schema

  # Create a new table with the same schema
  spark.sql(f"CREATE TABLE IF NOT EXISTS {target_schema}.{adm_table_name}"+" ({})".format(
      ", ".join([f"`{field.name}` {field.dataType.simpleString()}" for field in schema])
  ))

  # Add primary key constraint to the table
  spark.sql(f"ALTER TABLE {target_schema}.{adm_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

matching_cols = set(adm_features.columns) & set(spark.table(f"{target_schema}.{adm_table_name}").columns)

# COMMAND ----------

# if new categories come up, don't include them, as the model won't be trained on those features
# todo - create a dynamic system that will create a new feature
adm_features.createOrReplaceTempView('adm_features')

merge_query = f"""
MERGE INTO {target_schema}.{adm_table_name} AS target
USING adm_features AS source
ON target.subject_id = source.subject_id AND target.hadm_id = source.hadm_id
WHEN MATCHED and target.commit_timestamp < source.commit_timestamp THEN
  UPDATE SET {", ".join([f"target.`{_col}` = target.`{_col}`" for _col in matching_cols])}
WHEN NOT MATCHED THEN
  INSERT ({", ".join([f"`{_col}`" for _col in matching_cols])}) VALUES ({", ".join([f"source.`{_col}`" for _col in matching_cols])})
"""
spark.sql(merge_query)

# COMMAND ----------

patients = (spark.readStream.option("readChangeFeed", "true").table(f'{source_schema}.patients')
  .filter(col('_change_type').isin('insert', 'update_postimage'))
  .withColumnRenamed('_commit_timestamp', 'commit_timestamp')
  .drop('_change_type')
  .drop('_commit_version'))

tables = spark.table(f'{target_schema.split(".")[0]}.information_schema.tables').select('table_name').collect()

if '_patients' in [x[0] for x in tables]:
  spark.sql(f'truncate table {target_schema}._admissions')

# COMMAND ----------

catalog, schema = target_schema.split('.')

(patients.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"/Volumes/{catalog}/{schema}/streaming_checkpoints/_patients")
    .trigger(availableNow=True)
    .toTable(f"{target_schema}._patients")
    .awaitTermination())

# COMMAND ----------

from pyspark.sql import functions as f
from pyspark.sql.window import Window
from pyspark.sql import Column

_pat = (
    spark.table(f'{target_schema}._patients')
    .withColumn('rnk', f.rank().over(Window.partitionBy("subject_id").orderBy(col("commit_timestamp"))))
    .filter(col('rnk') == '1')
    .select('subject_id', col('commit_timestamp'))
)

# COMMAND ----------



# COMMAND ----------

# df.display()
# adm_features = calc_encounters_features(encounters)
# adm_features.display()
# adm_features_table = fe.create_table(
#   name=f'{target_schema}.adm_features',
#   primary_keys='Id',
#   schema=adm_features.schema,
#   description=f'Features derived from {source_schema}.encounters'
# )

# fe.write_table(
#   name=f'{target_schema}.adm_features',
#   df = adm_features,
#   mode = 'merge'
# )
# # # You wouldn't need to do this - for you you might use the f.current_date() function in spark instead
# # # see commented out code
# # current_date = encounters.select(f.max('START')).collect()[0][0]

# # filtered_adm_features = (
# #   adm_features
# #   # Taking away the last 30 days because we don't know if those patients readmitted or not
# #   .filter(col('START') > current_date - f.expr(f'INTERVAL {int(training_months_history)*30 + 30} days'))
# #   .filter(col('START') < current_date - f.expr(f'INTERVAL 30 days'))
# #   )
# # Define Patient Features logic
# def calc_pat_features(data):
#   data = ps.get_dummies(data.pandas_api(), columns=['MARITAL', 'RACE', 'ETHNICITY', 'GENDER'],dtype = 'int64').to_spark()
#   return data.select(
#     'Id',
#     'HEALTHCARE_COVERAGE',
#     'INCOME',
#     'MARITAL_D',
#     'MARITAL_M',
#     'MARITAL_S',
#     'MARITAL_W',
#     'RACE_asian',
#     'RACE_black',
#     'RACE_hawaiian',
#     'RACE_native',
#     'RACE_other',
#     'RACE_white',
#     'ETHNICITY_hispanic',
#     'ETHNICITY_nonhispanic',
#     'GENDER_F',
#     'GENDER_M'
#   )
# patients = spark.table(f'{source_schema}.patients')
# patients_features = calc_pat_features(patients)
# patients
# adm_features_table = fe.create_table(
#   name=f'{target_schema}.pat_features',
#   primary_keys='Id',
#   schema=patients_features.schema,
#   description=f'Features derived from {source_schema}.patients'
# )

# fe.write_table(
#   name=f'{target_schema}.pat_features',
#   df = patients_features,
#   mode = 'merge'
# )
# def calc_age_at_encounter(encounters, patients):
#   return (
#     encounters
#     .join(patients, patients.Id == encounters.PATIENT)
#     .withColumn("age_at_encounter", ((f.datediff(col('START'), col('BIRTHDATE'))) / 365.25))
#     .select(encounters.Id, 'age_at_encounter')
#     )

# age_at_encounter = calc_age_at_encounter(spark.table(f'{source_schema}.encounters'), spark.table(f'{source_schema}.patients'))

# customer_feature_table = fe.create_table(
#   name=f'{target_schema}.age_at_encounter',
#   primary_keys='Id',
#   schema=age_at_encounter.schema,
#   description='What age was the patient when they were admitted. Id in this table coresponds to encounters.Id'
# )

# fe.write_table(
#   name=f'{target_schema}.age_at_encounter',
#   df = age_at_encounter,
#   mode = 'merge'
# )

# COMMAND ----------


