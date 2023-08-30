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

# COMMAND ----------

# MAGIC %md
# MAGIC Load data for the demo

# COMMAND ----------

# MAGIC %run ./00-setup $reset_all_data=$reset_all_data $catalog=dbdemos $db=$db $folder=$folder

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

if reset_all_data:
  dbutils.fs.rm(source_path,True)
  dbutils.fs.rm(ingest_path,True)
  spark.sql(f'drop database if exists {db} cascade')
  for dir_ in dbutils.fs.ls(landed_path):
    print(ingest_path+dir_.name)
    dbutils.fs.mkdirs(ingest_path+dir_.name)

  # todo: also drop FS tables from FS?

# COMMAND ----------

tables = [x[0].replace('/','') for x in spark.createDataFrame(dbutils.fs.ls(landed_path)).select('name').collect()]
tables

# COMMAND ----------

landed_path

# COMMAND ----------

if reset_all_data:
  encounters = spark.read.parquet(landed_path+'encounters')
  encounters.createOrReplaceTempView('encounters')

  # figure out what the last day of the dataset is, and backtrack 1 year to start the demo
  # This should give is 365 updates worth to play with before needing to update
  demo_start_encounter_date = spark.sql("""
  SELECT max(date_trunc('dd', START) - INTERVAL 365 DAY) FROM encounters
  """).collect()[0][0]
  
  incremental_encounters = encounters.filter(col('START') > demo_start_encounter_date)

  # write the most recent years data into the source path, where we'll pull incrememntals from
  (incremental_encounters
    .withColumn('encounter_date', F.date_trunc('dd', 'START'))
    .write.format('parquet')
    #TODO partition by STOP
    .partitionBy('encounter_date')
    .mode('overwrite')
    .save(source_path+'encounters')
  )

  # Write the rest of the encounters out to the ingest path
  # We'll assume that that's backfill data to be loaded
  historical_encounters = encounters.filter(col('START') <= demo_start_encounter_date)
  historical_encounters.write.mode('overwrite').parquet(ingest_path+"encounters")

# COMMAND ----------

# in this case, we're moving one days worth of data changes into encounters
if not reset_all_data:

  # Find what files exist in both source and ingest directory
  source_files = dbutils.fs.ls(source_path+'encounters')
  source_files_set = set([x.name for x in source_files if 'encounter_date' in x.name])
  ingest_files_set = set([x.name for x in dbutils.fs.ls(ingest_path+'encounters')])

  # Find what files haven't been moved yet
  # need to use builtins because we overwrote the min function with the pyspark version of "min"
  file_to_move = __builtins__.min(source_files_set - ingest_files_set)

  # Copy the next day's worth of data into the ingest directory
  for file_ in source_files:
    if file_.name == file_to_move:
      print(file_.path, ingest_path+f'encounters/{file_.name}')
      dbutils.fs.cp(file_.path, ingest_path+f'encounters/{file_.name}/', True)

# COMMAND ----------

# Need to use the wildcard, *, for it to read in the partitions and the historical data together
spark.read.parquet(ingest_path+'encounters/*').createOrReplaceTempView('encounters_incremental')

# for initial load, you have to specify the schema because the ingest table is empty
spark.read.parquet(landed_path+'/patients').createOrReplaceTempView('patients')
spark.read.schema(spark.table('patients').schema).parquet(ingest_path+'/patients').createOrReplaceTempView('patients_incremental')

df = spark.sql(
"""select * from patients
where patients.id in (select distinct PATIENT from encounters_incremental)
and patients.id not in (select patients_incremental.id from patients_incremental)"""
)

df.write.mode('append').parquet(ingest_path+'/patients')

# COMMAND ----------


def update_enc_based_table(table):
  # Need to use the wildcard, *, for it to read in the partitions and the historical data together
  spark.read.parquet(ingest_path+'encounters/*').createOrReplaceTempView('encounters_incremental')

  # for initial load, you have to specify the schema because the ingest table is empty
  spark.read.parquet(landed_path+f'/{table}').createOrReplaceTempView(f'{table}')
  spark.read.schema(spark.table(f'{table}').schema).parquet(ingest_path+f'/{table}').createOrReplaceTempView(f'{table}_incremental')

  df = spark.sql(
  f"""select * from {table}
  where {table}.ENCOUNTER in (select distinct id from encounters_incremental)
  and {table}.ENCOUNTER not in (select distinct ENCOUNTER from {table}_incremental)
  """
  ).coalesce(1)

  df.write.mode('append').parquet(ingest_path+f'/{table}')

  

for table in dbutils.fs.ls(landed_path):
  if table.name not in ['encounters/', 'patients/', 'location_ref/']:
    print(table.name)
    update_enc_based_table(table.name.replace('/',''))

# COMMAND ----------

if reset_all_data:
  # Move all static tables to ingest path
  spark.read.parquet(landed_path+'location_ref').write.mode('overwrite').parquet(ingest_path+'location_ref')
  spark.read.parquet(vocab_path+'CONCEPT').write.parquet(ingest_path+'CONCEPT')
  spark.read.parquet(vocab_path+'CONCEPT_RELATIONSHIP').write.parquet(ingest_path+'CONCEPT_RELATIONSHIP')

# COMMAND ----------

spark.read.parquet(ingest_path+'patients').display()

# COMMAND ----------


