# Databricks notebook source
from databricks.feature_store import FeatureLookup
from databricks import feature_store
import mlflow
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.sql import Window
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# COMMAND ----------

dbutils.widgets.text('dbName', 'dbdemos.hls_ml_features')
dbName = dbutils.widgets.get('dbName')

dbutils.widgets.text('max_evals', '50')
max_evals = int(dbutils.widgets.get('max_evals'))

# COMMAND ----------

retrain_model = dbutils.jobs.taskValues.get(taskKey    = "model_monitor",
                            key        = "retrain_model",
                            default    = True,
                            debugValue = True)
print(retrain_model)
if not retrain_model:
  dbutils.notebook.exit()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define our Taget Variable `30_DAY_READMISSION`

# COMMAND ----------

encounters = spark.table('dbdemos.hls_ml_source.encounters')

max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]
print(max_enc_date)

windowSpec = Window.partitionBy("PATIENT").orderBy("START")

data = (
  encounters
  
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('START') < f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=30)))
  # Limit training data to the last 3 years
  .filter(col('START') > f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=365*3)))
  # We're only interested in hospitalizations - see 4.1 for ideas on additional cohort filters
  .filter(col('ENCOUNTERCLASS').isin(['emergency','inpatient','urgentcare']))
  # Calculate the target variable
  .withColumn('last_discharge', F.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
  # Calculate if their most recent discharge was within 30 days
  .withColumn('30_DAY_READMISSION', F.when(col('START').cast('timestamp').cast('long') - col('last_discharge').cast('timestamp').cast('long') < 60*60*24*30, 1).otherwise(0))
  .select('Id', 'PATIENT', 'START', 'STOP', '30_DAY_READMISSION')
  .orderBy(['START'], desc=True)
)
  
# data.select(f.max(f.col('START'))).collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train/Test Split
# MAGIC

# COMMAND ----------

training_data = (
  data
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('START') < f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=90)))
  .drop('START')
)

validation_data = (
  data
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('START') < f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=60)))
  .filter(col('START') > f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=90)))
  .drop('START')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define What Features To Look Up

# COMMAND ----------


patient_features_table = f'{dbName}.patients_features'
encounter_features_table = f'{dbName}.encounters_features'
age_at_enc_features_table = f'{dbName}.age_at_encounter'
 
patient_feature_lookups = [
   FeatureLookup( 
     table_name = patient_features_table,
     feature_names = [
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
      'GENDER_M'],
     lookup_key = ["PATIENT"],
   ),
]
 
encounter_feature_lookups = [
   FeatureLookup( 
     table_name = encounter_features_table,
     feature_names = [
        'TOTAL_CLAIM_COST',
        'new_patient',
        '30_DAY_READMISSION_6_months',
        '30_DAY_READMISSION_12_months',
        'prev_admissions_6_months',
        'prev_admissions_12_months',
        'ENCOUNTERCLASS_emergency',
        'ENCOUNTERCLASS_inpatient',
        'ENCOUNTERCLASS_urgentcare'
     ],
     lookup_key = ["Id"],
   ),
]

age_at_enc_feature_lookups = [
   FeatureLookup( 
     table_name = age_at_enc_features_table,
     feature_names = ['age_at_encounter'],
     lookup_key = ["Id"],
   ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Feature Store to Create Dataset Based on Lookups

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
training_set = fs.create_training_set(
  training_data,
  feature_lookups = patient_feature_lookups + encounter_feature_lookups + age_at_enc_feature_lookups,
  label = "30_DAY_READMISSION",
  exclude_columns = ["Id", "PATIENT", 'STOP', 'START']
)

train = training_set.load_df().toPandas()

validation_set = fs.create_training_set(
  validation_data,
  feature_lookups = patient_feature_lookups + encounter_feature_lookups + age_at_enc_feature_lookups,
  label = "30_DAY_READMISSION",
  exclude_columns = ["Id", "PATIENT", 'STOP', 'START']
)

val = validation_set.load_df().toPandas()

# COMMAND ----------

val.shape,train.shape

# COMMAND ----------

train#[train.isna()]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the MLFlow Experiement

# COMMAND ----------

#TODO: what is the prod location for an experiment
experiment_name = f"/Users/riley.rustad@databricks.com/hls_readmissions_demo_{datetime.today().strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)
target_col = "30_DAY_READMISSION"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Iterate Through Hyperparameters
# MAGIC Logging all ML Model parameters, metrics, and artifacts

# COMMAND ----------

train

# COMMAND ----------

# TODO: Add tagging of 

# COMMAND ----------

import mlflow
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, SparkTrials
import mlflow.sklearn
from sklearn.metrics import roc_auc_score

def optimize(
             #trials, 
             max_evals,
             random_state=42):
    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth':  hp.choice('max_depth', range(1, 14)),
        'min_samples_split': hp.quniform('min_samples_split', 2, 6, 1),
        'random_state': random_state
    }
#     spark_trials = SparkTrials()
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest, 
#                 trials=spark_trials, 
                max_evals=max_evals)
    return best
  
def score(params):
  with mlflow.start_run() as mlflow_run:
    for key, value in params.items():
      mlflow.log_param(key, value)
    
    model = RandomForestClassifier(
      n_estimators = int(params['n_estimators']),
      max_depth = int(params['max_depth']),
      min_samples_split = int(params['min_samples_split'])
    )

    model.fit(train.drop(target_col,axis=1), train[target_col])

    preds = model.predict(val.drop(target_col,axis=1))
    score = roc_auc_score(val[target_col], preds)
    
    train_preds = model.predict(train.drop(target_col,axis=1))
    train_score = roc_auc_score(train[target_col], train_preds)
    
    mlflow.log_metric('val_auc', score)
    mlflow.log_metric('train_auc', train_score)
    # I like to compare val and train metrics so that I can measure overfitting
    mlflow.log_metric('diff', train_score - score)
    
#     mlflow.sklearn.log_model(model, 'model')
    fs.log_model(
      model,
      artifact_path="model",
      flavor=mlflow.sklearn,
      training_set=training_set,
#       registered_model_name="taxi_example_fare_packaged"
    )
  
    loss = 1 - score
  return {'loss': loss, 'status': STATUS_OK}


# COMMAND ----------

best_hyperparams = optimize(
                            #trials,
                            max_evals
                            )

# COMMAND ----------

dbutils.jobs.taskValues.set(key= "experiment_name",value = experiment_name)
