# Databricks notebook source


# COMMAND ----------

# %pip install mlflow>=3.0 --upgrade
# dbutils.library.restartPython()

# COMMAND ----------

# TODO convert off of hyperopt

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
import pyspark.sql.functions as F
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.sql import Window
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
from ray import tune
from mlflow.entities import Dataset

# COMMAND ----------

dbutils.widgets.text('source_schema', 'kp_catalog.mimic_incr')
source_schema = dbutils.widgets.get('source_schema')

dbutils.widgets.text('target_schema', 'kp_catalog.hls_ml')
target_schema = dbutils.widgets.get('target_schema')

dbutils.widgets.text('max_evals', '50')
max_evals = int(dbutils.widgets.get('max_evals'))

dbutils.widgets.text('model_name', 'kp_catalog.hls_ml.hls_ml_demo')
model_name = dbutils.widgets.get('model_name')
model_name = model_name.split('.')[-1]

# Select the number of months of training history that you want to pull
dbutils.widgets.text('training_months_history', '24')
training_months_history = int(dbutils.widgets.get('training_months_history'))

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

admissions = spark.table(f'{source_schema}.admissions')

max_adm_date = admissions.select(f.max(f.col('admittime'))).collect()[0][0]
print(max_adm_date)

w = Window.partitionBy("subject_id").orderBy("admittime")

data = (
  admissions
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=30)))
  # # Limit training data to the last 3 years
  # .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=365*3)))
  # Calculate the target variable
  .withColumn('last_discharge', f.lag(f.col('dischtime')).over(w))
  .withColumn('new_patient', f.when(f.col('last_discharge').isNull(), 1).otherwise(0))
  .withColumn('IS_A_READMISSION', f.when(
      f.col('last_discharge') > f.date_trunc('dd', f.col('admittime')) - f.expr('INTERVAL 30 DAYS'), 1
  ).otherwise(0))
  .withColumn('30_DAY_READMISSION', f.coalesce(f.lead('IS_A_READMISSION').over(w), f.lit(0)))
  .select('hadm_id', 'subject_id', 'admittime', 'dischtime', '30_DAY_READMISSION')
  .orderBy(['admittime'], desc=True)
)
  
# data.select(f.max(f.col('admittime'))).collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train/Test Split

# COMMAND ----------

training_data = (
  data
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=90)))
  .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=training_months_history*30) - timedelta(days=90)))
  .drop('admittime')
)

validation_data = (
  data
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=60)))
  .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=90)))
  .drop('admittime')
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define What Features To Look Up

# COMMAND ----------

spark.table(f"{source_schema}.patients").select('gender').distinct().display()

# COMMAND ----------

patient_feature_lookups = [
   FeatureLookup( 
     table_name = f"{target_schema}.patient_features",
     feature_names = [
      'gender_f',
      'gender_m'],
     lookup_key = ["subject_id"],
   ),
]
 
admissions_feature_lookups = [
   FeatureLookup( 
     table_name = f"{target_schema}.admissions_features",
     feature_names = [
        "admission_type_direct_observation",
        "admission_type_eu_observation",
        "admission_type_ew_emer",
        "admission_type_elective",
        "admission_type_surgical_same_day_admission",
        "admission_type_observation_admit",
        "admission_type_ambulatory_observation",
        "admission_type_direct_emer",
        "admission_type_urgent",
        "admission_location_internal_transfer_to_or_from_psych",
        "admission_location_procedure_site",
        "admission_location_emergency_room",
        "admission_location_physician_referral",
        "admission_location_transfer_from_skilled_nursing_facility",
        "admission_location_walk_in_self_referral",
        "admission_location_clinic_referral",
        "admission_location_pacu",
        "admission_location_transfer_from_hospital",
        "admission_location_information_not_available",
        "admission_location_ambulatory_surgery_transfer",
        "insurance_private",
        "insurance_other",
        "insurance_medicaid",
        "insurance_no_charge",
        "insurance_medicare",
        "insurance_none",
        "marital_status_widowed",
        "marital_status_single",
        "marital_status_married",
        "marital_status_divorced",
        "marital_status_none",
     ],
     lookup_key = ["hadm_id"],
   ),
]

age_at_enc_feature_lookups = [
   FeatureLookup( 
     table_name = f"{target_schema}.age_at_admission",
     feature_names = ['age_at_admission'],
     lookup_key = ["hadm_id"],
   ),
]

historic_admission_feature_lookups = [
   FeatureLookup( 
     table_name = f"{target_schema}.historic_admissions_features",
     feature_names = [
        "new_patient",
        "IS_A_READMISSION",
        "30_DAY_READMISSION_6_months",
        "30_DAY_READMISSION_12_months",
        "prev_admissions_6_months",
        "prev_admissions_12_months",
     ],
     lookup_key = ["hadm_id"],
   ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Feature Store to Create Dataset Based on Lookups

# COMMAND ----------


fe = FeatureEngineeringClient()
training_set = fe.create_training_set(
  df = data,
  feature_lookups = patient_feature_lookups + admissions_feature_lookups + age_at_enc_feature_lookups + historic_admission_feature_lookups,
  label = "30_DAY_READMISSION",
  exclude_columns = ["hadm_id", "subject_id", 'dischtime', 'admittime']
)

enriched = training_set.load_df()

train = (
  enriched
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=90)))
  .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=training_months_history*30) - timedelta(days=90)))
  .drop('admittime')
).toPandas()

val = (
  enriched
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=60)))
  .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=90)))
  .drop('admittime')
).toPandas()



# COMMAND ----------

# fe = FeatureEngineeringClient()
# training_set = fe.create_training_set(
#   df = training_data,
#   feature_lookups = patient_feature_lookups + admissions_feature_lookups + age_at_enc_feature_lookups + historic_admission_feature_lookups,
#   label = "30_DAY_READMISSION",
#   exclude_columns = ["hadm_id", "subject_id", 'dischtime', 'admittime']
# )

# train = training_set.load_df().toPandas()

# validation_set = fe.create_training_set(
#   df = validation_data,
#   feature_lookups = patient_feature_lookups + admissions_feature_lookups + age_at_enc_feature_lookups + historic_admission_feature_lookups,
#   label = "30_DAY_READMISSION",
#   exclude_columns = ["hadm_id", "subject_id", 'dischtime', 'admittime']
# )

# val = validation_set.load_df().toPandas()

# COMMAND ----------

val.shape,train.shape

# COMMAND ----------

train
# train.isnull().values.any()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the MLFlow Experiement

# COMMAND ----------

max_adm_date

# COMMAND ----------

# from datetime import datetime
# demo_date = datetime.today()
# demo_date

# COMMAND ----------

#TODO: what is the prod location for an experiment
experiment_name = f"/Users/riley.rustad@databricks.com/{model_name}_{max_adm_date.strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)
target_col = "30_DAY_READMISSION"

# COMMAND ----------

experiment_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### Iterate Through Hyperparameters
# MAGIC Logging all ML Model parameters, metrics, and artifacts

# COMMAND ----------

# TODO: exclude id columns from the training set

# COMMAND ----------

import mlflow
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, SparkTrials
import mlflow.sklearn
from sklearn.metrics import roc_auc_score

# mlflow.sklearn.autolog(log_models=False)
mlflow.autolog(disable=True)

def optimize(
            #  trials, 
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
    # spark_trials = SparkTrials()
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, space, algo=tpe.suggest, 
                # trials=trials, 
                max_evals=max_evals)
    return best
  
def score(params):
  with mlflow.start_run() as training_run:

    train_dataset: Dataset = mlflow.data.from_pandas(train, name="train")
    validation_dataset: Dataset = mlflow.data.from_pandas(val, name="val")

    for key, value in params.items():
      mlflow.log_param(key, value)
    
    model = RandomForestClassifier(
      n_estimators = int(params['n_estimators']),
      max_depth = int(params['max_depth']),
      min_samples_split = int(params['min_samples_split'])
    )

    model.fit(train.drop(target_col,axis=1), train[target_col])

#     mlflow.sklearn.log_model(model, 'model')
    fe.log_model(
      model=model,
      artifact_path="model",
      flavor=mlflow.sklearn,
      training_set=training_set,
      infer_input_example=True,
      # dataset=train_dataset
      # registered_model_name="kp_catalog.hls_ml.readmissions"
    )

    preds = model.predict(val.drop(target_col,axis=1))
    score = roc_auc_score(val[target_col], preds)
    
    train_preds = model.predict(train.drop(target_col,axis=1))
    train_score = roc_auc_score(train[target_col], train_preds)

  #   mlflow.log_metrics(
  #     metrics={
  #       "rmse": rmse,
  #       "r2": r2,
  #       "mae": mae,
  #     }, 
  #   dataset=test_dataset,
  #   model_id=logged_model.model_id
  # )

    model_info = mlflow.pyfunc.load_model(f"runs:/{mlflow.active_run().info.run_id}/model")

    mlflow.log_metric('val_auc', score, model_id=model_info.model_id, dataset=validation_dataset)
    mlflow.log_metric('train_auc', train_score, model_id=model_info.model_id, dataset=train_dataset)
    # I like to compare val and train metrics so that I can measure overfitting
    mlflow.log_metric('diff', train_score - score,model_id=model_info.model_id)
  
    loss = 1 - score
  return {'loss': loss, 'status': STATUS_OK}


# COMMAND ----------

# spark_trials = SparkTrials()
best_hyperparams = optimize(
                            # trials = spark_trials,
                            max_evals
                            )

# COMMAND ----------

dbutils.jobs.taskValues.set(key= "experiment_name",value = experiment_name)
