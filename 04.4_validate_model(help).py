# Databricks notebook source
retrain_model = dbutils.jobs.taskValues.get(taskKey    = "model_monitor",
                            key        = "retrain_model",
                            default    = True,
                            debugValue = True)
if not retrain_model:
  dbutils.notebook.exit()

# COMMAND ----------

import mlflow
import pyspark.sql.functions as f
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import Window
from datetime import datetime, timedelta
import json

# COMMAND ----------

experiment_name = dbutils.jobs.taskValues.get(taskKey= "train_model", 
                            key        = "experiment_name", 
                            default    = "/Users/riley.rustad@databricks.com/hls_ml_demo_20250122", \
                            debugValue = "/Users/riley.rustad@databricks.com/hls_ml_demo_20250122")

# model_version = dbutils.jobs.taskValues.get(taskKey= "retrain_model", 
#                             key        = "model_version", 
#                             default    = 1, \
#                             debugValue = 1)

dbutils.widgets.text('model_name', 'kp_catalog.hls_ml.hls_ml_demo')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('source_schema', 'kp_catalog.mimic_incr')
source_schema = dbutils.widgets.get('source_schema')

dbutils.widgets.text('feature_schema', 'kp_catalog.hls_ml')
feature_schema = dbutils.widgets.get('feature_schema')

dbutils.widgets.text('accuracy_threshold', '.6')
accuracy_threshold = float(dbutils.widgets.get('accuracy_threshold'))

dbutils.widgets.text('demographic_accuracy_threshold', '.5')
demographic_accuracy_threshold = float(dbutils.widgets.get('demographic_accuracy_threshold'))

# dbutils.widgets.text('demographic_vars', 'RACE_asian,RACE_black,RACE_hawaiian,RACE_native,RACE_other,RACE_white,ETHNICITY_hispanic,ETHNICITY_nonhispanic,GENDER_F,GENDER_M')
# demographic_vars = dbutils.widgets.get('demographic_vars')

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')
client = mlflow.tracking.MlflowClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Current Staging Model

# COMMAND ----------

model_details = client.get_model_version_by_alias(model_name, "staged")

model_version = model_details.version

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Target Variable to Validate

# COMMAND ----------

# TODO: make a function that calculates the target variable 30 day readmission
# TODO: parameterize the number of days of train, val, test

# COMMAND ----------

admissions = spark.table(f'{source_schema}.admissions')

max_adm_date = admissions.select(f.max(f.col('admittime'))).collect()[0][0]
print(max_adm_date)

w = Window.partitionBy("subject_id").orderBy("admittime")

test = (
  admissions
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('admittime') < f.lit(max_adm_date - timedelta(days=30)))
  # Limit training data to the last 3 years
  .filter(col('admittime') > f.lit(max_adm_date - timedelta(days=60)))
  # Calculate the target variable
 .withColumn('last_discharge', f.lag(f.col('dischtime')).over(w))
  .withColumn('new_patient', f.when(f.col('last_discharge').isNull(), 1).otherwise(0))
  .withColumn('IS_A_READMISSION', f.when(
      f.col('last_discharge') > f.date_trunc('dd', f.col('admittime')) - f.expr('INTERVAL 30 DAYS'), 1
  ).otherwise(0))
  .withColumn('30_DAY_READMISSION', f.coalesce(f.lead('IS_A_READMISSION').over(w), f.lit(0)).cast("double"))
  # .select('hadm_id', 'subject_id', '30_DAY_READMISSION')
  # .orderBy(['admittime'], desc=True)
)
  
# data.select(f.max(f.col('admittime'))).collect()[0][0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Feature Store to Score Batch
# MAGIC Notice, we didn't need to rewrite our data prep ETL for inference. It's a single line of code to score data with the proper key

# COMMAND ----------

from sklearn.metrics import roc_auc_score

# COMMAND ----------

# Using score batch to get FE to do our feature look ups for us
preds = (
  fe.score_batch(model_uri=f"models:/{model_details.name}@staged", df=test)
  # .select('hadm_id','subject_id', '30_DAY_READMISSION', 'prediction')
  # choose to cache the dataframe because we use the output many times - keeps the result in memory
).cache()

# COMMAND ----------

mlflow.set_experiment(experiment_name)

# COMMAND ----------

pd_preds =preds.toPandas()

# COMMAND ----------

pd_preds_prepped = pd_preds[['hadm_id','subject_id', 
      #      'admittime', 'dischtime', 'deathtime',
      #  'admission_type', 'admit_provider_id', 'admission_location',
      #  'discharge_location', 'insurance', 'language', 'marital_status', 'race',
      #  'edregtime', 'edouttime', 'hospital_expire_flag', 
      #  'last_discharge',
       'gender_f',
       'gender_m', 'admission_type_direct_observation',
       'admission_type_eu_observation', 'admission_type_ew_emer',
       'admission_type_elective', 'admission_type_surgical_same_day_admission',
       'admission_type_observation_admit',
       'admission_type_ambulatory_observation', 'admission_type_direct_emer',
       'admission_type_urgent',
       'admission_location_internal_transfer_to_or_from_psych',
       'admission_location_procedure_site',
       'admission_location_emergency_room',
       'admission_location_physician_referral',
       'admission_location_transfer_from_skilled_nursing_facility',
       'admission_location_walk_in_self_referral',
       'admission_location_clinic_referral', 'admission_location_pacu',
       'admission_location_transfer_from_hospital',
       'admission_location_information_not_available',
       'admission_location_ambulatory_surgery_transfer', 'insurance_private',
       'insurance_other', 'insurance_medicaid', 'insurance_no_charge',
       'insurance_medicare', 'insurance_none', 'marital_status_widowed',
       'marital_status_single', 'marital_status_married',
       'marital_status_divorced', 'marital_status_none', 'age_at_admission',
       'new_patient', 'IS_A_READMISSION', '30_DAY_READMISSION', 
        
       '30_DAY_READMISSION_6_months', '30_DAY_READMISSION_12_months',
       'prev_admissions_6_months', 'prev_admissions_12_months', 
      #  'prediction'
       ]]

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_details.name}@staged")

# COMMAND ----------

preds = model.predict(pd_preds_prepped)
preds['prediction'].unique()

# COMMAND ----------



# COMMAND ----------

preds.select('30_DAY_READMISSION').distinct().display()

# COMMAND ----------

preds.select('prediction').distinct().display()

# COMMAND ----------

preds.schema

# COMMAND ----------

preds.count()

# COMMAND ----------

import mlflow

model_uri = 'models:/kp_catalog.hls_ml.hls_ml_demo/1'
model = mlflow.pyfunc.load_model(model_uri)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------

model.metadata.flavors#["python_function"]["loader_module"]

# COMMAND ----------

mlflow.models.evaluate()

# COMMAND ----------

from mlflow.models.evaluation.evaluators.classifier import _extract_predict_fn_and_prodict_proba_fn
_extract_predict_fn_and_prodict_proba_fn(model)

# COMMAND ----------

model.predict_proba()

# COMMAND ----------

with mlflow.start_run():

    # Comprehensive evaluation with one line
    result = mlflow.models.evaluate(
        # model=mlflow.pyfunc.load_model(f"models:/{model_details.name}@staged"),
        # model=f"models:/{model_details.name}@staged",
        data=pd_preds,
        targets="30_DAY_READMISSION",
        predictions="prediction",
        model_type="classifier",
        evaluators=["default"],
        # evaluator_config={"log_explainer": True},
    )

# COMMAND ----------

result.metrics

# COMMAND ----------

print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
print(f"F1 Score: {result.metrics['f1_score']:.3f}")
print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")

# COMMAND ----------

accuracy = (
  preds
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
accuracy

# COMMAND ----------

# True positive rate
sensitivity = (
  preds
  .filter(col('30_DAY_READMISSION') == 1)
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
sensitivity

# COMMAND ----------

# True Negative Rate
specificity = (
  preds
  .filter(col('30_DAY_READMISSION') == 0)
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
specificity

# COMMAND ----------

# MAGIC %md
# MAGIC So the model is 

# COMMAND ----------

#TODO: make the model use predict_proba and calculate model AUC instead (better measure for binary classifier)
if accuracy > accuracy_threshold:
  client.set_tag(model_details.run_id, key='test_accuracy', value=accuracy)
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="meets_accuracy_threshold", value=True)
else:
  print("Model does not meet mandatory accuracy threshold")
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="meets_accuracy_threshold", value=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for Model Fairness

# COMMAND ----------

demographic_vars_ = demographic_vars.split(",")
# demographic_vars

# COMMAND ----------

try:
  for demographic_var in demographic_vars_:
    
    demo_accuracy = (
      preds
      .filter(col(demographic_var) == 1)
      .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
    ).collect()[0][0]
    print(demographic_var, demo_accuracy)
    client.set_tag(model_details.run_id, key=demographic_var, value=demo_accuracy)
    client.set_model_version_tag(name=model_details.name, version=model_details.version, key=demographic_var, value=demo_accuracy > demographic_accuracy_threshold)

    client.set_model_version_tag(name=model_name, version=model_version, key="demo_test", value=True)
except KeyError:
  print("KeyError: No demographics_vars tagged with this model version.")
  client.set_model_version_tag(name=model_name, version=model_version, key="demo_test", value=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate That Model Has Necessary Metadata

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”. The model signature defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based.
# MAGIC
# MAGIC See here for more details.

# COMMAND ----------

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_details.source)
if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="has_signature", value=False)
else:
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="has_signature", value=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Description
# MAGIC If someone comes looking for this model, will the be able to reasonably figure out what it's trying to do?

# COMMAND ----------

if not model_details.description:
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="has_description", value=False)
  print("Did you forget to add a description?")
elif not len(model_details.description) > 20:
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="has_description", value=False)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="has_description", value=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Results

# COMMAND ----------

# Model tags have been updated since we created the `model_details` variable
results = client.get_model_version(name=model_details.name, version=model_details.version)
results.tags


# COMMAND ----------

# are all of the tags True?
all_true = all([bool(tag) for tag in results.tags])
all_true

# COMMAND ----------



# COMMAND ----------

host_creds = client._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()


def transition(model_name, version, stage):
  
  prod_request = {'name': model_name,
                     'version': version,
                     'stage': stage,
                     'archive_existing_versions': 'false' if stage == 'Archived' else 'true'
                    }
  response = mlflow_call_endpoint('model-versions/transition-stage', 'POST', json.dumps(prod_request))
  # This version will automatically transition the model to Prod. If you'd like manual approval, you can use the commented code below
  # response = mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
  return(response)

# COMMAND ----------

# Optional: you can also break model promotion to PROD into it's own separate notebook
if all_true:
  client.set_registered_model_alias(model_name, "production", model_details.version)
  client.delete_registered_model_alias(model_name, "staged")

  # Give the model framework the needed info
  catalog, schema, model = model_details.name.split('.')
  #TODO: come back and automate all the manual tagging
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="inference_table", value=f"kp_catalog.hls_ml.{model}_predictions")
  client.set_model_version_tag(name=model_details.name, version=model_details.version, key="outcomes_table", value=f"kp_catalog.hls_ml.{model}_outcomes")
else:
  print('Model did not qualify for production')
  client.delete_registered_model_alias(model_name, "staged")

# COMMAND ----------

model_details.name, model_details.version

# COMMAND ----------

model_details

# COMMAND ----------


