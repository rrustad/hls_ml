# Databricks notebook source
import mlflow
import pyspark.sql.functions as f
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from databricks import feature_store
from pyspark.sql import Window
from datetime import datetime, timedelta
import json
from mlflow.utils.rest_utils import http_request

# COMMAND ----------



# COMMAND ----------

experiment_name = dbutils.jobs.taskValues.get(taskKey= "train_model", 
                            key        = "experiment_name", 
                            default    = "/Users/riley.rustad@databricks.com/hls_readmissions_demo_20230823", \
                            debugValue = "/Users/riley.rustad@databricks.com/hls_readmissions_demo_20230823")

model_version = dbutils.jobs.taskValues.get(taskKey= "train_model", 
                            key        = "model_version", 
                            default    = 1, \
                            debugValue = 1)

dbutils.widgets.text('model_name', 'hls_ml_demo')
model_name = dbutils.widgets.get('model_name')

dbutils.widgets.text('accuracy_threshold', '.6')
accuracy_threshold = float(dbutils.widgets.get('accuracy_threshold'))

dbutils.widgets.text('demographic_accuracy_threshold', '.5')
demographic_accuracy_threshold = float(dbutils.widgets.get('demographic_accuracy_threshold'))

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Current Staging Model

# COMMAND ----------

model_details = client.get_latest_versions(model_name, ['Staging'])[0]
model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Target Variable to Validate

# COMMAND ----------

encounters = spark.table('dbdemos.hls_ml_source.encounters')

max_enc_date = encounters.select(f.max(f.col('START'))).collect()[0][0]
print(max_enc_date)

windowSpec = Window.partitionBy("PATIENT").orderBy("START")

test = (
  encounters
  
  # We can't definitively say if anyone from the last 30 days has readmitted
  .filter(col('START') < f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=30)))
  # Limit training data to the last 3 years
  .filter(col('START') > f.lit(datetime.strptime(max_enc_date, '%Y-%m-%dT%H:%M:%SZ') - timedelta(days=60)))
  # We're only interested in hospitalizations - see 4.1 for ideas on additional cohort filters
  .filter(col('ENCOUNTERCLASS').isin(['emergency','inpatient','urgentcare']))
  # Calculate the target variable
  .withColumn('last_discharge', F.lag(col('STOP')).over(Window.partitionBy("PATIENT").orderBy("START")))
  # Calculate if their most recent discharge was within 30 days
  .withColumn('30_DAY_READMISSION', F.when(col('START').cast('timestamp').cast('long') - col('last_discharge').cast('timestamp').cast('long') < 60*60*24*30, 1).otherwise(0))
  .select('Id', 'PATIENT', '30_DAY_READMISSION')
  .orderBy(['START'], desc=True)
)
  
# data.select(f.max(f.col('START'))).collect()[0][0]

# COMMAND ----------

max_enc_date

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use Feature Store to Score Batch
# MAGIC Notice, we didn't need to rewrite our data prep ETL for inference. It's a single line of code to score data with the proper key

# COMMAND ----------

from sklearn.metrics import roc_auc_score

# COMMAND ----------

preds = (
  fs.score_batch(f"models:/{model_details.name}/Staging", test)
  # .select('Id', '30_DAY_READMISSION', 'prediction')
)
# preds.display()

# COMMAND ----------

accuracy = (
  preds
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
accuracy

# COMMAND ----------

sensitivity = (
  preds
  .filter(col('30_DAY_READMISSION') == 1)
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
sensitivity

# COMMAND ----------

specificity = (
  preds
  .filter(col('30_DAY_READMISSION') == 0)
  .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
).collect()[0][0]
specificity

# COMMAND ----------



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

run_info = client.get_run(run_id=model_details.run_id)
demographic_vars = run_info.data.tags['demographic_vars'].split(",")
demographic_vars

# COMMAND ----------

try:
  for demographic_var in demographic_vars:
    print(demographic_var)
    demo_accuracy = (
      preds
      .filter(col(demographic_var) == 1)
      .select(f.mean(f.when(col('30_DAY_READMISSION') == col('prediction'), 1).otherwise(0)))
    ).collect()[0][0]
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
  transition(model_name = model_name, version = model_details.version, stage = "Production")
else:
  print('Model did not qualify for production')
  transition(model_name = model_name, version = model_details.version, stage = "Archived")

# COMMAND ----------


