# Databricks notebook source
retrain_model = dbutils.jobs.taskValues.get(taskKey    = "model_monitor",
                            key        = "retrain_model",
                            default    = True,
                            debugValue = True)
if not retrain_model:
  dbutils.notebook.exit()

# COMMAND ----------

import mlflow
import json
from mlflow.utils.rest_utils import http_request

# COMMAND ----------

# dbutils.widgets.text('experiment_name', '/Users/riley.rustad@databricks.com/hls_readmissions_demo')
# experiment_name = dbutils.widgets.get('experiment_name')

dbutils.widgets.text('model_path', 'model')
model_path = dbutils.widgets.get('model_path')

dbutils.widgets.text('model_name', 'hls_ml_demo')
model_name = dbutils.widgets.get('model_name')

experiment_name = dbutils.jobs.taskValues.get(taskKey= "retrain_model", 
                            key        = "experiment_name", 
                            default    = "/Users/riley.rustad@databricks.com/hls_readmissions_demo_20240125", \
                            debugValue = "/Users/riley.rustad@databricks.com/hls_readmissions_demo_20240125")

dbutils.widgets.text('demographic_vars', 'RACE_asian,RACE_black,RACE_hawaiian,RACE_native,RACE_other,RACE_white,ETHNICITY_hispanic,ETHNICITY_nonhispanic,GENDER_F,GENDER_M')
demographic_vars = dbutils.widgets.get('demographic_vars')

# COMMAND ----------

experiment_name

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look Up Best Run within Experiment
# MAGIC I can do this programmatically or through the UI

# COMMAND ----------

expId = mlflow.get_experiment_by_name(experiment_name).experiment_id

df = spark.read.format("mlflow-experiment").load(expId)
best_run_id = (
  df.orderBy('metrics.val_auc',ascending=False)
  .select('run_id')
  .limit(1)
  .collect()[0][0]
)
# or
best_run_id = mlflow.search_runs(
  experiment_ids=[expId], 
  order_by=["metrics.val_auc DESC"], 
  max_results=1, 
  #
  filter_string="status = 'FINISHED'"
  ).iloc[0]['run_id']

best_run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register The Best Model

# COMMAND ----------

model_details = mlflow.register_model(f"runs:/{best_run_id}/{model_path}", model_name)

# COMMAND ----------

model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Model With Descriptions Metatdata

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=model_details.version)

#The main model description, typically done once.
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a patient will Readmit.  It is used to update the Readmissions Dashboard in DB SQL."
)

#Gives more details on this specific model version
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using SKlearn Random forest"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Programmatically Transition Model to Staging

# COMMAND ----------

client.set_registered_model_alias(model_name, "staged", model_details.version)

# COMMAND ----------

model_details.version

# COMMAND ----------

client.set_model_version_tag(model_name, model_details.version, "demographic_vars", demographic_vars)


# COMMAND ----------

dbutils.jobs.taskValues.set(key= "model_version",value = model_details.version)
