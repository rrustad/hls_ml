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

dbutils.widgets.text('model_name', 'kp_catalog.hls_ml.hls_ml_demo')
model_name = dbutils.widgets.get('model_name')
catalog,schema,model = model_name.split('.')

experiment_name = dbutils.jobs.taskValues.get(taskKey= "train_model", 
                            key        = "experiment_name", 
                            default    = "/Users/riley.rustad@databricks.com/hls_ml_demo_20250122", \
                            debugValue = f"/Users/riley.rustad@databricks.com/hls_ml_demo_20250122")

dbutils.widgets.text('demographic_vars', '{"kp_catalog.mimic_incr.admissions.gender":"hadm_id","kp_catalog.mimic_incr.admissions.race":"hadm_id"}')
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

# COMMAND ----------

model_uri =client.search_logged_models(
  experiment_ids=[expId],
  filter_string="metrics.diff < .02",
  order_by = [
    {"field_name": "metrics.val_auc", "ascending": False}
  ])[0].model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register The Best Model

# COMMAND ----------

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

model_details

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Model With Descriptions Metatdata

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=model_details.version)

#The main model description, typically done once.
if model_details.version == 1:
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

# client.set_model_version_tag(model_name, model_details.version, "demographic_vars", demographic_vars)


# COMMAND ----------

dbutils.jobs.taskValues.set(key= "model_version",value = model_details.version)

# COMMAND ----------

model_version_details.version

# COMMAND ----------

# TODO - log model dependencies
# import mlflow.models.utils
# model_version_uri = f"models:/{model_version_details.model_id}"
# # mlflow.models.add_libraries_to_model(model_version_uri)
# mlflow.models.utils.add_libraries_to_model(model_uri=f"models:/{model_name}/{model_details.version}")

# COMMAND ----------

model_version_uri

# COMMAND ----------


